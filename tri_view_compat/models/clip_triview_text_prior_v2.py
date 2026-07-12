import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dims, out_dim, dropout=0.1):
        super().__init__()
        dims = [in_dim] + list(hidden_dims)
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-1], out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class CLIPTriViewTextPriorV2(nn.Module):
    """
    V2: Explicit Prior-Text Compatibility.

    Key feature:
      p_replacement = P(replacement_object | masked_context)

    The model uses:
      - tri-view CLIP features
      - replacement object text feature
      - masked-context object prior
      - explicit compatibility scalars:
          p_replacement, log_p_replacement, max_prior_prob,
          margin=max_prior_prob-p_replacement, entropy, ratio
    """

    def __init__(
        self,
        clip_name,
        num_classes=2,
        num_prior_classes=80,
        use_full=True,
        use_crop=True,
        use_masked=True,
        use_text=True,
        use_geo=False,
        prior_dim=128,
        scalar_dim=32,
        geo_dim=128,
        dropout=0.1,
        freeze_clip=True,
    ):
        super().__init__()

        self.use_full = use_full
        self.use_crop = use_crop
        self.use_masked = use_masked
        self.use_text = use_text
        self.use_geo = use_geo
        self.freeze_clip = freeze_clip
        self.num_prior_classes = num_prior_classes

        try:
            self.clip = CLIPModel.from_pretrained(clip_name, use_safetensors=True)
        except Exception:
            self.clip = CLIPModel.from_pretrained(clip_name)

        if freeze_clip:
            for p in self.clip.parameters():
                p.requires_grad = False
            self.clip.eval()

        self.clip_dim = self.clip.config.projection_dim

        if use_geo:
            self.geo_mlp = MLP(10, [128], geo_dim, dropout=dropout)
        else:
            self.geo_mlp = None
            geo_dim = 0

        self.prior_head = MLP(
            self.clip_dim,
            [512, 256],
            num_prior_classes,
            dropout=dropout,
        )

        self.prior_proj = MLP(
            num_prior_classes,
            [256],
            prior_dim,
            dropout=dropout,
        )

        self.text_prior_proj = MLP(
            self.clip_dim,
            [256],
            prior_dim,
            dropout=dropout,
        )

        self.scalar_mlp = MLP(
            6,
            [64],
            scalar_dim,
            dropout=dropout,
        )

        compat_dim = prior_dim * 4 + scalar_dim

        fusion_dim = 0
        if use_full:
            fusion_dim += self.clip_dim
        if use_crop:
            fusion_dim += self.clip_dim
        if use_masked:
            fusion_dim += self.clip_dim
        if use_text:
            fusion_dim += self.clip_dim
        if use_geo:
            fusion_dim += geo_dim

        fusion_dim += compat_dim

        self.classifier = MLP(
            fusion_dim,
            [512, 256],
            num_classes,
            dropout=dropout,
        )

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_clip:
            self.clip.eval()
        return self

    @torch.no_grad()
    def encode_image(self, pixel_values):
        self.clip.eval()
        out = self.clip.vision_model(pixel_values=pixel_values)
        pooled = out.pooler_output
        feat = self.clip.visual_projection(pooled)
        feat = F.normalize(feat, dim=-1)
        return feat

    @torch.no_grad()
    def encode_text(self, input_ids, attention_mask):
        self.clip.eval()
        out = self.clip.text_model(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.pooler_output
        feat = self.clip.text_projection(pooled)
        feat = F.normalize(feat, dim=-1)
        return feat

    def forward(
        self,
        full_image=None,
        object_crop=None,
        masked_context=None,
        input_ids=None,
        attention_mask=None,
        geometry=None,
        replacement_label=None,
    ):
        feats = []

        if self.use_full:
            full_feat = self.encode_image(full_image)
            feats.append(full_feat)

        if self.use_crop:
            crop_feat = self.encode_image(object_crop)
            feats.append(crop_feat)

        # Prior branch always uses masked context.
        masked_feat = self.encode_image(masked_context)

        if self.use_masked:
            feats.append(masked_feat)

        if self.use_text:
            text_feat = self.encode_text(input_ids, attention_mask)
            feats.append(text_feat)
        else:
            text_feat = torch.zeros(
                masked_feat.shape[0],
                self.clip_dim,
                device=masked_feat.device,
                dtype=masked_feat.dtype,
            )

        if self.use_geo:
            geo_feat = self.geo_mlp(geometry)
            feats.append(geo_feat)

        prior_logits = self.prior_head(masked_feat)
        prior_prob = torch.softmax(prior_logits, dim=-1)

        if replacement_label is None:
            replacement_label = torch.zeros(
                prior_prob.shape[0],
                device=prior_prob.device,
                dtype=torch.long,
            )

        replacement_label = replacement_label.long().clamp(0, self.num_prior_classes - 1)
        rep_prob = prior_prob.gather(1, replacement_label.view(-1, 1))

        eps = 1e-8
        max_prob = prior_prob.max(dim=1, keepdim=True).values
        log_rep_prob = torch.log(rep_prob + eps)
        margin = max_prob - rep_prob
        entropy = -(prior_prob * torch.log(prior_prob + eps)).sum(dim=1, keepdim=True)
        entropy = entropy / math.log(self.num_prior_classes)
        ratio = rep_prob / (max_prob + eps)

        scalar_feat = torch.cat(
            [
                rep_prob,
                log_rep_prob,
                max_prob,
                margin,
                entropy,
                ratio,
            ],
            dim=1,
        )
        scalar_feat = self.scalar_mlp(scalar_feat)

        expected_feat = self.prior_proj(prior_prob)
        inserted_feat = self.text_prior_proj(text_feat)

        compat_feat = torch.cat(
            [
                expected_feat,
                inserted_feat,
                expected_feat * inserted_feat,
                torch.abs(expected_feat - inserted_feat),
                scalar_feat,
            ],
            dim=-1,
        )

        feats.append(compat_feat)

        fusion = torch.cat(feats, dim=-1)
        logits = self.classifier(fusion)

        return {
            "logits": logits,
            "prior_logits": prior_logits,
            "rep_prob": rep_prob.detach(),
            "max_prior_prob": max_prob.detach(),
            "prior_entropy": entropy.detach(),
        }
