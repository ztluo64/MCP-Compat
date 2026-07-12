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


class CLIPTriViewTextPrior(nn.Module):
    """
    Tri-view + Text + Masked-context Object Prior.

    Frozen CLIP encodes:
      - full image
      - object crop
      - masked context
      - replacement object text

    Prior branch:
      - masked context feature -> 80-way expected object prior
      - prior distribution -> expected-object feature
      - replacement text feature -> inserted-object feature
      - compatibility feature = [expected, inserted, expected*inserted, |expected-inserted|]
      - final classifier uses both tri-view features and compatibility feature
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

        try:
            self.clip = CLIPModel.from_pretrained(clip_name, use_safetensors=True)
        except Exception:
            self.clip = CLIPModel.from_pretrained(clip_name)

        if freeze_clip:
            for p in self.clip.parameters():
                p.requires_grad = False
            self.clip.eval()

        self.clip_dim = self.clip.config.projection_dim

        # Optional geometry encoder.
        if use_geo:
            self.geo_mlp = MLP(10, [128], geo_dim, dropout=dropout)
        else:
            self.geo_mlp = None
            geo_dim = 0

        # Prior branch: masked context -> expected object distribution.
        self.prior_head = MLP(
            self.clip_dim,
            [512, 256],
            num_prior_classes,
            dropout=dropout,
        )

        # Turn 80-way prior distribution into a compact expected-object feature.
        self.prior_proj = MLP(
            num_prior_classes,
            [256],
            prior_dim,
            dropout=dropout,
        )

        # Project replacement object text feature into the same prior space.
        self.text_prior_proj = MLP(
            self.clip_dim,
            [256],
            prior_dim,
            dropout=dropout,
        )

        # Compatibility feature:
        # expected prior, inserted text, element-wise product, absolute difference.
        compat_dim = prior_dim * 4

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
        out = self.clip.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
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
    ):
        feats = []

        full_feat = None
        crop_feat = None
        masked_feat = None
        text_feat = None

        if self.use_full:
            full_feat = self.encode_image(full_image)
            feats.append(full_feat)

        if self.use_crop:
            crop_feat = self.encode_image(object_crop)
            feats.append(crop_feat)

        # Prior branch always needs masked context.
        masked_feat = self.encode_image(masked_context)

        if self.use_masked:
            feats.append(masked_feat)

        if self.use_text:
            text_feat = self.encode_text(input_ids, attention_mask)
            feats.append(text_feat)
        else:
            # If text is disabled, use zeros so compatibility branch still has a stable shape.
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

        expected_feat = self.prior_proj(prior_prob)
        inserted_feat = self.text_prior_proj(text_feat)

        compat_feat = torch.cat(
            [
                expected_feat,
                inserted_feat,
                expected_feat * inserted_feat,
                torch.abs(expected_feat - inserted_feat),
            ],
            dim=-1,
        )

        feats.append(compat_feat)

        fusion = torch.cat(feats, dim=-1)
        logits = self.classifier(fusion)

        return {
            "logits": logits,
            "prior_logits": prior_logits,
        }
