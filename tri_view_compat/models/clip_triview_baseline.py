import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel


class CLIPTriViewBaseline(nn.Module):
    def __init__(
        self,
        clip_name="openai/clip-vit-base-patch32",
        num_classes=2,
        geo_dim=10,
        geo_hidden=128,
        fusion_hidden=512,
        dropout=0.2,
        use_full=True,
        use_crop=True,
        use_masked=True,
        use_geo=True,
        use_text=True,
    ):
        super().__init__()

        self.use_full = use_full
        self.use_crop = use_crop
        self.use_masked = use_masked
        self.use_geo = use_geo
        self.use_text = use_text

        self.clip = CLIPModel.from_pretrained(clip_name)
        for p in self.clip.parameters():
            p.requires_grad = False
        self.clip.eval()

        clip_dim = self.clip.config.projection_dim

        self.geo_mlp = nn.Sequential(
            nn.Linear(geo_dim, geo_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(geo_hidden, geo_hidden),
            nn.ReLU(inplace=True),
        )

        fusion_dim = 0
        if use_full:
            fusion_dim += clip_dim
        if use_crop:
            fusion_dim += clip_dim
        if use_masked:
            fusion_dim += clip_dim
        if use_text:
            fusion_dim += clip_dim
        if use_geo:
            fusion_dim += geo_hidden

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, fusion_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, fusion_hidden // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden // 2, num_classes),
        )

    @torch.no_grad()
    def encode_image(self, pixel_values):
        # Compatible with different transformers versions.
        feat = self.clip.get_image_features(pixel_values=pixel_values)

        # Some newer/changed versions may return BaseModelOutputWithPooling
        # instead of a projected tensor.
        if hasattr(feat, "pooler_output"):
            feat = feat.pooler_output
            if feat.shape[-1] != self.clip.config.projection_dim:
                feat = self.clip.visual_projection(feat)

        feat = F.normalize(feat, dim=-1)
        return feat

    @torch.no_grad()
    def encode_text(self, input_ids, attention_mask):
        # Compatible with different transformers versions.
        feat = self.clip.get_text_features(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # Some newer/changed versions may return BaseModelOutputWithPooling
        # instead of a projected tensor.
        if hasattr(feat, "pooler_output"):
            feat = feat.pooler_output
            if feat.shape[-1] != self.clip.config.projection_dim:
                feat = self.clip.text_projection(feat)

        feat = F.normalize(feat, dim=-1)
        return feat

    def forward(self, batch, text_inputs=None):
        feats = []

        if self.use_full:
            feats.append(self.encode_image(batch["full_image"]))

        if self.use_crop:
            feats.append(self.encode_image(batch["object_crop"]))

        if self.use_masked:
            feats.append(self.encode_image(batch["masked_context"]))

        if self.use_text:
            if text_inputs is None:
                raise ValueError("text_inputs is required when use_text=True")
            feats.append(self.encode_text(
                input_ids=text_inputs["input_ids"],
                attention_mask=text_inputs["attention_mask"],
            ))

        if self.use_geo:
            feats.append(self.geo_mlp(batch["geometry"]))

        z = torch.cat(feats, dim=-1)
        logits = self.classifier(z)
        return logits
