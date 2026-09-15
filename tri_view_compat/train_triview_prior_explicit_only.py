"""
Train Explicit-Only ablation.

This script intentionally reuses the complete training/evaluation
protocol from train_triview_prior_v2.py and replaces ONLY the model:

    Full:
        Prior + Latent + Explicit

    This ablation:
        Prior + Explicit
        NO Latent

Therefore data splits, optimizer, class balancing, checkpoint
selection, metrics, and prior loss are exactly aligned with V2.
"""

from tri_view_compat.models.clip_triview_text_prior_explicit_only import (
    CLIPTriViewTextPriorExplicitOnly,
)

import tri_view_compat.train_triview_prior_v2 as trainer


# train_triview_prior_v2.main() resolves this symbol at runtime.
# Replace it with our explicit-only model while keeping every
# other part of the protocol unchanged.
trainer.CLIPTriViewTextPriorV2 = CLIPTriViewTextPriorExplicitOnly


if __name__ == "__main__":
    trainer.main()
