"""
Threshold-calibrated evaluation for Explicit-Only ablation.

Internally we reuse the prior_v2 evaluation path because:
    - it uses COinCOTriViewDatasetV2
    - it provides replacement_label to the model
    - it performs validation-based threshold calibration

The actual model class is replaced with:
    CLIPTriViewTextPriorExplicitOnly
"""

import sys

from tri_view_compat.models.clip_triview_text_prior_explicit_only import (
    CLIPTriViewTextPriorExplicitOnly,
)

import tri_view_compat.tools.eval_threshold_calibrated as evaluator


# Replace V2 model with Explicit-Only model.
evaluator.CLIPTriViewTextPriorV2 = CLIPTriViewTextPriorExplicitOnly


if __name__ == "__main__":
    # Reuse the prior_v2 data/evaluation path.
    if "--model_type" not in sys.argv:
        sys.argv.extend(
            [
                "--model_type",
                "prior_v2",
            ]
        )

    evaluator.main()
