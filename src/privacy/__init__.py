"""Privacy attack suite: membership and attribute inference against
synthetic EHR / waveform generators. See README - RQ1/RQ2."""

from .membership_inference import shadow_model_attack, threshold_attack, evaluate_attack
from .attribute_inference import attribute_inference_attack, attribute_lift
from .metrics import make_membership_splits, privacy_report

__all__ = [
    "shadow_model_attack",
    "threshold_attack",
    "evaluate_attack",
    "attribute_inference_attack",
    "attribute_lift",
    "make_membership_splits",
    "privacy_report",
]
