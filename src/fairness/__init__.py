"""Fairness stress tests: subgroup privacy and utility gaps."""
from src.fairness.stress import (
    subgroup_utility_gap,
    subgroup_privacy_risk,
    privacy_utility_tradeoff,
    fairness_report,
)
from src.fairness.metrics import fairness_summary

__all__ = [
    'subgroup_utility_gap',
    'subgroup_privacy_risk',
    'privacy_utility_tradeoff',
    'fairness_report',
    'fairness_summary',
]
