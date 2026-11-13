"""Aggregate utility report across all evaluation views."""
from __future__ import annotations

from .evaluation import tstr, trts, discriminative_score


def utility_report(real_train, real_test, synth_train, models=('logistic', 'rf')):
    """One dictionary collecting every utility signal for the benchmark tables:
    TSTR and TRTS per model kind, and the discriminative score. The privacy and
    fairness suites report separately; analysis combines them.
    """
    report = {
        'tstr': tstr(real_train, real_test, synth_train, model=list(models)),
        'trts': trts(real_train, real_test, synth_train, model=list(models)),
    }
    try:
        report['discriminative'] = discriminative_score(real_test[0], synth_train[0])
    except Exception as exc:
        report['discriminative'] = {'error': str(exc)}
    # Baseline: train on real, test on real, for reference.
    from .downstream import _real_only_auc
    try:
        report['baseline_real_only_auc'] = _real_only_auc(
            real_train[0], real_train[1], real_test[0], real_test[1])
    except Exception as exc:
        report['baseline_real_only_auc'] = {'error': str(exc)}
    return report
