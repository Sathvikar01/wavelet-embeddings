"""Phase-4 baselines that aggregate every predictor's correlation numbers."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from evaluation.task_loss import AblationEffect
from pruning.registry import PREDICTOR_NAMES, predict_wavelet
from experiments.predictive_validation import (
    PredictorValidationReport, validate_predictor,
)


def validate_all_predictors(*args, **kwargs) -> Dict[str, PredictorValidationReport]:
    """Validate every predictor, sharing one ablation sweep.

    The per-head ablation effects are predictor-independent, so they are
    computed once (unless the caller supplies ``cached_effects``) and
    reused for every predictor -- a ~7x speed-up over naive looping.

    Positional args mirror :func:`validate_predictor`:
    ``model, model_key, tokenizer, sentences, rows, extra``.
    """
    out: Dict[str, PredictorValidationReport] = {}
    cached: Optional[List[AblationEffect]] = kwargs.pop("cached_effects", None)
    if cached is None:
        from experiments.predictive_validation import _per_head_validation
        model, model_key, tokenizer, sentences, rows, extra = args
        all_heads = [(int(r["layer"]), int(r["head"])) for r in rows]
        from evaluation.task_loss import run_model
        orig_runs = run_model(model, model_key, sentences, tokenizer,
                               device=kwargs.get("device"))
        cached = _per_head_validation(
            model, model_key, tokenizer, sentences, all_heads,
            orig_runs, device=kwargs.get("device"),
            ablation_mode=kwargs.get("ablation_mode", "zero"),
        )
    for name in PREDICTOR_NAMES:
        out[name] = validate_predictor(*args, predictor_name=name,
                                        cached_effects=cached, **kwargs)
    return out
