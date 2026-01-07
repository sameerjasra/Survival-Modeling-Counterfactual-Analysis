from __future__ import annotations

import numpy as np
import pandas as pd
from lifelines.utils import concordance_index

def c_index(time: pd.Series, event: pd.Series, risk_score: np.ndarray) -> float:
    """
    Higher risk_score should mean earlier event.
    Cox model yields partial hazards; we can use them as risk_score.
    """
    return float(concordance_index(time, -risk_score, event))

def risk_from_cox(model, X: pd.DataFrame) -> np.ndarray:
    """
    CoxPHFitter has predict_partial_hazard (higher => higher risk).
    """
    ph = model.model.predict_partial_hazard(X)
    return ph.values.reshape(-1)

def ate_at_horizon(ite: np.ndarray) -> float:
    return float(np.mean(ite))
