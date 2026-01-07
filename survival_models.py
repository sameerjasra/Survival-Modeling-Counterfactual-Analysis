from __future__ import annotations

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

class CoxSurvivalModel:
    def __init__(self, penalizer: float = 0.01, l1_ratio: float = 0.0):
        self.model = CoxPHFitter(penalizer=penalizer, l1_ratio=l1_ratio)
        self._fitted = False

    def fit(self, X, time, event, sample_weight=None):
        df = X.copy()
        df["_time"] = pd.Series(time).values
        df["_event"] = pd.Series(event).values

    # Add weights BEFORE calling lifelines
        if sample_weight is not None:
            df["_w"] = np.asarray(sample_weight).astype(float)
            weights_col = "_w"
        else:
            weights_col = None

    # Fit CoxPH
        self.model.fit(
            df,
            duration_col="_time",
            event_col="_event",
            weights_col=weights_col,
        )

    # IMPORTANT: mark fitted
        self._fitted = True
        return self


    def predict_survival_curve(self, X, times):
        if not self._fitted:
            raise RuntimeError("Model not fitted.")
        return self.model.predict_survival_function(X, times=times)

    def predict_survival_prob(self, X, horizon_days: int):
        times = np.array([float(horizon_days)])
        sf = self.predict_survival_curve(X, times=times)
        return sf.loc[times[0]].values
