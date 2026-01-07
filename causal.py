from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .survival_models import CoxSurvivalModel

class PropensityModel:
    """
    Logistic regression propensity score e(x)=P(T=1|X).
    """
    def __init__(self, C: float = 1.0):
        self.clf = LogisticRegression(max_iter=2000, C=C, n_jobs=None)
        self._fitted = False

    def fit(self, X: pd.DataFrame, T: pd.Series):
        self.clf.fit(X.values, T.values)
        self._fitted = True
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("PropensityModel not fitted.")
        return self.clf.predict_proba(X.values)[:, 1]

def ipw_weights(T: pd.Series, e: np.ndarray, clip: float = 0.01) -> np.ndarray:
    """
    Stabilized IPW weights (simple):
      w = T/e + (1-T)/(1-e)
    """
    eps = 0.02  # start with 0.02 for stability
    e = np.clip(e, eps, 1 - eps)
    t = T.values.astype(int)
    w = t / e + (1 - t) / (1 - e)
    return w

class TLearnerSurvivalITE:
    """
    Two outcome models:
      - model1: trained on treated
      - model0: trained on control
    Optional IPW inside each arm to reduce confounding impact.
    """
    def __init__(self, penalizer=0.01, l1_ratio=0.0, use_ipw=True):
        self.model1 = CoxSurvivalModel(penalizer=penalizer, l1_ratio=l1_ratio)
        self.model0 = CoxSurvivalModel(penalizer=penalizer, l1_ratio=l1_ratio)
        
        print("model1 fitted:", getattr(self.model1, "_fitted", None))
        print("model0 fitted:", getattr(self.model0, "_fitted", None))

        self.use_ipw = bool(use_ipw)
        self.propensity = PropensityModel()
        self._fitted = False

    def fit(self, X: pd.DataFrame, T: pd.Series, time: pd.Series, event: pd.Series):
        # Fit propensity on full data (for IPW)
        self.propensity.fit(X, T)
        e = self.propensity.predict_proba(X)

        # Split by treatment
        mask1 = (T == 1)
        mask0 = (T == 0)

        X1, t1, e1 = X[mask1], time[mask1], event[mask1]
        X0, t0, e0 = X[mask0], time[mask0], event[mask0]

        if self.use_ipw:
            w = ipw_weights(T, e)
            w1 = w[mask1.values] if hasattr(mask1, "values") else w[mask1]
            w0 = w[mask0.values] if hasattr(mask0, "values") else w[mask0]
        else:
            w1 = w0 = None

        self.model1.fit(X1, t1, e1, sample_weight=w1)
        self.model0.fit(X0, t0, e0, sample_weight=w0)

        self._fitted = True
        return self

    def predict_ite_survival_prob(self, X: pd.DataFrame, horizon_days: int) -> np.ndarray:
        """
        ITE_tau(x) = S1(tau|x) - S0(tau|x)
        """
        if not self._fitted:
            raise RuntimeError("TLearnerSurvivalITE not fitted.")
        s1 = self.model1.predict_survival_prob(X, horizon_days=horizon_days)
        s0 = self.model0.predict_survival_prob(X, horizon_days=horizon_days)
        return s1 - s0

    def predict_counterfactual_survival(self, X: pd.DataFrame, horizon_days: int):
        if not self._fitted:
            raise RuntimeError("TLearnerSurvivalITE not fitted.")
        s1 = self.model1.predict_survival_prob(X, horizon_days=horizon_days)
        s0 = self.model0.predict_survival_prob(X, horizon_days=horizon_days)
        return s0, s1
