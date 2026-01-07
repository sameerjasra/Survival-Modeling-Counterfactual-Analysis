from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .utils import assert_columns

def _to_binary_event(s: pd.Series) -> pd.Series:
    # SUPPORT2 often has death as 0/1; sometimes {0,1} or {False,True}
    if s.dropna().isin([0, 1]).all():
        return s.astype(int)
    # if it's something else (e.g., 'dead'/'alive'), try to coerce:
    s2 = s.astype(str).str.lower().str.strip()
    if set(s2.unique()) <= {"0", "1"}:
        return s2.astype(int)
    # Heuristic: treat non-zero numeric as event
    if pd.api.types.is_numeric_dtype(s):
        return (s.fillna(0) != 0).astype(int)
    raise ValueError("Could not coerce event column to binary 0/1. Please map event_col properly.")

def _basic_clean(df: pd.DataFrame, time_col: str, event_col: str) -> pd.DataFrame:
    df = df.copy()
    df = df.dropna(subset=[time_col])  # must have time
    df[time_col] = pd.to_numeric(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col])
    df[event_col] = _to_binary_event(df[event_col])
    df = df[df[time_col] > 0]
    return df

def _impute_and_encode(df: pd.DataFrame, exclude_cols: list[str]) -> pd.DataFrame:
    """
    Simple, transparent preprocessing:
    - numeric: median imputation
    - categorical: fill 'missing' + one-hot encode
    """
    X = df.drop(columns=exclude_cols, errors="ignore").copy()

    # Separate types
    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if c not in num_cols]

    for c in num_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce")
        X[c] = X[c].fillna(X[c].median())

    for c in cat_cols:
        X[c] = X[c].astype("string").fillna("missing")

    X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
    return X

def load_support2(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return df

def prepare_survival_dataset(
    df: pd.DataFrame,
    time_col: str,
    event_col: str,
    treatment_col: str | None = None,
    id_col: str | None = None,
    min_followup_days: float = 1.0,
):
    """
    Returns:
      X: processed features (one-hot + imputed)
      T: treatment (0/1) if treatment_col provided
      y_time: durations
      y_event: event indicators (1=event occurred)
      meta: original columns kept for plotting/debug
    """
    required = [time_col, event_col]
    if id_col:
        required.append(id_col)
    assert_columns(df, required)

    df = _basic_clean(df, time_col, event_col)
    df = df[df[time_col] >= float(min_followup_days)]

    T = None
    if treatment_col is not None:
        if treatment_col not in df.columns:
            raise ValueError(
                f"treatment_col='{treatment_col}' not found. "
                f"Pick a binary exposure column (e.g., 'dnr' if present)."
            )
        # force 0/1
        t = df[treatment_col]
        if not t.dropna().isin([0, 1]).all():
            # attempt coercion
            t = pd.to_numeric(t, errors="coerce")
        if not t.dropna().isin([0, 1]).all():
            raise ValueError("Treatment column is not binary 0/1; please map a binary treatment.")
        T = t.fillna(0).astype(int)

    exclude = [time_col, event_col]
    if treatment_col:
        exclude.append(treatment_col)
    if id_col:
        exclude.append(id_col)

    X = _impute_and_encode(df, exclude_cols=exclude)
    y_time = df[time_col].astype(float)
    y_event = df[event_col].astype(int)

    meta_cols = []
    if id_col and id_col in df.columns:
        meta_cols.append(id_col)
    if treatment_col and treatment_col in df.columns:
        meta_cols.append(treatment_col)
    meta = df[meta_cols].copy() if meta_cols else pd.DataFrame(index=df.index)

    return X, T, y_time, y_event, meta

def train_test_split_survival(X, T, y_time, y_event, test_size=0.2, random_state=42):
    idx = np.arange(len(X))
    train_idx, test_idx = train_test_split(idx, test_size=test_size, random_state=random_state, stratify=y_event)

    def _sel(arr):
        return arr.iloc[train_idx], arr.iloc[test_idx]

    X_tr, X_te = _sel(X)
    ytime_tr, ytime_te = _sel(y_time)
    yevent_tr, yevent_te = _sel(y_event)

    if T is None:
        T_tr = T_te = None
    else:
        T_tr, T_te = _sel(T)

    return (X_tr, T_tr, ytime_tr, yevent_tr), (X_te, T_te, ytime_te, yevent_te)
