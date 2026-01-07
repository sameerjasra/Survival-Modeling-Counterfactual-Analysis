from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_overlap(propensity_scores: np.ndarray, T: pd.Series, title="Propensity overlap"):
    t = T.values.astype(int)
    plt.figure()
    plt.hist(propensity_scores[t==0], bins=30, alpha=0.7, label="T=0")
    plt.hist(propensity_scores[t==1], bins=30, alpha=0.7, label="T=1")
    plt.xlabel("e(x) = P(T=1|X)")
    plt.ylabel("count")
    plt.title(title)
    plt.legend()
    plt.tight_layout()

def plot_ite_hist(ite: np.ndarray, title="ITE at horizon"):
    plt.figure()
    plt.hist(ite, bins=40, alpha=0.85)
    plt.xlabel("S1(τ|x) - S0(τ|x)")
    plt.ylabel("count")
    plt.title(title)
    plt.tight_layout()

def plot_individual_counterfactual(s0: float, s1: float, horizon_days: int, title="Counterfactual survival"):
    plt.figure()
    plt.bar(["do(T=0)", "do(T=1)"], [s0, s1])
    plt.ylabel(f"P(T > {horizon_days} days)")
    plt.title(title)
    plt.tight_layout()
