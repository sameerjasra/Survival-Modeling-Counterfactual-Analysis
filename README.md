# Survival-Modeling-Counterfactual-Analysis

This project implements a compact survival + counterfactual analysis workflow on the SUPPORT2 ICU dataset:
1) Baseline survival modeling (Cox) and risk visualization   **(01_survival_baseline.ipynb)**
2) Counterfactual survival at a fixed horizon (180 days) using a survival T-learner with IPW **(02_counterfactual_survival_ite.ipynb)**

The objective is to move beyond static risk estimation and toward decision-aware modeling, enabling the analysis of how patient outcomes may differ under alternative treatment scenarios.The work integrates classical survival analysis with causal inference techniques to demonstrate how longitudinal patient data can support individualized outcome estimation and exploratory treatment effect analysis in real-world medical datasets.

## Project Highlights

1. End-to-end survival analysis pipeline using real-world ICU data
2. Extension of survival models to counterfactual and causal settings
3. Estimation of individual and average treatment effects on survival
4. Emphasis on interpretability, uncertainty awareness and transparency
5. Reproducible notebooks suitable for research and teaching purposes

## Key Features

- Baseline survival modeling using Cox proportional hazards models
- Visualization of patient-specific survival curves and risk stratification
- Counterfactual survival estimation at clinically relevant horizon
- Survival T-learner framework with optional inverse probability weighting
- Propensity score diagnostics to assess treatment overlap and confounding
- Modular Python package structure for reuse and extension

##  Methodology  

The analysis is conducted in two stages. First, a baseline survival model is trained using Cox proportional hazards regression to estimate time-to-event outcomes and patient-level risk trajectories. Model performance is evaluated using concordance metrics and survival curves are visualized to highlight heterogeneity in patient risk.

In the second stage, survival modeling is extended into a causal inference framework by treating treatment assignment as a counterfactual intervention. A T-learner approach is used to train separate survival models for treated and untreated groups, enabling estimation of individual treatment effects (ITE) on survival probability at a fixed time horizon. To partially address confounding in observational data, inverse probability weighting (IPW) is applied based on estimated propensity scores and overlap diagnostics are performed to assess the reliability of causal estimates.

##  Future Work
- Incorporation of time-varying treatments and covariates
- Extension to Marginal Structural Models (MSMs) and dynamic treatment regimes
- Integration of Bayesian survival models for improved uncertainty quantification
- Exploration of multi-event and competing risk models
- Evaluation of reinforcement learning approaches for sequential treatment optimization

## Author  

**Sameer Kumar Jasra, PhD**  
Machine Learning Specialist | Researcher in Applied AI  
✉️ sameerjasra@gmail.com | 🌐 [LinkedIn](https://linkedin.com/in/sameerjasra)
