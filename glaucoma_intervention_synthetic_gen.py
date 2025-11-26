#!/usr/bin/env python3
#!/usr/bin/env python3
#!/usr/bin/env python3
"""
Synthetic dataset for 'need for glaucoma intervention' with feature-driven outcome
+ realistic noise (no degenerate TPR=1).

Outcome generation:
  1) Build a linear predictor from clinical features (and optional small group effects).
  2) Calibrate an intercept 'b0' so mean(sigmoid((linear + b0)/temperature)) ~= desired prevalence.
  3) Add Gaussian noise on the log-odds (aleatoric noise).
  4) Convert to probabilities with temperature scaling and sample Bernoulli.
  5) Optional small label-flip noise to mimic miscoding.

Protected: Race (White/Black/Asian), Gender (Male/Female) [imbalanced].
"""

from __future__ import annotations
import numpy as np
import pandas as pd

# -------------------------- configuration --------------------------

N = 10_000
SEED = 42

# Imbalanced protected attributes
RACE_P   = {"White": 0.70, "Black": 0.20, "Asian": 0.10}
GENDER_P = {"Male": 0.60, "Female": 0.40}

# Outcome/prevalence + noise controls
DESIRED_PREVALENCE   = 0.18   # target mean of outcome
TEMPERATURE          = 1.0    # >1 softens probs (less extreme); <1 sharpens
LOGIT_NOISE_SD       = 0.75   # Gaussian noise on the linear predictor (aleatoric)
LABEL_FLIP_RATE      = 0.02   # small symmetric label noise after sampling

# Group effects in risk? (small offsets so you can analyze fairness)
GROUP_EFFECTS_IN_OUTCOME = True

# Output
OUT_CSV = "synthetic_glaucoma_intervention.csv"

# -------------------------- helpers --------------------------------

rng = np.random.default_rng(SEED)

def clip(a, low=None, high=None):
    if low is not None:
        a = np.maximum(a, low)
    if high is not None:
        a = np.minimum(a, high)
    return a

def logistic(x):
    return 1.0 / (1.0 + np.exp(-x))

def sample_categorical(choices_dict, size):
    keys = list(choices_dict.keys())
    p = np.array([choices_dict[k] for k in keys])
    return rng.choice(keys, size=size, p=p)

def calibrate_intercept(linear_no_b0: np.ndarray, target_prev: float, temperature: float,
                        tol: float = 1e-6, max_iter: int = 100) -> float:
    """
    Find b0 such that mean(sigmoid((linear_no_b0 + b0)/temperature)) ~= target_prev.
    Bisection on b0 over a wide range of logits.
    """
    lo, hi = -20.0, 20.0
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        p = logistic((linear_no_b0 + mid) / temperature)
        m = p.mean()
        if abs(m - target_prev) < tol:
            return float(mid)
        if m < target_prev:
            lo = mid
        else:
            hi = mid
    return float(0.5 * (lo + hi))

# -------------------------- protected attrs -------------------------

race   = sample_categorical(RACE_P, N)
gender = sample_categorical(GENDER_P, N)

# -------------------------- base physiologic ------------------------

mean_sbp = clip(rng.normal(loc=130, scale=15, size=N), 90, 220)
mean_dbp = clip(rng.normal(loc=80,  scale=10, size=N), 50, 130)

sbp_drop = np.abs(rng.normal(loc=15, scale=8, size=N))
dbp_drop = np.abs(rng.normal(loc=10, scale=6, size=N))

min_sbp = clip(mean_sbp - sbp_drop, 70, None)
min_dbp = clip(mean_dbp - dbp_drop, 40, None)

# -------------------------- utilization -----------------------------

zero_inflation = rng.binomial(1, 0.35, size=N)
days_contact = np.where(
    zero_inflation == 1,
    rng.integers(low=0, high=2, size=N),        # 0–1 days
    rng.poisson(lam=6, size=N)                  # average ~6 days
).astype(int)

# -------------------------- conditions ------------------------------

dementia_logit   = -4.2 + 0.08 * np.minimum(days_contact, 30)
metastatic_logit = -5.2 + 0.05 * np.minimum(days_contact, 30)

dementia           = rng.binomial(1, logistic(dementia_logit))
metastatic_disease = rng.binomial(1, logistic(metastatic_logit))

# -------------------------- medications -----------------------------

antihyperlipidemic_pr = logistic(-0.5 + 0.02*(mean_sbp-130) + 0.015*np.maximum(days_contact-4, 0))
antihyperlipidemic_med = rng.binomial(1, antihyperlipidemic_pr)

calcium_blocker_pr = logistic(-1.0 + 0.018*(mean_sbp-130) + 0.01*np.maximum(days_contact-4, 0))
calcium_blocker_med = rng.binomial(1, calcium_blocker_pr)

anticoagulant_pr = logistic(-2.2 + 0.05*np.minimum(days_contact, 25) + 0.6*metastatic_disease + 0.4*dementia)
anticoagulant_med = rng.binomial(1, anticoagulant_pr)

nonopioid_analgesic_pr = logistic(-0.9 + 0.04*np.minimum(days_contact, 20))
nonopioid_analgesic_med = rng.binomial(1, nonopioid_analgesic_pr)

is_female = (gender == "Female").astype(int)
antidepressant_pr = logistic(-1.9 + 0.5*is_female + 0.035*np.minimum(days_contact, 20))
antidepressant_med = rng.binomial(1, antidepressant_pr)

macrolide_antibiotic_pr = logistic(-2.5 + 0.03*np.minimum(days_contact, 15))
macrolide_antibiotic_med = rng.binomial(1, macrolide_antibiotic_pr)

cold_cough_med_pr = logistic(-2.2 + 0.04*np.minimum(days_contact, 15))
cold_cough_med = rng.binomial(1, cold_cough_med_pr)

ophthalmic_med_pr = logistic(-1.6 + 0.045*np.minimum(days_contact, 20))
ophthalmic_med = rng.binomial(1, ophthalmic_med_pr)

# -------------------------- outcome with noise ----------------------

# Strong clinical signal (same structure as before)
linear_clinical = (
    1.8*ophthalmic_med
    + 0.6*(days_contact >= 12).astype(int)
    + 0.35*calcium_blocker_med
    + 0.30*anticoagulant_med
    + 0.25*nonopioid_analgesic_med
    + 0.25*antidepressant_med
    + 0.15*macrolide_antibiotic_med
    + 0.10*cold_cough_med
    + 0.60*dementia
    + 0.90*metastatic_disease
    - 0.008*(mean_sbp - 130.0)
    - 0.006*(min_dbp - 70.0)
)

if GROUP_EFFECTS_IN_OUTCOME:
    is_black = (race == "Black").astype(int)
    is_asian = (race == "Asian").astype(int)
    is_male  = (gender == "Male").astype(int)
    linear_group = 0.30*is_black + 0.10*is_asian - 0.10*(1 - is_male) + 0.10*(is_black * is_male)
else:
    linear_group = 0.0

linear_no_b0 = linear_clinical + linear_group

# 1) Calibrate intercept to hit desired prevalence in expectation
b0 = calibrate_intercept(linear_no_b0, DESIRED_PREVALENCE, TEMPERATURE)

# 2) Add aleatoric noise on log-odds (Gaussian)
eps = rng.normal(loc=0.0, scale=LOGIT_NOISE_SD, size=N)

# 3) Compute probabilities with temperature scaling
logit = (linear_no_b0 + b0 + eps) / TEMPERATURE
prob  = logistic(logit)

# 4) Sample labels from Bernoulli
y = rng.binomial(1, prob)

# 5) Optional symmetric label-flip noise
if LABEL_FLIP_RATE > 0:
    flip = rng.binomial(1, LABEL_FLIP_RATE, size=N)
    y = np.where(flip == 1, 1 - y, y)

# -------------------------- assemble frame --------------------------

df = pd.DataFrame({
    "Race": race,
    "Gender": gender,
    "ophthalmic_med": ophthalmic_med,
    "min_sbp": np.round(min_sbp, 1),
    "mean_sbp": np.round(mean_sbp, 1),
    "nonopioid_analgesic_med": nonopioid_analgesic_med,
    "antihyperlipidemic_med": antihyperlipidemic_med,
    "days_contact": days_contact,
    "calcium_blocker_med": calcium_blocker_med,
    "macrolide_antibiotic_med": macrolide_antibiotic_med,
    "anticoagulant_med": anticoagulant_med,
    "cold_cough_med": cold_cough_med,
    "min_dbp": np.round(min_dbp, 1),
    "dementia": dementia,
    "antidepressant_med": antidepressant_med,
    "metastatic_disease": metastatic_disease,
    "risk_score": prob,  # probability after noise & scaling; useful for debugging
    "glaucoma_intervention": y,
})

# -------------------------- summary + save --------------------------

print(df.head(10))
prev = df["glaucoma_intervention"].mean()
print(f"\nTarget prevalence={DESIRED_PREVALENCE:.3f} | Achieved={prev:.3f}")
print(f"Temperature={TEMPERATURE}, Logit noise SD={LOGIT_NOISE_SD}, Label flip={LABEL_FLIP_RATE}")

race_dist = df["Race"].value_counts(normalize=True).round(3).to_dict()
gender_dist = df["Gender"].value_counts(normalize=True).round(3).to_dict()
print("Race distribution:", race_dist)
print("Gender distribution:", gender_dist)

df.to_csv(OUT_CSV, index=False)
print(f"Saved: {OUT_CSV}")
