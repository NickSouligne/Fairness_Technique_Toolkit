# fairness_tool/utils.py
import inspect
import math
from typing import Any, Dict, List
import numpy as np
import pandas as pd

from deps import (
    accuracy_score, roc_auc_score, average_precision_score, f1_score,
    brier_score_loss, IsotonicRegression,
)


def estimator_accepts_sample_weight(estimator) -> bool:
    """
    Check whether an estimator's .fit method accepts a 'sample_weight' argument.

    Uses Python's introspection to inspect the function signature of estimator.fit
    and see if 'sample_weight' is one of the parameters. Safe-guards against
    estimators that don't define a normal signature or raise errors on inspection.
    """
    try:
        sig = inspect.signature(estimator.fit)
        return "sample_weight" in sig.parameters
    except (TypeError, ValueError):
        return False


def fit_with_optional_sample_weight(estimator, X, y, sample_weight=None):
    """
    Fit an estimator, using sample_weight if possible; otherwise, approximate
    weighting via bootstrap resampling.

    Logic:
      - If no sample_weight is provided, just call estimator.fit(X, y).
      - If sample_weight is provided and the estimator supports it, pass it through.
      - If sample_weight is provided but the estimator does NOT support it,
        emulate the effect by drawing a weighted bootstrap sample of the data
        and fitting on that resampled subset.
    """
    #Case 1: No sample weights, do a normal fit
    if sample_weight is None: 
        return estimator.fit(X, y)
    
    #Case 2: Estimator supports sample weights, pass them through
    if estimator_accepts_sample_weight(estimator):
        return estimator.fit(X, y, sample_weight=sample_weight)
    
    #Case 3: Estimator does not support sample weights, do weighted bootstrap

    #Convert weights to a non-negative array
    w = np.asarray(sample_weight, dtype=float)
    w = np.clip(w, 1e-12, None) #Avoid zero or negative weights
    #Turn weights into a probability distribution over samples
    p = w / w.sum()
    n = len(y) #Number of samples
    #Draw bootstrap sample indices according to weights
    idx = np.random.choice(n, size=n, replace=True, p=p)
    return estimator.fit(X[idx], np.asarray(y)[idx])


def _fmt(x):
    """
    Format a scalar as a string:
      - Return 'NA' if the value is NaN.
      - Otherwise, format as a floating-point number with 3 decimals.
    """
    return "NA" if pd.isna(x) else f"{x:.3f}"


def _fmt_delta(curr, base, *, invert=False):
    """
    Format a difference (delta) between curr and base.

    Parameters
    ----------
    curr : float
        Current value.
    base : float
        Baseline value to compare against.
    invert : bool, default False
        If True, the delta is computed as -(curr - base), effectively flipping
        the sign (useful when lower-is-better metrics are being compared).

    Returns
    -------
    str
        "+0.123", "-0.456", or "NA" if curr or base is NaN.
    """
    if pd.isna(curr) or pd.isna(base):
        return "NA"
    
    d = (curr - base) #Raw difference
    if invert: #Invert sign for lower-is-better metrics
        d = -d
    return f"{d:+.3f}" #Format with sign and 3 decimals


def coerce_value(ptype, raw, choices=None):
    """
    Coerce a raw (often string) value from UI/config into an appropriate type.

    Parameters
    ----------
    ptype : type or str
        Target type or a special label:
          - bool, int, float, str
          - "choice" for enumerated options.
    raw : Any
        Raw input value (often string, may be None).
    choices : list, optional
        Allowed set of values when ptype == "choice".

    Returns
    -------
    Any
        Value converted to the requested type, or None if appropriate.
    """
    if ptype == bool:
        return bool(raw)
    if ptype == "choice":
        if raw in (None, ""):
            return None
        if choices and raw not in choices:
            raise ValueError(f"Value '{raw}' not in {choices}.")
        return raw
    if ptype == int:
        return None if raw in ("", "None") else int(raw)
    if ptype == float:
        return None if raw in ("", "None") else float(raw)
    if ptype == str:
        return None if raw in ("", "None") else str(raw)
    return raw


def eval_tuple(s):
    """
    Parse a string representation of a tuple of ints into an actual tuple.

    Example inputs:
      "1, 2, 3"     -> (1, 2, 3)
      "(1, 2, 3)"   -> (1, 2, 3)
      "" or None    -> None
      "  5 ,  "     -> (5,)

    Ignores empty parts and strips whitespace around commas.
    """
    if s is None or s == "":
        return None
    text = str(s).strip()
    if text.startswith("(") and text.endswith(")"): #remove parentheses
        text = text[1:-1]
    if text == "":
        return None
    parts = [p.strip() for p in text.split(",")] #split commans and strip whitespace
    return tuple(int(p) for p in parts if p != "")


def to_proba(model, X):
    """
    Convert model outputs into probabilities for the positive class.

    Logic:
      - If model has predict_proba:
          * If 2 columns, return the probability of class 1.
          * If more, return the last column (assumed positive/target class).
      - Else if model has decision_function:
          * Apply logistic transform to map scores to [0,1].
      - Else:
          * Use model.predict(X) and cast to float (assumed already probs or 0/1).

    This allows a uniform interface when computing metrics.
    """
    if hasattr(model, "predict_proba"): #prefer predict_proba if available
        p = model.predict_proba(X)
        if p.shape[1] == 2:
            return p[:, 1] #Standard binary classification: col 1 is P(y=1)
        return p[:, -1]
    if hasattr(model, "decision_function"): #second option: decision_function uses logistic transform
        z = model.decision_function(X)
        return 1.0 / (1.0 + np.exp(-z)) #Map logits to probabilities via sigmoid
    return model.predict(X).astype(float)


def ece_bin(y_true, y_prob, n_bins=10):
    """
    Compute Expected Calibration Error (ECE) using fixed-width bins in [0,1].

    ECE definition:
      - Partition the predicted probabilities into n_bins bins.
      - For each bin b, compute:
            acc_b  = mean(y_true in bin b)
            conf_b = mean(y_prob in bin b)
            w_b    = fraction of samples in bin b
      - ECE = sum_b w_b * |acc_b - conf_b|

    Parameters
    ----------
    y_true : array-like
        True binary labels.
    y_prob : array-like
        Predicted probabilities for the positive class.
    n_bins : int, default 10
        Number of probability bins.

    Returns
    -------
    float
        Estimated ECE in [0,1].
    """
    #Convert inputs to numpy arrays
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    #Bin edges evenly spaced in [0,1]
    bins = np.linspace(0, 1, n_bins + 1)
    #Digitize probabilities into bin indices [0, n_bins-1]
    idx = np.digitize(y_prob, bins) - 1
    ece = 0.0
    #Loop over each bin and accumulate weighted absolute error
    for b in range(n_bins):
        #Boolean mask for samples in bin b
        m = idx == b
        if not np.any(m): #skip empty bins
            continue
        #Accuracy within bin: fraction of true positives
        acc = y_true[m].mean()
        #Confidence within bin: mean predicted probability
        conf = y_prob[m].mean()
        #Compute ece contribution from this bin
        ece += (m.mean()) * abs(acc - conf)
    return float(ece)


def group_key(df: pd.DataFrame, protected_cols: List[str]) -> pd.Series:
    """
    Collapse one or more protected columns into a single intersectional group key.

    If protected_cols is empty, return a constant "ALL" group key.

    Example:
      protected_cols = ["race", "sex"]
      race = "White", sex = "Female" -> "White|Female"
    """
    if len(protected_cols) == 0:
        return pd.Series(["ALL"] * len(df), index=df.index)
    return df[protected_cols].astype(str).agg("|".join, axis=1)


def safe_auroc(y, p):
    """
    Safely compute AUROC, returning NaN if it cannot be computed.

    Conditions:
      - If only one class is present in y, AUROC is undefined -> return NaN.
      - If roc_auc_score raises any exception, trap it and return NaN.
    """
    try:
        if len(np.unique(y)) < 2:
            return np.nan
        return roc_auc_score(y, p)
    except Exception:
        return np.nan


def safe_auprc(y, p):
    """
    Safely compute AUPRC, returning NaN if it cannot be computed.

    Similar to safe_auroc:
      - Require both classes present in y.
      - Catch any exceptions and return NaN on failure.
    """
    try:
        if len(np.unique(y)) < 2:
            return np.nan
        return average_precision_score(y, p)
    except Exception:
        return np.nan


def youden_threshold(y, p):
    """
    Find the threshold t in [0,1] that maximizes Youden's J statistic:

        J(t) = TPR(t) + TNR(t) - 1

    Procedure:
      - Sort unique predicted probabilities.
      - For each candidate t, threshold p >= t to get predictions.
      - Compute TP, TN, FP, FN, then TPR and TNR.
      - Keep track of the t that yields the highest J(t).

    Returns
    -------
    float
        Best threshold according to Youden's J. Defaults to 0.5 if no
        valid threshold found (e.g., degenerate cases).
    """
    #Convert inputs to numpy arrays
    y = np.asarray(y)
    p = np.asarray(p)
    #Sort thresholds by predicted probabilities
    order = np.argsort(p) 
    p_sorted = p[order]


    best_j = -1 #Initialize best J statistic
    best_t = 0.5 #Default threshold if none found

    #Evaluate J statistic at each unique predicted probability
    for t in np.unique(p_sorted):
        pred = (p >= t).astype(int) #Predictions at threshold t

        #Compute confusion matrix components
        tp = ((pred == 1) & (y == 1)).sum()
        tn = ((pred == 0) & (y == 0)).sum()
        fp = ((pred == 1) & (y == 0)).sum()
        fn = ((pred == 0) & (y == 1)).sum()
        if tp + fn == 0 or tn + fp == 0: #No positive or negative ground truth samples, skip this group
            continue
        #True positive rate (sensitivity) and true negative rate (specificity)
        tpr = tp / (tp + fn)
        tnr = tn / (tn + fp)
        #Youden's J statistic
        j = tpr + tnr - 1
        #Update best threshold if J is improved
        if j > best_j:
            best_j = j
            best_t = float(t)
    return float(best_t)


def confusion_rates(y, yhat):
    """
    Compute core confusion-derived rates given true labels and predictions.

    Metrics:
      - TPR = TP / (TP + FN)
      - FPR = FP / (FP + TN)
      - PPV = TP / (TP + FP)
      - NPV = TN / (TN + FN)
      - PPR = mean(predicted positive) = Pr(yhat=1)
      - n   = number of samples

    Returns
    -------
    dict
        {"TPR": ..., "FPR": ..., "PPV": ..., "NPV": ..., "PPR": ..., "n": ...}
        with NaNs when denominators are zero.
    """
    #Confusion Matrix counts
    tp = ((yhat == 1) & (y == 1)).sum()
    tn = ((yhat == 0) & (y == 0)).sum()
    fp = ((yhat == 1) & (y == 0)).sum()
    fn = ((yhat == 0) & (y == 1)).sum()
    #Compute rates with safe-guards against division by zero
    tpr = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    fpr = fp / (fp + tn) if (fp + tn) > 0 else np.nan
    ppv = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    npv = tn / (tn + fn) if (tn + fn) > 0 else np.nan
    #Positive prediction rate: fraction prediction as positive
    ppr = (yhat == 1).mean()
    return dict(TPR=tpr, FPR=fpr, PPV=ppv, NPV=npv, PPR=ppr, n=int(len(y)))


def metrics_block(y, p, yhat):
    """
    Compute a standard block of overall performance metrics.

    Includes:
      - ACC: Accuracy
      - AUROC: Area under ROC curve (safe)
      - AUPRC: Area under PR curve (safe)
      - F1: F1 score (NaN if predictions are constant)
      - Brier: Brier score loss (calibration error)
      - ECE: Expected Calibration Error (10 bins)

    Parameters
    ----------
    y : array-like
        True labels.
    p : array-like
        Predicted probabilities for positive class.
    yhat : array-like
        Binary predictions.

    Returns
    -------
    dict
        Mapping metric name -> value.
    """
    return dict(
        ACC=accuracy_score(y, yhat),
        AUROC=safe_auroc(y, p),
        AUPRC=safe_auprc(y, p),
        F1=f1_score(y, yhat) if len(np.unique(yhat)) > 1 else np.nan,
        Brier=brier_score_loss(y, p),
        ECE=ece_bin(y, p, n_bins=10),
    )


def macro_gaps(group_stats: pd.DataFrame, cols=("PPR", "TPR", "FPR")):
    """
    Compute macro group fairness gaps given per-group statistics.

    For each metric column c in cols:
      - c_diff = max_c - min_c  over groups with non-NaN values.

    Additionally computes:
      - EO_diff = max(TPR_diff, FPR_diff), a crude equalized-odds gap summary.

    Parameters
    ----------
    group_stats : pd.DataFrame
        DataFrame where each row is a group and columns include metrics in `cols`.
    cols : tuple of str
        Metric columns for which to compute gaps.

    Returns
    -------
    dict
        {
          "PPR_diff": ...,
          "TPR_diff": ...,
          "FPR_diff": ...,
          "EO_diff":  ...
        }
    """
    out: Dict[str, Any] = {}
    for c in cols: #Compute range (max - min) for each requested metrics across groups
        vals = group_stats[c].dropna()
        out[f"{c}_diff"] = float(vals.max() - vals.min()) if len(vals) > 0 else np.nan
    out["EO_diff"] = float(max(out.get("TPR_diff", np.nan), out.get("FPR_diff", np.nan)))
    return out


def group_balanced_bootstrap_indices(a_train: np.ndarray, size: int) -> np.ndarray:
    """
    Draw bootstrap indices with roughly equal representation from each group.

    Logic:
      - Let G be the set of unique groups in a_train.
      - Compute per = max(1, size // len(G)), the number of samples to draw
        per group initially.
      - For each group g:
          * Draw 'per' indices with replacement from that group's pool.
      - Concatenate all per-group draws to get idx.
      - If we still have fewer than 'size' indices (due to rounding),
        draw additional indices from the entire training set (uniformly).

    Parameters
    ----------
    a_train : np.ndarray
        Array of group labels for each training sample.
    size : int
        Desired total number of bootstrap samples.

    Returns
    -------
    np.ndarray
        Array of indices of length 'size', suitable for indexing X and y.
    """
    #Get unique group labels
    groups = pd.Series(a_train).unique()
    per = max(1, size // len(groups)) #Samples per group
    idxs = []

    #For each group, draw 'per' samples with replacement
    for g in groups:
        pool = np.where(a_train == g)[0] #Indices of this group
        if len(pool) == 0: #skip empty groups
            continue
        #Randomly sample
        take = np.random.choice(pool, size=per, replace=True)
        idxs.append(take)
    #Concatenate per-group samples
    idx = np.concatenate(idxs)
    #If fewer indices than expected, draw extra samples 
    if len(idx) < size:
        extra = np.random.choice(len(a_train), size=size - len(idx), replace=True)
        idx = np.concatenate([idx, extra])
    return idx


def input_repair_standardize_by_group(X_train_df: pd.DataFrame, X_test_df: pd.DataFrame, a_train: pd.Series, a_test: pd.Series) -> pd.DataFrame:
    """
    Standardize each test group relative to the *global* training distribution.

    High-level idea (post-processing):
      - Compute global mean and std for each numeric feature using the training set.
      - For each group g in the test set:
          * Replace X_test rows for group g with
                (X_test[g] - global_mean) / global_std
        i.e., z-scores relative to the overall training distribution.

    Notes:
      - This function does *not* use per-group stats in training, only the
        global stats computed across all training data.
      - Features with zero std in training are given std=1 to avoid division
        by zero (they become zeroed out).

    This is not well validated; use with caution.
    """
    #z-score each numeric feature per group to the global (train) mean/std
    #Identify numeric features in training data
    num_cols = X_train_df.select_dtypes(include=[np.number]).columns
    #Global mean of each numeric features
    glob_mean = X_train_df[num_cols].mean()
    #Global standard deviation of each numeric feature
    glob_std  = X_train_df[num_cols].std().replace(0, 1.0)
    #Create a copy to avoid modifying original
    X_rep = X_test_df.copy()

    #Loop through each group in the test set
    for g in pd.Series(a_test).unique():
        m = (a_test==g) #Boolean mask to identify groups
        #Standardize the groups rows so they are expressed as z-scores w.r.t the training distribution
        X_rep.loc[m, num_cols] = (X_rep.loc[m, num_cols] - glob_mean) / glob_std
    return X_rep