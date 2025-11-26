from typing import Dict

import numpy as np
import pandas as pd

from core import RunResult, build_estimator, build_preprocessor, evaluate_run
from utils import (
    youden_threshold, to_proba, confusion_rates,input_repair_standardize_by_group
)
from deps import LogisticRegression


def group_thresholds_youden(groups: pd.Series, y_val: np.ndarray, p_val: np.ndarray) -> Dict[str, float]:
    """
    Compute group-specific decision thresholds using Youden's J statistic.

    For each unique group label in `groups`, this function:
      1. Selects the validation examples belonging to that group.
      2. Calls `youden_threshold` on that group's labels and probabilities to
         find the probability cutoff that maximizes:
             J = TPR - FPR
      3. Stores this optimal threshold in a dictionary keyed by the group label
         (as a string).

    Parameters
    ----------
    groups : pd.Series
        Group labels for each validation example (e.g., intersectional A_va).
    y_val : np.ndarray
        Binary ground-truth labels (0/1) for the validation set.
    p_val : np.ndarray
        Predicted probabilities for the validation set (same order as y_val).

    Returns
    -------
    Dict[str, float]
        Mapping from group label (stringified) to its Youden-optimal threshold.
    """
    #Dictionary to hold group-specific thresholds
    th = {}

    #Iterate over each unique group
    for g in np.unique(groups):
        #Boolean mask for this group
        m = (groups==g)
        #Skip if no members in this group
        if m.sum()==0: continue
        #Compute and store Youden-optimal threshold for this group
        th[str(g)] = youden_threshold(y_val[m], p_val[m])
    return th

def predict_with_group_thresholds(groups: pd.Series, p: np.ndarray, thresholds: Dict[str,float], default: float=0.5) -> np.ndarray:
    """
    Convert probabilities into predictions using group-specific thresholds.

    For each group:
      - Look up its threshold from `thresholds` (if missing, use `default`).
      - Apply that threshold to all probability scores for examples in that group.

    Parameters
    ----------
    groups : pd.Series
        Group labels for each example in the set to be predicted (e.g., A_te).
    p : np.ndarray
        Predicted probabilities for the corresponding examples.
    thresholds : Dict[str, float]
        Mapping from group label (as string) to its decision threshold.
    default : float, optional
        Default threshold to use if a group's threshold is not found in the
        dictionary, by default 0.5.

    Returns
    -------
    np.ndarray
        Binary predictions (0/1) of the same shape as `p`, after applying
        group-specific thresholds.
    """
    #Initialize prediction array
    yhat = np.zeros_like(p, dtype=int)
    #Iterate over each group
    for g in np.unique(groups):
        #Get this group's threshold (or default if missing)
        t = thresholds.get(str(g), default)
        #Boolean mask for this group
        m = (groups==g)
        #Apply threshold to this group's probabilities
        yhat[m] = (p[m] >= t).astype(int)
    return yhat


def run_group_youden_postproc(model_name, params, X_tr, X_va, X_te, y_tr, y_va, y_te, A_tr, A_va, A_te, protected_cols, all_df_train):
    '''
    Post-processing: per-group Youden-optimal thresholds.

    Workflow:
      1. Train a base classifier on the training set with a shared preprocessor.
      2. On the validation set, get predicted probabilities p_val.
      3. For each group, compute the Youden-optimal threshold on (y_val, p_val).
      4. On the test set, get predicted probabilities p_test from the same model.
      5. Convert p_test to predictions using the group-specific thresholds.

    This does not alter the model itself, only the decision thresholds per group.
    '''
    #Build preprocessor on train/val
    prep = build_preprocessor(pd.concat([X_tr,X_va]), protected_cols)
    #Fit preprocessor on train/val
    prep.fit(pd.concat([X_tr,X_va]))
    #Create the base estimator
    clf = build_estimator(model_name, params)
    #Fit base estimator on training data
    clf.fit(prep.transform(X_tr), y_tr)
    #Compute predicted probabilities on validation set
    p_val = to_proba(clf, prep.transform(X_va))
    #Compute group-wise Youden-optimal thresholds on validation set
    th = group_thresholds_youden(A_va, y_va.to_numpy(), p_val)
    #Compute predicted probabilities on test set
    p_test = to_proba(clf, prep.transform(X_te))
    #Apply group-specific thresholds to test predictions to get final predictions
    yhat = predict_with_group_thresholds(A_te, p_test, th, default=0.5)
    return evaluate_run("Post: Youden per group", y_te.to_numpy(), p_test, yhat, A_te)

def run_multiaccuracy_boost(model_name, params, X_tr, X_va, X_te, y_tr, y_va, y_te, A_tr, A_va, A_te, protected_cols, all_df_train):
    """
    Post-processing: multiaccuracy-style residual model (based on Kim et al. 2018).

    Idea:
      - First, train a *base* model on the training data.
      - On the validation data, compute base probabilities p_val.
      - Construct a new feature space Z on validation, consisting of:
          [ base_prob, 1{group=g1}, 1{group=g2}, ..., 1{group=gK} ]
      - Train a small logistic regression (residual learner) on Z to predict y_va.
      - On the test data, build Z_test analogously and use the residual model
        to produce adjusted probabilities p_adj.
      - Classify using p_adj with a 0.5 threshold.

    This allows the residual model to correct systematic miscalibration or
    mispredictions that correlate with group membership and base probabilities.
    """
    #Build and fit preprocessor on train/val
    prep = build_preprocessor(pd.concat([X_tr,X_va]), protected_cols)
    prep.fit(pd.concat([X_tr,X_va]))
    
    #Build and fit base estimator on training data
    base = build_estimator(model_name, params)
    base.fit(prep.transform(X_tr), y_tr)
    
    #Build residual features on validation, features = [base_prob, group one-hots]
    #Base predicted probabilities on validation
    p_val = to_proba(base, prep.transform(X_va))
    #One-hot encode groups in validation
    G = pd.get_dummies(A_va, drop_first=False)
    #Construct feature matrix Z
    Z = np.column_stack([p_val] + [G[c].to_numpy() for c in G.columns])

    #Train a LR on (Z, y_va) to learn residual corrections
    #This LR learns to map (base_prob + groups) to true label, correcting systematic residuals
    resid = LogisticRegression(max_iter=200)
    resid.fit(Z, y_va)

    #Apply residual correction on test
    #Base probabilities for test data
    p_test = to_proba(base, prep.transform(X_te))
    #One hot encode test groups, align columns with training one-hots
    Gt = pd.get_dummies(A_te, drop_first=False).reindex(columns=G.columns, fill_value=0)
    Zt = np.column_stack([p_test] + [Gt[c].to_numpy() for c in G.columns]) #Feature matrix
    #Residual model adjusted probabilities
    p_adj = resid.predict_proba(Zt)[:,1]
    yhat = (p_adj >= 0.5).astype(int) #Hard predictions at 0.5 threshold
    return evaluate_run("Post: Multiaccuracy Boost", y_te.to_numpy(), p_adj, yhat, A_te)

def run_reject_option_shift(model_name, params, X_tr, X_va, X_te, y_tr, y_va, y_te, A_tr, A_va, A_te, protected_cols, all_df_train):
    """
    Post-processing: simple reject-option-style threshold shifting.

    Workflow:
      1. Train a base classifier on the training set.
      2. Get probabilities p_val on the validation set.
      3. At a fixed threshold 0.5, compute group-wise TPRs on validation.
      4. Identify:
           - group with lowest TPR ("worst")
           - group with highest TPR ("best")
      5. On test:
           - Start with a threshold of 0.5 for all groups.
           - Decrease threshold for worst group by 0.05 (down to a minimum of 0.0).
           - Increase threshold for best group by 0.05 (up to a maximum of 1.0).
           - Use these group-specific thresholds for final predictions.

    Intuition:
      - Groups with lower TPR (more false negatives) get a slightly more lenient
        threshold (more positives).
      - Groups with higher TPR get a slightly stricter threshold.
    """
    
    #simple boundary widening in favor of group with lowest TPR on val
    #build and fit preprocessor on train/val
    prep = build_preprocessor(pd.concat([X_tr,X_va]), protected_cols)
    prep.fit(pd.concat([X_tr,X_va]))
    
    #Build base estimator
    clf = build_estimator(model_name, params)
    #Fit base estimator on training data
    clf.fit(prep.transform(X_tr), y_tr)
    #Predict probabilities on validation set
    p_val = to_proba(clf, prep.transform(X_va))

    #Compute group-wise TPRs at global threshold 0.5
    yhat_val = (p_val>=0.5).astype(int) #Hard predictions at 0.5
    groups = pd.Series(A_va).unique() #Find unique groups in validation set
    tprs = {} #Dict to map TPRs to group

    #Iterate over each group to compute TPR
    for g in groups:
        #Boolean mask for this group
        m = (A_va==g).to_numpy()
        #Compute confusion rates (TPR, FPR, etc.)
        cr = confusion_rates(y_va.to_numpy()[m], yhat_val[m])
        #Store TPR for this group
        tprs[str(g)] = cr["TPR"]
    #Identify the worst group by TPR
    worst = min(tprs, key=lambda k: (tprs[k] if not np.isnan(tprs[k]) else -1))
    #Predict probabilities on test set
    p_test = to_proba(clf, prep.transform(X_te))
    #Identify the best group by TPR
    best = max(tprs, key=lambda k: (tprs[k] if not np.isnan(tprs[k]) else -1))
    #Start with global threshold of 0.5
    th = {g:0.5 for g in groups}
    #Adjust thresholds: lower for worst TPR group, raise for best TPR group
    th[str(worst)] = max(0.0, th[str(worst)]-0.05)
    th[str(best)]  = min(1.0, th[str(best)]+0.05)
    #Apply adjusted thresholds to test set
    yhat = predict_with_group_thresholds(A_te, p_test, th, default=0.5)
    return evaluate_run("Post: Reject-Option Shift", y_te.to_numpy(), p_test, yhat, A_te)

def run_input_repair(model_name, params, X_tr, X_va, X_te, y_tr, y_va, y_te, A_tr, A_va, A_te, protected_cols, all_df_train):
    '''
    Post-processing: input repair via per-group standardization alignment.

    Idea:
      - Train a base classifier on the original (unrepaired) training data.
      - At test time, "repair" the test inputs so that, for each group, the
        test distribution is standardized to align with that group's training
        distribution (z-alignment).
      - Then feed the repaired test inputs into the same trained model.

    This approach changes only the test features (post-hoc), not the trained
    model parameters, and attempts to mitigate distributional shifts or group
    disparities in feature scaling.
    '''
    #Build and fit preprocessor on train/val
    prep = build_preprocessor(pd.concat([X_tr,X_va]), protected_cols)
    prep.fit(pd.concat([X_tr,X_va]))
    #Build base estimator
    clf = build_estimator(model_name, params)
    #Fit base estimator on training data
    clf.fit(prep.transform(X_tr), y_tr)
    #Combine train and val for input repair fitting
    A_train_all = pd.concat([A_tr, A_va], axis=0)
    #Repair the rest features X_te by aligning them with the train/val distribution for each group
    X_rep = input_repair_standardize_by_group(pd.concat([X_tr, X_va]), X_te, A_train_all, A_te)
    #Predict on repaired test inputs
    p = to_proba(clf, prep.transform(X_rep))
    yhat = (p>=0.5).astype(int) #Hard predictions at 0.5 threshold
    return evaluate_run("Post: Input Repair (z-align)", y_te.to_numpy(), p, yhat, A_te)