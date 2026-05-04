from typing import Dict, Any
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from .core import build_estimator, build_preprocessor, evaluate_run, RunResult
from .utils import (
    to_proba, fit_with_optional_sample_weight,
    group_balanced_bootstrap_indices, confusion_rates,
    ece_bin, youden_threshold,
)
from .techniques_pre import compute_reweights, local_massaging_fit_flip
from .techniques_post import (
    group_thresholds_youden, predict_with_group_thresholds,
    input_repair_standardize_by_group, apply_multiaccuracy_boost
)
from .deps import IMBLEARN_OK, FAIRLEARN_OK, LogisticRegression
from .techniques_in import run_prejudice_remover, fit_isotonic_by_group, apply_isotonic_by_group



def run_combined_pipeline(model_name, params,
                          X_tr, X_va, X_te, y_tr, y_va, y_te,
                          A_tr, A_va, A_te, protected_cols, all_df_train,
                          selected: Dict[str, bool]) -> RunResult:
    """
    Compose selected techniques into one run in this order:
      PRE  : Local Massaging -> SMOTE/Oversample -> Reweight (y,a)
      IN   : (choose at most one) {Reductions(EO), Compositional, Ensemble(K=5), Prejudice Remover}
             + optional Multicalibration (isotonic)
      POST : Input Repair -> Multiaccuracy Boost -> Youden per group -> Reject-Option Shift

    Returns a single RunResult named with the chain.
    """

    #Canonical keys (must match self.tech_vars keys exactly)
    PRE_KEYS  = ["Pre:Local Massaging", "Pre:SMOTE / Oversample", "Pre:Reweight (y,a)"]
    IN_TRAIN  = ["In:Reductions (EO)", "In:Compositional per-group", "In:Ensemble (K=5)", "In:Fairness Regularization (Prejudice Remover)"]
    IN_CAL    = "In:Multicalibration (isotonic)"
    POST_KEYS = ["Post:Input Repair", "Post:Multiaccuracy Boost", "Post:Youden per group", "Post:Reject-Option Shift"]

    #Execution plans based on selections that preserve order
    pre_plan  = [k for k in PRE_KEYS  if selected.get(k, False)]
    in_train  = [k for k in IN_TRAIN  if selected.get(k, False)][:1]  #at most one trainer
    use_mcal  = bool(selected.get(IN_CAL, False))
    post_plan = [k for k in POST_KEYS if selected.get(k, False)]

    #Readable title
    parts = [k.split(":",1)[1].strip() for k in (pre_plan + in_train)]
    if use_mcal: parts.append("Multicalibration (isotonic)")
    parts += [k.split(":",1)[1].strip() for k in post_plan]
    title = "Combined: " + (" -> ".join(parts) if parts else "(no techniques)")

    #Working copies
    Xtr, Xva, Xte = X_tr.copy(), X_va.copy(), X_te.copy() #Features
    ytr, yva, yte = y_tr.copy(), y_va.copy(), y_te.copy() #Labels
    Atr, Ava, Ate = A_tr.copy(), A_va.copy(), A_te.copy() #Group labels (intersectional)

    #---------- PRE ----------
    #We train on either: (a) preprocessed + SMOTE matrix, or (b) normal pipeline.

    #Flags indicating if we are using SMOTE or reweighting
    did_smote = False; sample_weight = None

    #Local Massaging
    #Relabels training examples to equalize positive rate across groups based on initial scores.
    if "Pre:Local Massaging" in pre_plan:
        #Build a temporary preprocessor on train/validation to approximate actual pipeline
        prep_tmp = build_preprocessor(pd.concat([Xtr, Xva]), protected_cols)

        #Fit the temporary preprocessor and concatenate the data for scoring
        Xt_tmp = prep_tmp.fit_transform(pd.concat([Xtr, Xva]))

        #Create an estimator for a temporary baseline model to get scores on training data
        est_tmp = build_estimator(model_name, params)

        #Fit on training portion only
        est_tmp.fit(Xt_tmp[:len(Xtr)], ytr)

        #Compute predicted scores on the training data
        scores_tr = to_proba(est_tmp, Xt_tmp[:len(Xtr)])

        #Apply local massaging relabeling, flipping and overwriting labels in ytr  
        ytr = pd.Series(local_massaging_fit_flip(ytr, scores_tr, Atr), index=ytr.index)

    #SMOTE / Oversample (class-balance on transformed space)
    #Flags indicating if class balances were applied
    Xt_for_fit = None; y_for_fit = None

    if "Pre:SMOTE / Oversample" in pre_plan:
        #Build a temporary preprocessor on train/validation to approximate actual pipeline
        prep_tmp = build_preprocessor(pd.concat([Xtr, Xva]), protected_cols)
        Xt_tr = prep_tmp.fit_transform(Xtr)
        yt_tr = ytr.to_numpy()

        #If imblearn is available, use SMOTE to balance classes (TODO: Reimplement manual SMOTE)
        if IMBLEARN_OK:
            try:
                from imblearn.over_sampling import SMOTE
                sampler = SMOTE()
                #Fit SMOTE to training data, and generate balanced dataset by oversampling minority class
                Xt_bal, yt_bal = sampler.fit_resample(Xt_tr, yt_tr)
                #Get SMOTE flag and variables
                did_smote, Xt_for_fit, y_for_fit = True, Xt_bal, yt_bal
            except Exception:
                did_smote = False
        #If SMOTE unavailable, train normally (or rely on sample_weight if set)

    #Reweight (y,a) 
    #Kamiran-Calders style reweighting based on joint (y,a) distribution
    if "Pre:Reweight (y,a)" in pre_plan:
        #Computes sample weights for each training instance based on:
        # w(y,a) = P(y) * P(a) / P(y,a)
        sample_weight = compute_reweights(ytr, Atr)

    #---------- IN (train classifier once) ----------
    #Preprocessor for (train+val) – used for inference consistently
    prep = build_preprocessor(pd.concat([Xtr, Xva]), protected_cols)
    prep.fit(pd.concat([Xtr, Xva]))

    #Training matrix (based off whether SMOTE was performed)
    if did_smote:
        Xfit, yfit = Xt_for_fit, y_for_fit
    else:
        Xfit, yfit = prep.transform(Xtr), ytr.to_numpy()


    trained_est = None #Contains the trained estimator if applicable
    P_val = None  #validation probabilities (needed by many post steps)
    ensemble_estimators = None

    #Helper fn: Given a trained estimator, score val and test sets
    def _score_val_and_test(est_like):
        nonlocal P_val
        P_val = to_proba(est_like, prep.transform(Xva))
        p_test = to_proba(est_like, prep.transform(Xte))
        return p_test

    #Check if an in-processing method was selected
    if in_train:
        #Only choose the first in-processing method (TODO: Fix GUI to prevent this, or extend functionality to multiple?)
        choice = in_train[0]

        #Reductions on Equalized Odds using fairlearn
        if choice == "In:Reductions (EO)" and FAIRLEARN_OK:
            from fairlearn.reductions import ExponentiatedGradient, EqualizedOdds
            #Build base estimator without fairness constraints
            base = build_estimator(model_name, params)
            #Wrap the estimator in a ExponentiatedGradient with EO constraints (modifies the gradient of the steps during training to enforce EO)
            eg = ExponentiatedGradient(estimator=base, constraints=EqualizedOdds())
            #Fit the training data using group labels as senstive features
            eg.fit(Xfit, yfit, sensitive_features=Atr)
            trained_est = eg
            #Score val and test sets
            p_test = _score_val_and_test(trained_est)

        #Compositional training: one model per group, fallback to pooled
        elif choice == "In:Compositional per-group":
            #Train one model per group (on non-SMOTE pipeline if SMOTE selected)
            #If SMOTE was selected, we ignore it here (per-group SMOTE is messy) and use standard pipeline. (TODO: Build out SMOTE based per-group function)
            
            #Transform datasets with standard preprocessor
            Xtr_t = prep.transform(Xtr)
            Xte_t = prep.transform(Xte)
            Xva_t = prep.transform(Xva)
            
            #Find unique groups
            groups = pd.Series(Atr).unique()
            models = {}

            #Train per-group models
            for g in groups:
                #Boolean mask to select each group
                m = (Atr == g).to_numpy()
                #Skip groups with less than 5 samples to avoid extreme overfitting (number arbitrary, may need to revisit)
                if m.sum() < 5:
                    continue
                #Build and fit estimator on group-specific data
                est = build_estimator(model_name, params)
                #Fit the data for this group only (Functionality with reweighting not validated)
                fit_with_optional_sample_weight(est, Xtr_t[m], ytr.to_numpy()[m], sample_weight=None)
                models[str(g)] = est

            #Generate predictions on test set using the per-group models
            P_test = np.zeros(len(Xte_t))
            for i, g in enumerate(Ate):
                est = models.get(str(g)) #Get the model for this group
                if est is None: 
                    #If no model was found for this group, fall back to a pooled model training across all groups
                    pooled = build_estimator(model_name, params)
                    fit_with_optional_sample_weight(pooled, Xtr_t, ytr.to_numpy(), sample_weight=sample_weight)
                    #Make prediction with pooled model
                    P_test[i] = to_proba(pooled, Xte_t[i:i+1])[0]
                else:
                    #Make prediction with group-specific model
                    P_test[i] = to_proba(est, Xte_t[i:i+1])[0]

            #Some post-processing requires validation probabilities; compute them now using a pooled model (TODO: Integrate per-group val scoring)
            pooled = build_estimator(model_name, params)
            fit_with_optional_sample_weight(pooled, Xtr_t, ytr.to_numpy(), sample_weight=sample_weight)
            P_val = to_proba(pooled, Xva_t)
            p_test = P_test
            trained_est = pooled  #use pooled for re-scoring if needed later

        #Ensemble of K=5 group-balanced bootstrapped models
        elif choice == "In:Ensemble (K=5)":
            K = 5 #Number of bootstrap samples (TODO: Allow user to specific K)
            preds_test, preds_val = [], []
            ensemble_estimators = []  #Store individual estimators if we want to analyze them later (TODO: Integrate into RunResult)
            #Transform datasets with standard preprocessor
            Xte_t = prep.transform(Xte)
            Xva_t = prep.transform(Xva)
            for _ in range(K):
                #Build and fit general estimator
                est = build_estimator(model_name, params)
                #Draw a bootstrap sample that is roughly group-balanced
                idx = group_balanced_bootstrap_indices(Atr.to_numpy(), size=len(Atr))
                #Fit the estimator on the bootstrap sample
                fit_with_optional_sample_weight(est, Xfit[idx], yfit[idx], sample_weight=None)
                #Score test and validation probabilities for this ensemble
                preds_test.append(to_proba(est, Xte_t))
                preds_val.append(to_proba(est, Xva_t))
                #Store the individual estimator
                ensemble_estimators.append(est)
            #Aggregate predictions by averaging across K models
            p_test = np.mean(np.vstack(preds_test), axis=0)
            P_val  = np.mean(np.vstack(preds_val),  axis=0)
            trained_est = None  #ensemble isn't a single estimator; keep None

        #Prejudice Remover (AIF360)
        elif choice == "In:Fairness Regularization (Prejudice Remover)":
            #Delegate to the PR runner; return its evaluation as the combined result.
            rr = run_prejudice_remover(model_name, params,
                                       Xtr, Xva, Xte, ytr, yva, yte,
                                       Atr, Ava, Ate, protected_cols, pd.concat([Xtr, Xva]),
                                       eta=25.0)
            return rr

        else:
            #Unknown or unavailable, use vanilla fit
            est = build_estimator(model_name, params)
            fit_with_optional_sample_weight(est, Xfit, yfit, sample_weight=sample_weight)
            trained_est = est
            p_test = _score_val_and_test(trained_est)

    else:
        #No special in-process trainer, use base model
        est = build_estimator(model_name, params)
        fit_with_optional_sample_weight(est, Xfit, yfit, sample_weight=sample_weight)
        trained_est = est
        p_test = _score_val_and_test(trained_est)

    #Optional in-process calibration layer (per-group isotonic)
    if use_mcal and P_val is not None:
        iso_map = fit_isotonic_by_group(A_va, P_val, yva.to_numpy())
        p_test = apply_isotonic_by_group(Ate, p_test, iso_map)

    #---------- POST ----------


    #Input Repair (rescore with repaired X_test)
    #Modifies test features group wise to align distributions most closely with training distribution
    if "Post:Input Repair" in post_plan:
        #Combine training and validation groups to define training distribution
        A_train_all = pd.concat([Atr, Ava], axis=0)
        #Compute repaired test features
        X_rep = input_repair_standardize_by_group(pd.concat([Xtr, Xva]), Xte, A_train_all, Ate)
        #If we have a single trained estimator, rescore on repaired features
        if trained_est is not None: 
            p_test = to_proba(trained_est, prep.transform(X_rep))
        elif ensemble_estimators is not None and len(ensemble_estimators) > 0:
            repaired_preds = [
                to_proba(est, X_rep_t)
                for est in ensemble_estimators
            ]
            p_test = np.mean(np.vstack(repaired_preds), axis=0)
        else:
            # No available estimator object to rescore repaired inputs.
            # Keep p_test unchanged, but make this explicit.
            print(
                "[Input Repair] No trained estimator or ensemble estimators available; "
                "keeping existing p_test unchanged."
            )

    #Multiaccuracy Boost (residual model on validation) -- Based on Kim 2018
    #Fits a residual model on validation data to boost probabilities using group wise signals, then applies to test data
    if "Post:Multiaccuracy Boost" in post_plan and P_val is not None:
        p_test = apply_multiaccuracy_boost(
                    X_va=Xva,
                    X_te=Xte,
                    y_va=yva,
                    A_va=Ava,
                    A_te=Ate,
                    p_val=P_val,
                    p_test=p_test,
                    prep=prep,
                    alpha=0.02,
                    eta=None,
                    max_iters=25,
                    auditor_type="ridge",
                    random_state=0,
                    eps=1e-6,
                    include_group_in_auditor=True,
                )

    #Youden per group (threshold learning)
    #Learns optimal per-group thresholds on validation data to maximize Youden's J statistic

    yhat = (p_test >= 0.5).astype(int) #Global threshold 0.5 by default
    if "Post:Youden per group" in post_plan and P_val is not None:
        #Learn per-group thresholds on validation set
        th = group_thresholds_youden(Ava, yva.to_numpy(), P_val)
        #Apply per-group thresholds to test set
        yhat = predict_with_group_thresholds(Ate, p_test, th, default=0.5)

    #Reject-Option Shift (based on validation TPRs at 0.5)
    #Slightly shifts thresholds in favor of group with lowest TPR on validation and against group with highest TPR
    if "Post:Reject-Option Shift" in post_plan and P_val is not None:
        #Validation hard predictions at global 0.5 threshold
        yhat_val = (P_val >= 0.5).astype(int)
        #Find unique groups in validation set
        groups = pd.Series(Ava).unique()
        tprs = {}

        #Compute TPR per group on validation data
        for g in groups:
            #Boolean mask for this group
            m = (Ava == g).to_numpy()
            #Compute confusion rates (TPR, FPR, etc.)
            cr = confusion_rates(yva.to_numpy()[m], yhat_val[m])
            tprs[str(g)] = cr["TPR"]
        #Identify worst and best TPR groups
        worst = min(tprs, key=lambda k: (-1 if np.isnan(tprs[k]) else tprs[k]))
        best  = max(tprs, key=lambda k: (-1 if np.isnan(tprs[k]) else tprs[k]))
        #Start at global threshold
        th = {g: 0.5 for g in groups}
        #Lower threshold for worst TPR group, raise for best TPR group
        th[str(worst)] = max(0.0, th[str(worst)] - 0.05)
        th[str(best)]  = min(1.0, th[str(best)]  + 0.05)
        #Apply adjusted thresholds to test set
        yhat = predict_with_group_thresholds(Ate, p_test, th, default=0.5)

    #---------- Evaluate ----------
    return evaluate_run(title, yte.to_numpy(), p_test, yhat, Ate)



