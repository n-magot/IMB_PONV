# -*- coding: utf-8 -*-
"""
FindIMB: Bayesian model averaging over subsets of Markov Boundary,
combining observational (Do) and experimental (De) data.

Author: Konstantina Lelova

What this script does
---------------------
1. Runs a pruned forward search over covariate subsets Z and, for each Z,
   scores two competing hypotheses:
       H_Z^c     : the observational and experimental distributions
                   P(Y | T, Z) coincide, so Do and De can be pooled,
       H_Z^c_bar : they differ, so only De is informative.
   It then computes the posterior over (Z, hypothesis).
2. Predicts P(Y | do(T), Z) on held-out experimental data by Bayesian model
   averaging, and compares it with three baselines learned from De only,
   Do only, and Do + De pooled.
3. Reports log-loss, ECE, AUC and DEU, with bootstrap CIs and paired
   bootstrap tests against the algorithm.

Data requirements
-----------------
- Two CSV files (observational and experimental) with the SAME columns.
- All variables discrete (discretize continuous variables beforehand).
- Treatment coded 0/1. Outcome coded 0/1, where Y = 0 is the GOOD outcome:
  ECE, AUC and DEU all treat Y = 0 as the positive class.
- Every column other than the treatment and the outcome is used as a
  candidate covariate. Drop identifiers, dates, etc. before running.
- Rows with a missing value in the variables of a subset are skipped when
  counting that subset.
"""

import itertools
import os
import time
from datetime import datetime
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy.special import logsumexp, gammaln, comb
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import KFold


# ===========================================================================
# USER INPUT
# ===========================================================================

OBS_DATA_PATH = "observational_data.csv"   # observational data (Do)
EXP_DATA_PATH = "experimental_data.csv"    # experimental data (De)
OUTPUT_DIR    = "results"

treatment = 'T'          # name of the treatment column (0/1)
outcome   = 'Y'          # name of the outcome column (0/1, 0 = good outcome)

N_SPLITS        = 20     # cross-validation folds over the experimental data
THRESHOLD       = 0.1    # pruning threshold of the forward search
N_BOOT          = 10000  # bootstrap resamples for the summary tables
SEED            = 42
RUN_DIAGNOSTICS = True   # data checks printed before the CV loop

np.random.seed(SEED)

Do     = pd.read_csv(OBS_DATA_PATH)
De_all = pd.read_csv(EXP_DATA_PATH)


# ===========================================================================
# Counting
# ===========================================================================

def get_counts_multiZ(df, Z_cols, Y_col, Z_reference=None, Y_reference=None):
    """
    Contingency counts N_jk of Y over the configurations j of Z_cols.

    Pass Z_reference and Y_reference whenever counts from different datasets
    are combined: they fix the row and column order, so Do and De count
    arrays always have the same shape and alignment.
    """
    Z_cols = list(Z_cols)
    df = df.copy()

    for col in Z_cols:
        df[col] = df[col].astype('category')

    df[Y_col] = df[Y_col].astype('category')

    if Y_reference is not None:
        Y_values = list(Y_reference)
    else:
        Y_values = df[Y_col].cat.categories.tolist()

    if Z_reference is None:
        Z_categories = [sorted(df[col].cat.categories) for col in Z_cols]
        Z_values = list(itertools.product(*Z_categories))
    else:
        Z_values = list(Z_reference)

    index_map = {z: i for i, z in enumerate(Z_values)}
    y_map = {y: i for i, y in enumerate(Y_values)}

    N_j = np.zeros(len(Z_values), dtype=int)
    N_jk = np.zeros((len(Z_values), len(Y_values)), dtype=int)

    for _, row in df.iterrows():
        z_tuple = tuple(row[Z_cols])
        y_val = row[Y_col]
        if any(pd.isnull(v) for v in z_tuple) or pd.isnull(y_val):
            continue
        if z_tuple in index_map and y_val in y_map:
            i = index_map[z_tuple]
            k = y_map[y_val]
            N_j[i] += 1
            N_jk[i, k] += 1

    df_counts = pd.DataFrame(N_jk, index=Z_values, columns=Y_values)
    df_counts.index.name = "Z_configuration"
    df_counts["Total"] = N_j

    return Z_values, Y_values, N_j, N_jk, df_counts


# ===========================================================================
# Scoring
# ===========================================================================

def dirichlet_bayesian_score(counts, priors=None):
    """Log BD marginal likelihood with alpha_jk = priors + 1."""
    counts = np.asarray(counts)

    if priors is None:
        priors = np.zeros_like(counts)

    N = np.sum(counts, axis=-1)
    sum_priors = np.sum(priors + 1, axis=-1)

    score = (
        gammaln(sum_priors)
        - np.sum(gammaln(priors + 1), axis=-1)
        + np.sum(gammaln(counts + priors + 1), axis=-1)
        - gammaln(N + sum_priors)
    )

    return np.sum(score)


def make_priors(N_jk, priors_val):
    """Prior array giving alpha_jk = priors_val (float, so fractional values work)."""
    return np.full(np.shape(N_jk), priors_val - 1, dtype=float)


def P_De_given_HZc_log(N_o_jk, N_e_jk, priors):
    """log P(De | Do, H_Z^c)."""
    assert N_o_jk.shape == N_e_jk.shape, (
        f"count-array shape mismatch {N_o_jk.shape} vs {N_e_jk.shape} -- "
        "Z_reference or Y_reference is not shared between Do and De")
    return dirichlet_bayesian_score(N_o_jk + N_e_jk, priors) - \
           dirichlet_bayesian_score(N_o_jk, priors)


def P_De_given_HZc_bar_log(N_e_jk, priors):
    """log P(De | H_Z^c_bar); also used for log P(Do | H_Z^o)."""
    return dirichlet_bayesian_score(N_e_jk, priors)


# ===========================================================================
# Posterior predictive
# ===========================================================================

def compute_posterior_predictive_both_hypotheses(N_o_jk, N_o_j, N_e_jk, N_e_j,
                                                 alpha_jk, alpha_j):
    """
    Returns predictive P(Y | do(T), Z) under H_Z^c (Do + De pooled),
    under H_Z^c_bar (De only), and the observational P(Y | T, Z).
    """
    numerator_HZc = N_o_jk + N_e_jk + alpha_jk
    denominator_HZc = (N_o_j + N_e_j + alpha_j)[:, np.newaxis]
    probs_Y_HZc = numerator_HZc / denominator_HZc

    numerator_HZc_bar = N_e_jk + alpha_jk
    denominator_HZc_bar = (N_e_j + alpha_j)[:, np.newaxis]
    probs_Y_HZc_bar = numerator_HZc_bar / denominator_HZc_bar

    numerator_obs = N_o_jk + alpha_jk
    denominator_obs = (N_o_j + alpha_j)[:, np.newaxis]
    probs_Y_obs = numerator_obs / denominator_obs

    return probs_Y_HZc, probs_Y_HZc_bar, probs_Y_obs


def compute_posterior_predictive_single_model(N_jk, N_j):
    """Predictive P(Y | Z) from a single dataset, alpha_jk = 1."""
    alpha_jk = np.ones_like(N_jk)
    alpha_j = np.sum(alpha_jk, axis=1)

    numerator = N_jk + alpha_jk
    denominator = (N_j + alpha_j)[:, np.newaxis]

    return numerator / denominator


# ===========================================================================
# Metrics
# ===========================================================================

def expected_calibration_error(y_true, y_prob, n_bins=10):
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    N = len(y_true)

    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        if i == n_bins - 1:   # last bin closed on the right, so p = 1.0 is counted
            mask = (y_prob >= bins[i]) & (y_prob <= bins[i + 1])
        else:
            mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        bin_size = np.sum(mask)
        if bin_size > 0:
            acc = np.mean(y_true[mask])
            conf = np.mean(y_prob[mask])
            ece += (bin_size / N) * np.abs(acc - conf)

    return ece


def make_subset(X_col, Z_iterable):
    """Canonical node label: (treatment, *sorted covariates)."""
    return (X_col,) + tuple(sorted(set(Z_iterable)))


# ===========================================================================
# Search: FindIMB forward
# ===========================================================================

def greedy_search_FindIMB_forward(
        Do, De, X_col, Z_cols, Y_col,
        threshold=0.1, priors_val=1
):
    """
    Forward search over subsets Z (layer by layer, with pruning), then the
    posterior over (Z, H_Z^c / H_Z^c_bar).

    Each node Z gets three BD log-likelihoods:
        P(De | Do, H_Z^c),  P(De | H_Z^c_bar),  P(Do | H_Z^o).
    After each layer, a new node is expanded if any of its three normalized
    scores exceeds `threshold`. P(Do | H_Z^c_bar) is then approximated by a
    weighted average of P(Do | H_U^o) over the supersets U of Z reachable in
    the lattice T.

    Returns
    -------
    df_results : log-likelihoods and posteriors P_HZ_c, P_HZ_c_bar per subset.
    T          : the search lattice (parent -> children).
    """
    Z_cols = tuple(Z_cols)

    list_to_invest = [make_subset(X_col, [])]

    log_P_HZc     = {}
    log_P_HZc_bar = {}
    log_P_HZo     = {}

    # Search lattice. Every (parent, child) edge is recorded, including edges
    # into already-scored children, because P(Do | H_Z^c_bar) averages over all
    # nodes reachable from Z.
    T = {}

    def get_Z_reference(Z_subset):
        return get_reference(make_subset(X_col, Z_subset[1:]))

    def score_node(Z_subset):
        Z_reference = get_Z_reference(Z_subset)

        _, _, _, N_o_jk, _ = get_counts_multiZ(
            Do, list(Z_subset), Y_col,
            Z_reference=Z_reference, Y_reference=Y_REFERENCE
        )
        _, _, _, N_e_jk, _ = get_counts_multiZ(
            De, list(Z_subset), Y_col,
            Z_reference=Z_reference, Y_reference=Y_REFERENCE
        )

        priors = make_priors(N_o_jk, priors_val)

        log_P_HZc[Z_subset]     = P_De_given_HZc_log(N_o_jk, N_e_jk, priors)
        log_P_HZc_bar[Z_subset] = P_De_given_HZc_bar_log(N_e_jk, priors)
        log_P_HZo[Z_subset]     = P_De_given_HZc_bar_log(N_o_jk, priors)
        T[Z_subset]             = []

    # --- Initial node: treatment only ---
    score_node(make_subset(X_col, []))

    # --- Forward layered search ---
    for _ in range(len(Z_cols)):

        next_layer = set()

        for subset in list_to_invest:
            used_Zs = set(subset[1:])
            remaining = [z for z in Z_cols if z not in used_Zs]
            for z in remaining:
                new_subset = make_subset(X_col, list(used_Zs) + [z])
                next_layer.add((new_subset, subset))

        list_to_invest = []

        # sorted() makes the search order, and so the results, reproducible
        for Z_subset, parent in sorted(next_layer):

            T[parent].append(Z_subset)

            if Z_subset in log_P_HZc:
                continue

            score_node(Z_subset)
            list_to_invest.append(Z_subset)

        if not log_P_HZc:
            break

        # Normalization is global, over every subset scored so far
        log_total = logsumexp([
            np.logaddexp(log_P_HZc[z], log_P_HZc_bar[z])
            for z in log_P_HZc
        ])

        Scores_HZc = {z: np.exp(log_P_HZc[z] - log_total) for z in log_P_HZc}
        Scores_HZc_bar = {z: np.exp(log_P_HZc_bar[z] - log_total) for z in log_P_HZc_bar}

        log_total_HZo = logsumexp(list(log_P_HZo.values()))
        Scores_HZo = {z: np.exp(log_P_HZo[z] - log_total_HZo) for z in log_P_HZo}

        # Expand only subsets whose score passes the threshold
        list_to_invest = [
            z for z in list_to_invest
            if Scores_HZc[z]     > threshold
            or Scores_HZc_bar[z] > threshold
            or Scores_HZo[z]     > threshold
        ]

        if not list_to_invest:
            break

    # --- Final phase: posterior over (Z, hypothesis) ---

    def get_reachable_node_sets(node):
        """S_Z: all nodes reachable from `node` in the lattice, including itself."""
        seen, stack = set(), [node]
        while stack:
            n = stack.pop()
            if n not in seen:
                seen.add(n)
                stack.extend(T[n])
        return list(seen)

    V_size = len(Z_cols)

    def compute_log_P_HZo_bar(Z_subset):
        """log P(Do | H_Z^c_bar): weighted average of P(Do | H_U^o) over U in S_Z."""
        S_Z   = get_reachable_node_sets(Z_subset)
        Z_len = len(Z_subset) - 1   # exclude the treatment
        denom = V_size - Z_len      # |V| - |Z|

        log_terms   = []
        log_weights = []
        for U in S_Z:
            diff  = (len(U) - 1) - Z_len
            binom = comb(denom, diff, exact=True)
            log_w = -np.log(binom) - np.log(denom + 1)
            log_weights.append(log_w)
            log_terms.append(log_P_HZo[U] + log_w)

        # Renormalize the weights over S_Z (normalization constant K). K = 1 when
        # nothing was pruned; otherwise it restores the weight of pruned nodes.
        log_K = logsumexp(log_weights)

        return logsumexp(log_terms) - log_K

    log_numerators_Hz     = {}
    log_numerators_Hz_bar = {}
    log_P_HZo_bar         = {}

    for Z_subset in log_P_HZc:
        log_numerators_Hz[Z_subset] = log_P_HZc[Z_subset] + log_P_HZo[Z_subset]
        log_P_HZo_bar[Z_subset] = compute_log_P_HZo_bar(Z_subset)
        log_numerators_Hz_bar[Z_subset] = log_P_HZc_bar[Z_subset] + log_P_HZo_bar[Z_subset]

    log_denominator = logsumexp([
        np.logaddexp(log_numerators_Hz[z], log_numerators_Hz_bar[z])
        for z in log_P_HZc
    ])

    P_HZ_c     = {z: np.exp(log_numerators_Hz[z] - log_denominator) for z in log_P_HZc}
    P_HZ_c_bar = {z: np.exp(log_numerators_Hz_bar[z] - log_denominator) for z in log_P_HZc_bar}

    df_results = pd.DataFrame({
        'Variables':            list(log_P_HZc.keys()),
        'P(De|Do,HcZ) log':     list(log_P_HZc.values()),
        'P(De|Do,HcZ_bar) log': list(log_P_HZc_bar.values()),
        'P(Do|HoZ) log':        list(log_P_HZo.values()),
        'P(Do|HoZ_bar) log':    [log_P_HZo_bar[z] for z in log_P_HZc],
        'P_HZ_c':               [P_HZ_c[z]     for z in log_P_HZc],
        'P_HZ_c_bar':           [P_HZ_c_bar[z] for z in log_P_HZc_bar]
    })

    return df_results, T


# ===========================================================================
# Search: single dataset forward (baselines)
# ===========================================================================

def greedy_search_single_dataset_forward(
        data, X_col, Z_cols, Y_col,
        threshold=0.1, priors_val=1
):
    """Forward search over subsets Z scored by P(data | Z) on one dataset."""
    Z_cols = tuple(Z_cols)

    list_to_invest = [(X_col,)]
    log_P_data = {}

    def score_node(Z_subset):
        _, _, _, N_jk, _ = get_counts_multiZ(
            data, list(Z_subset), Y_col,
            Z_reference=get_reference(tuple(Z_subset)), Y_reference=Y_REFERENCE
        )
        priors = make_priors(N_jk, priors_val)
        log_P_data[Z_subset] = P_De_given_HZc_bar_log(N_jk, priors)

    score_node((X_col,))

    for _ in range(len(Z_cols)):

        next_layer = []

        for subset in list_to_invest:
            used_Zs = set(subset[1:])
            remaining = [z for z in Z_cols if z not in used_Zs]
            for z in remaining:
                next_layer.append(make_subset(X_col, list(used_Zs) + [z]))

        next_layer = sorted(set(next_layer))
        list_to_invest = []

        for Z_subset in next_layer:
            if Z_subset in log_P_data:
                continue
            score_node(Z_subset)
            list_to_invest.append(Z_subset)

        if not log_P_data:
            break

        log_total = logsumexp(np.array(list(log_P_data.values())))
        Scores = {z: np.exp(v - log_total) for z, v in log_P_data.items()}

        list_to_invest = [z for z in list_to_invest if Scores[z] > threshold]

        if not list_to_invest:
            break

    df_results = pd.DataFrame({
        'Variables': list(log_P_data.keys()),
        'logP(data|Z)': list(log_P_data.values())
    })

    log_total = logsumexp(np.array(list(log_P_data.values())))
    df_results['P(data|Z)'] = np.exp(df_results['logP(data|Z)'] - log_total)

    return df_results


# ===========================================================================
# Expected outcome / DEU
# ===========================================================================

def evaluate_expected_outcome(
        De_test, Do, De,
        df_scores, df_Score_De, df_Score_Do, df_Score_Do_De,
        subsets, treatment, outcome
):
    """
    Predicts the outcome of every test patient under T = 0 and T = 1, picks
    the treatment with the higher P(Y = 0), and returns, for each of the four
    methods (FindIMB, De only, Do only, Do + De pooled):
        T0, T1 : number of patients for whom T = 0 / T = 1 is recommended;
        avgY   : mean predicted P(Y = 0) under the recommended treatment;
        DEU    : DEU of the recommendations.
    """
    Do_De = pd.concat([Do, De], ignore_index=True)
    N = len(De_test)
    K = 2

    Dtest_T0 = De_test.copy()
    Dtest_T0[treatment] = 0
    Dtest_T1 = De_test.copy()
    Dtest_T1[treatment] = 1

    # --- FindIMB: BMA over (Z, H_Z^c / H_Z^c_bar) ---
    def compute_BMA_alg(De_input):
        P_Y_accum = np.zeros((N, K))

        score_map_c = {tuple(r['Variables']): r['P_HZ_c'] for _, r in df_scores.iterrows()}
        score_map_cb = {tuple(r['Variables']): r['P_HZ_c_bar'] for _, r in df_scores.iterrows()}

        for Z_cols in subsets:
            w_c  = score_map_c.get(tuple(Z_cols), 0.0)
            w_cb = score_map_cb.get(tuple(Z_cols), 0.0)

            Z_reference = get_reference(tuple(Z_cols))

            Z_vals_Do, _, N_o_j, N_o_jk, _ = get_counts_multiZ(
                Do, Z_cols, outcome, Z_reference, Y_reference=Y_REFERENCE)
            Z_vals_De, _, N_e_j, N_e_jk, _ = get_counts_multiZ(
                De, Z_cols, outcome, Z_reference, Y_reference=Y_REFERENCE)

            alpha_jk = np.ones_like(N_o_jk)
            alpha_j = np.sum(alpha_jk, axis=1)

            probs_HZc, probs_HZc_bar, _ = compute_posterior_predictive_both_hypotheses(
                N_o_jk, N_o_j, N_e_jk, N_e_j, alpha_jk, alpha_j
            )

            Z_index = {tuple(z): i for i, z in enumerate(Z_vals_De)}

            rows_c, rows_cb = [], []
            for _, r in De_input.iterrows():
                idx = Z_index.get(tuple(r[col] for col in Z_cols))
                if idx is not None:
                    rows_c.append(probs_HZc[idx])
                    rows_cb.append(probs_HZc_bar[idx])
                else:
                    rows_c.append(np.ones(K) / K)
                    rows_cb.append(np.ones(K) / K)

            P_Y_accum += w_c * np.asarray(rows_c) + w_cb * np.asarray(rows_cb)

        return P_Y_accum / np.clip(P_Y_accum.sum(axis=1, keepdims=True), 1e-12, None)

    # --- Baselines: BMA over Z on a single dataset ---
    def compute_BMA_single(D_train, De_input, df_score):
        P_Y_accum = np.zeros((N, K))

        for _, row in df_score.iterrows():
            Z_cols = list(row['Variables'])
            w = float(row['P(data|Z)'])

            Z_reference = get_reference(tuple(Z_cols))

            Z_vals, _, N_j, N_jk, _ = get_counts_multiZ(
                D_train, Z_cols, outcome, Z_reference, Y_reference=Y_REFERENCE)

            probs_Y = compute_posterior_predictive_single_model(N_jk, N_j)

            Z_index = {tuple(z): i for i, z in enumerate(Z_vals)}

            rows = []
            for _, r in De_input.iterrows():
                idx = Z_index.get(tuple(r[col] for col in Z_cols))
                if idx is not None:
                    rows.append(probs_Y[idx])
                else:
                    rows.append(np.ones(K) / K)

            P_Y_accum += w * np.asarray(rows)

        return P_Y_accum / np.clip(P_Y_accum.sum(axis=1, keepdims=True), 1e-12, None)

    P_Y_alg_T0 = compute_BMA_alg(Dtest_T0)
    P_Y_alg_T1 = compute_BMA_alg(Dtest_T1)

    P_Y_exp_T0 = compute_BMA_single(De,    Dtest_T0, df_Score_De)
    P_Y_exp_T1 = compute_BMA_single(De,    Dtest_T1, df_Score_De)

    P_Y_obs_T0 = compute_BMA_single(Do,    Dtest_T0, df_Score_Do)
    P_Y_obs_T1 = compute_BMA_single(Do,    Dtest_T1, df_Score_Do)

    P_Y_all_T0 = compute_BMA_single(Do_De, Dtest_T0, df_Score_Do_De)
    P_Y_all_T1 = compute_BMA_single(Do_De, Dtest_T1, df_Score_Do_De)

    real_T = De_test[treatment].tolist()
    real_Y = De_test[outcome].tolist()

    def best_treatment_probs(P_T0, P_T1):
        """
        P_T0, P_T1: (N, K) outcome matrices; column 0 is P(Y = 0).

        If P(Y = 0) is equal under T = 0 and T = 1, the patient is assigned
        T = 0. This happens e.g. when the patient's covariate values never
        occur in the training data under either treatment: with zero counts
        the predictive is the prior, 1/K, for both.
        """
        a = 0.1
        b = 0.2

        best_T = (P_T1[:, 0] > P_T0[:, 0]).astype(int)

        best_probs = np.where(best_T[:, None] == 1, P_T1, P_T0)
        avgY0 = best_probs[:, 0].mean()

        T0Count = int(np.sum(best_T == 0))
        T1Count = int(np.sum(best_T == 1))

        P_BX_0 = (T0Count + a) / (T0Count + T1Count + b)
        P_BX_1 = (T1Count + a) / (T0Count + T1Count + b)

        common_T = [r if r == t else None for r, t in zip(real_T, best_T)]

        Y_T0 = [real_Y[i] for i, t in enumerate(common_T) if t == 0]
        Y_T1 = [real_Y[i] for i, t in enumerate(common_T) if t == 1]

        P_Y0_T0 = (Y_T0.count(0) + a) / (len(Y_T0) + b) if len(Y_T0) > 0 else a / b
        P_Y0_T1 = (Y_T1.count(0) + a) / (len(Y_T1) + b) if len(Y_T1) > 0 else a / b

        DEU = P_Y0_T0 * P_BX_0 + P_Y0_T1 * P_BX_1

        return T0Count, T1Count, avgY0, DEU

    # Per-patient P(Y = 0) under each treatment, pooled across folds for DEU
    fold_df_DEU = pd.DataFrame({
        'patient_id': De_test.index,
        'y_true': De_test[outcome],
        'real_T': real_T,
        'P_Y_alg_T0': P_Y_alg_T0[:, 0],
        'P_Y_alg_T1': P_Y_alg_T1[:, 0],
        'P_Y_exp_T0': P_Y_exp_T0[:, 0],
        'P_Y_exp_T1': P_Y_exp_T1[:, 0],
        'P_Y_obs_T0': P_Y_obs_T0[:, 0],
        'P_Y_obs_T1': P_Y_obs_T1[:, 0],
        'P_Y_all_T0': P_Y_all_T0[:, 0],
        'P_Y_all_T1': P_Y_all_T1[:, 0]
    })

    all_preds_DEU.append(fold_df_DEU)

    T0_alg, T1_alg, avgY_alg, DEU_alg = best_treatment_probs(P_Y_alg_T0, P_Y_alg_T1)
    T0_exp, T1_exp, avgY_exp, DEU_exp = best_treatment_probs(P_Y_exp_T0, P_Y_exp_T1)
    T0_obs, T1_obs, avgY_obs, DEU_obs = best_treatment_probs(P_Y_obs_T0, P_Y_obs_T1)
    T0_all, T1_all, avgY_all, DEU_all = best_treatment_probs(P_Y_all_T0, P_Y_all_T1)

    return (
        T0_obs, T1_obs, avgY_obs,
        T0_exp, T1_exp, avgY_exp,
        T0_alg, T1_alg, avgY_alg,
        T0_all, T1_all, avgY_all,
        DEU_alg, DEU_exp, DEU_obs, DEU_all
    )


def bma_predict_and_evaluate(Do, De, De_test, df_scores, treatment, outcome,
                             df_Score_De, df_Score_Do, df_Score_Do_De):
    """BMA predictions for the four methods, evaluated with log-loss, ECE, AUC and DEU."""
    Do_De = pd.concat([Do, De], ignore_index=True)

    Ntest = len(De_test)
    K = 2
    P_Y_alg_accum = np.zeros((Ntest, K))

    subsets = [list(t) for t in df_scores['Variables']]

    # --- FindIMB ---
    for _, row in df_scores.iterrows():
        Z_cols = list(row['Variables'])
        w_c  = float(row['P_HZ_c'])
        w_cb = float(row['P_HZ_c_bar'])

        Z_reference = get_reference(tuple(Z_cols))

        Z_vals_Do, _, N_o_j, N_o_jk, _ = get_counts_multiZ(
            Do, Z_cols, outcome, Z_reference=Z_reference, Y_reference=Y_REFERENCE)
        Z_vals_De, _, N_e_j, N_e_jk, _ = get_counts_multiZ(
            De, Z_cols, outcome, Z_reference=Z_reference, Y_reference=Y_REFERENCE)

        alpha_jk = np.ones_like(N_o_jk)
        alpha_j = np.sum(alpha_jk, axis=1)

        probs_Y_HZc, probs_Y_HZc_bar, _ = compute_posterior_predictive_both_hypotheses(
            N_o_jk, N_o_j, N_e_jk, N_e_j, alpha_jk, alpha_j
        )

        Z_config_to_index = {tuple(z): i for i, z in enumerate(Z_vals_De)}
        probs_rows_HZc, probs_rows_HZc_bar = [], []

        for _, rtest in De_test.iterrows():
            z_tuple = tuple(rtest[col] for col in Z_cols)
            idx = Z_config_to_index.get(z_tuple)
            if idx is not None:
                probs_rows_HZc.append(probs_Y_HZc[idx])
                probs_rows_HZc_bar.append(probs_Y_HZc_bar[idx])
            else:
                probs_rows_HZc.append(np.ones(K) / K)
                probs_rows_HZc_bar.append(np.ones(K) / K)

        P_Y_alg_accum += (w_c * np.asarray(probs_rows_HZc)
                          + w_cb * np.asarray(probs_rows_HZc_bar))

    P_Y_alg = P_Y_alg_accum / np.clip(P_Y_alg_accum.sum(axis=1, keepdims=True), 1e-12, None)

    # --- Baselines ---
    def bma_predict_single_source(D_train, D_test, df_scores_single, outcome, K=2):
        Ntest = len(D_test)
        P_Y_accum = np.zeros((Ntest, K))

        for _, row in df_scores_single.iterrows():
            Z_cols = list(row['Variables'])
            w = float(row['P(data|Z)'])

            Z_reference = get_reference(tuple(Z_cols))

            Z_vals, _, N_j, N_jk, _ = get_counts_multiZ(
                D_train, Z_cols, outcome,
                Z_reference=Z_reference, Y_reference=Y_REFERENCE)

            probs_Y = compute_posterior_predictive_single_model(N_jk, N_j)

            Z_config_to_index = {tuple(z): i for i, z in enumerate(Z_vals)}

            probs_rows = []
            for _, rtest in D_test.iterrows():
                z_tuple = tuple(rtest[col] for col in Z_cols)
                idx = Z_config_to_index.get(z_tuple)
                if idx is not None:
                    probs_rows.append(probs_Y[idx])
                else:
                    probs_rows.append(np.ones(K) / K)

            P_Y_accum += w * np.asarray(probs_rows)

        return P_Y_accum / np.clip(P_Y_accum.sum(axis=1, keepdims=True), 1e-12, None)

    P_Y_obs = np.asarray(bma_predict_single_source(Do,    De_test, df_Score_Do,    outcome))
    P_Y_exp = np.asarray(bma_predict_single_source(De,    De_test, df_Score_De,    outcome))
    P_Y_all = np.asarray(bma_predict_single_source(Do_De, De_test, df_Score_Do_De, outcome))

    (T0_obs, T1_obs, avgY1_obs,
     T0_exp, T1_exp, avgY1_exp,
     T0_alg, T1_alg, avgY1_alg,
     T0_all, T1_all, avgY1_all,
     DEU_alg, DEU_exp, DEU_obs, DEU_all) = evaluate_expected_outcome(
        De_test=De_test, Do=Do, De=De,
        df_scores=df_scores,
        df_Score_De=df_Score_De,
        df_Score_Do=df_Score_Do,
        df_Score_Do_De=df_Score_Do_De,
        subsets=subsets,
        treatment=treatment,
        outcome=outcome
    )

    y_true = De_test[outcome].to_numpy()
    alg_loss = log_loss(y_true, P_Y_alg, labels=[0, 1])
    exp_loss = log_loss(y_true, P_Y_exp, labels=[0, 1])
    obs_loss = log_loss(y_true, P_Y_obs, labels=[0, 1])
    all_loss = log_loss(y_true, P_Y_all, labels=[0, 1])

    # ECE and AUC treat the good outcome (Y = 0) as the positive class
    y_true_good = (y_true == 0).astype(int)

    ece_alg = expected_calibration_error(y_true_good, P_Y_alg[:, 0], n_bins=10)
    ece_exp = expected_calibration_error(y_true_good, P_Y_exp[:, 0], n_bins=10)
    ece_obs = expected_calibration_error(y_true_good, P_Y_obs[:, 0], n_bins=10)
    ece_all = expected_calibration_error(y_true_good, P_Y_all[:, 0], n_bins=10)

    if len(np.unique(y_true_good)) < 2:
        auc_alg = auc_exp = auc_obs = auc_all = np.nan
    else:
        auc_alg = roc_auc_score(y_true_good, P_Y_alg[:, 0])
        auc_exp = roc_auc_score(y_true_good, P_Y_exp[:, 0])
        auc_obs = roc_auc_score(y_true_good, P_Y_obs[:, 0])
        auc_all = roc_auc_score(y_true_good, P_Y_all[:, 0])

    fold_df = pd.DataFrame({
        'patient_id': De_test.index,
        'y_true': y_true,
        'P_Y_alg': P_Y_alg[:, 0],
        'P_Y_exp': P_Y_exp[:, 0],
        'P_Y_obs': P_Y_obs[:, 0],
        'P_Y_all': P_Y_all[:, 0]
    })

    all_preds.append(fold_df)

    metrics = {
        "algorithm":     dict(T0=T0_alg, T1=T1_alg, avgY1=avgY1_alg, logloss=alg_loss, ece=ece_alg, auc=auc_alg, DEU=DEU_alg),
        "experimental":  dict(T0=T0_exp, T1=T1_exp, avgY1=avgY1_exp, logloss=exp_loss, ece=ece_exp, auc=auc_exp, DEU=DEU_exp),
        "observational": dict(T0=T0_obs, T1=T1_obs, avgY1=avgY1_obs, logloss=obs_loss, ece=ece_obs, auc=auc_obs, DEU=DEU_obs),
        "all":           dict(T0=T0_all, T1=T1_all, avgY1=avgY1_all, logloss=all_loss, ece=ece_all, auc=auc_all, DEU=DEU_all),
        "P_Y":           dict(alg=P_Y_alg, exp=P_Y_exp, obs=P_Y_obs)
    }

    return metrics


def DEU(P_T0, P_T1, real_T, real_Y):
    """
    Direct Expected Utility.

    P_T0, P_T1 must be single-column (N, 1) arrays holding ONE method's
    P(Y = 0) under T = 0 and T = 1, e.g. df_DEU[[f"{model}_T0"]].values.
    """
    a = 0.1
    b = 0.2

    best_T = (P_T1[:, 0] > P_T0[:, 0]).astype(int)

    T0Count = int(np.sum(best_T == 0))
    T1Count = int(np.sum(best_T == 1))

    P_BX_0 = (T0Count + a) / (T0Count + T1Count + b)
    P_BX_1 = (T1Count + a) / (T0Count + T1Count + b)

    common_T = [r if r == t else None for r, t in zip(real_T, best_T)]

    Y_T0 = [real_Y[i] for i, t in enumerate(common_T) if t == 0]
    Y_T1 = [real_Y[i] for i, t in enumerate(common_T) if t == 1]

    P_Y0_T0 = (Y_T0.count(0) + a) / (len(Y_T0) + b) if len(Y_T0) > 0 else a / b
    P_Y0_T1 = (Y_T1.count(0) + a) / (len(Y_T1) + b) if len(Y_T1) > 0 else a / b

    return P_Y0_T0 * P_BX_0 + P_Y0_T1 * P_BX_1


# ===========================================================================
# Bootstrap summaries
# ===========================================================================

def bootstrap_metrics_summary(df, df_DEU, y_col='y_true', model_cols=None,
                              n_bins=10, n_boot=10000, random_state=None):
    """Bootstrap mean and 95% CI of ECE, AUC, LogLoss and DEU for each model."""
    rng = np.random.default_rng(random_state)

    if model_cols is None:
        model_cols = [c for c in df.columns if c != y_col]

    summary_rows = []
    y_true = df[y_col].values
    real_T = df_DEU['real_T'].values
    y_good_outcome = (y_true == 0).astype(int)

    n = len(df)

    for model in model_cols:
        deu_list, ece_list, auc_list, ll_list = [], [], [], []

        p_pred = df[model].values
        p_pred_T0 = df_DEU[[f"{model}_T0"]].values
        p_pred_T1 = df_DEU[[f"{model}_T1"]].values

        for _ in range(n_boot):
            idx = rng.choice(n, size=n, replace=True)
            y_sample = y_good_outcome[idx]
            p_sample = p_pred[idx]

            ece_list.append(expected_calibration_error(y_sample, p_sample, n_bins=n_bins))
            if len(np.unique(y_sample)) == 2:
                auc_list.append(roc_auc_score(y_sample, p_sample))
            else:
                auc_list.append(np.nan)
            ll_list.append(log_loss(y_sample, p_sample, labels=[0, 1]))

            deu_list.append(DEU(p_pred_T0[idx], p_pred_T1[idx], real_T[idx], y_true[idx]))

        for name, arr in [('ECE', ece_list), ('AUC', auc_list),
                          ('LogLoss', ll_list), ('DEU', deu_list)]:
            arr = np.array(arr)
            summary_rows.append([model, name, np.nanmean(arr),
                                 np.nanpercentile(arr, 2.5), np.nanpercentile(arr, 97.5)])

    # Reference rows: observed P(Y = 0) in each randomized arm
    ATE_A0_list, ATE_A1_list = [], []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)

        real_T_sample = real_T[idx]
        real_Y_sample = y_true[idx]

        mask0 = real_T_sample == 0
        mask1 = real_T_sample == 1

        ATE_A0_list.append((real_Y_sample[mask0] == 0).mean() if mask0.any() else np.nan)
        ATE_A1_list.append((real_Y_sample[mask1] == 0).mean() if mask1.any() else np.nan)

    for name, arr in [('ATE_A0', ATE_A0_list), ('ATE_A1', ATE_A1_list)]:
        arr = np.array(arr)
        summary_rows.append([name, 'DEU', np.nanmean(arr),
                             np.nanpercentile(arr, 2.5), np.nanpercentile(arr, 97.5)])

    return pd.DataFrame(summary_rows,
                        columns=['model', 'metric', 'mean', 'ci_lower', 'ci_upper'])


def _delta_row(base_model, other_model, metric_name, delta_arr):
    """Mean delta, 95% CI and two-sided bootstrap p-value."""
    finite = delta_arr[~np.isnan(delta_arr)]
    p_value = min(1.0, 2 * min(
        (np.sum(finite <= 0) + 1) / (len(finite) + 1),
        (np.sum(finite >= 0) + 1) / (len(finite) + 1)
    ))
    return [base_model, other_model, metric_name, np.nanmean(delta_arr),
            np.nanpercentile(delta_arr, 2.5), np.nanpercentile(delta_arr, 97.5), p_value]


def paired_bootstrap_summary_from_df(df, df_DEU, base_model='P_Y_alg', model_cols=None,
                                     y_col='y_true', n_bins=10, n_boot=10000,
                                     random_state=None):
    """
    Paired bootstrap comparison of base_model against every other model.
    All deltas are oriented so that a POSITIVE delta favours base_model.
    """
    if model_cols is None:
        model_cols = [c for c in df.columns if c != y_col]

    y_true = df[y_col].values
    y_good_outcome = (y_true == 0).astype(int)
    real_T = df_DEU['real_T'].values

    rng = np.random.default_rng(random_state)
    summary_rows = []

    n = len(df)
    pred_base = df[base_model].values
    p_base_T0 = df_DEU[[f"{base_model}_T0"]].values
    p_base_T1 = df_DEU[[f"{base_model}_T1"]].values

    for other_model in model_cols:
        if other_model == base_model:
            continue

        pred_other = df[other_model].values
        p_other_T0 = df_DEU[[f"{other_model}_T0"]].values
        p_other_T1 = df_DEU[[f"{other_model}_T1"]].values

        delta_ece, delta_auc, delta_ll, delta_deu = [], [], [], []

        for _ in range(n_boot):
            idx = rng.choice(n, size=n, replace=True)

            y_sample = y_good_outcome[idx]
            p_base_sample  = pred_base[idx]
            p_other_sample = pred_other[idx]

            # lower is better -> other - base
            delta_ece.append(expected_calibration_error(y_sample, p_other_sample, n_bins=n_bins) -
                             expected_calibration_error(y_sample, p_base_sample, n_bins=n_bins))

            # higher is better -> base - other
            if len(np.unique(y_sample)) == 2:
                delta_auc.append(roc_auc_score(y_sample, p_base_sample) -
                                 roc_auc_score(y_sample, p_other_sample))
            else:
                delta_auc.append(np.nan)

            # lower is better -> other - base
            delta_ll.append(log_loss(y_sample, p_other_sample, labels=[0, 1]) -
                            log_loss(y_sample, p_base_sample,  labels=[0, 1]))

            # higher is better -> base - other
            real_T_sample = real_T[idx]
            real_Y_sample = y_true[idx]
            delta_deu.append(
                DEU(p_base_T0[idx],  p_base_T1[idx],  real_T_sample, real_Y_sample) -
                DEU(p_other_T0[idx], p_other_T1[idx], real_T_sample, real_Y_sample)
            )

        for metric_name, delta in [('ECE', delta_ece), ('AUC', delta_auc),
                                   ('LogLoss', delta_ll), ('DEU', delta_deu)]:
            summary_rows.append(_delta_row(base_model, other_model, metric_name, np.array(delta)))

    # DEU of base_model vs. observed P(Y = 0) in each randomized arm
    delta_A0, delta_A1 = [], []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)

        real_T_sample = real_T[idx]
        real_Y_sample = y_true[idx]

        deu_base = DEU(p_base_T0[idx], p_base_T1[idx], real_T_sample, real_Y_sample)

        mask0 = real_T_sample == 0
        mask1 = real_T_sample == 1

        ATEforRx0 = (real_Y_sample[mask0] == 0).mean() if mask0.any() else np.nan
        ATEforRx1 = (real_Y_sample[mask1] == 0).mean() if mask1.any() else np.nan

        delta_A0.append(deu_base - ATEforRx0)
        delta_A1.append(deu_base - ATEforRx1)

    summary_rows.append(_delta_row(base_model, "ATE_A0", "DEU", np.array(delta_A0)))
    summary_rows.append(_delta_row(base_model, "ATE_A1", "DEU", np.array(delta_A1)))

    return pd.DataFrame(summary_rows, columns=[
        'base_model', 'other_model', 'metric', 'delta_obs', 'ci_lower', 'ci_upper', 'p_value'
    ])


# ===========================================================================
# Setup
# ===========================================================================

start_time = time.perf_counter()
start_cpu  = time.process_time()

os.makedirs(OUTPUT_DIR, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")


def out_path(name):
    return os.path.join(OUTPUT_DIR, f"{name}_{timestamp}.csv")


covariates_without_T = [col for col in Do.columns if col not in [outcome, treatment]]
ALL_COLS = [treatment] + covariates_without_T + [outcome]

_missing_in_De = [c for c in ALL_COLS if c not in De_all.columns]
if _missing_in_De:
    print("WARNING: columns of Do missing from the experimental data:", _missing_in_De)


# ---------------------------------------------------------------------------
# Shared category vocabulary
# ---------------------------------------------------------------------------
# All methods and all folds count over the same configuration space, so a
# configuration unseen in training falls back to uniform 1/K identically for
# every method. Only the SET OF LEVELS of each variable is read from the full
# experimental data; all counts come from training data only.

def _levels(df, c):
    return set(df[c].dropna().unique()) if c in df.columns else set()


CATEGORIES = {c: sorted(_levels(Do, c) | _levels(De_all, c)) for c in ALL_COLS}

Y_REFERENCE = CATEGORIES[outcome]


@lru_cache(maxsize=None)
def get_reference(cols):
    """Full Cartesian product of the category sets of `cols` (a tuple)."""
    return list(itertools.product(*[CATEGORIES[c] for c in cols]))


print(f"\nOutcome category order: {Y_REFERENCE}")
if list(Y_REFERENCE)[0] != 0:
    print("WARNING: the first outcome category is not 0. The script assumes "
          "column 0 of every probability matrix is the good outcome Y = 0.")


# ===========================================================================
# Diagnostics
# ===========================================================================

if RUN_DIAGNOSTICS:
    print("\n" + "=" * 72)
    print("DIAGNOSTIC A - does any training fold of De contain a single outcome class?")
    print("=" * 72)
    problem_A = False
    for _f, (_tr, _te) in enumerate(KFold(n_splits=N_SPLITS, shuffle=False).split(De_all)):
        _k = De_all.iloc[_tr][outcome].nunique()
        flag = "   <-- PROBLEM" if _k != 2 else ""
        print(f"fold {_f+1:2d}: training rows = {len(_tr):5d}   distinct outcome values = {_k}{flag}")
        if _k != 2:
            problem_A = True
    print("VERDICT:", "single-class fold present" if problem_A
          else "every fold has both outcome classes")

    print("\n" + "=" * 72)
    print("DIAGNOSTIC B - does the shared vocabulary cover every test row?")
    print("=" * 72)
    problem_B = False
    for _f, (_tr, _te) in enumerate(KFold(n_splits=N_SPLITS, shuffle=False).split(De_all)):
        _bad = [c for c in ALL_COLS
                if c in De_all.columns
                and not set(De_all.iloc[_te][c].dropna().unique()) <= set(CATEGORIES[c])]
        if _bad:
            print(f"fold {_f+1:2d}: uncovered levels in {_bad}")
            problem_B = True
    print("VERDICT:", "uncovered levels found" if problem_B
          else "every test row is covered")
    print("=" * 72 + "\n")


# ===========================================================================
# Observational search (fixed across folds)
# ===========================================================================

df_Score_Do = greedy_search_single_dataset_forward(
    Do, treatment, covariates_without_T, outcome, threshold=THRESHOLD)

max_row_Do = df_Score_Do.loc[df_Score_Do['P(data|Z)'].idxmax()]
best_set_Do = max_row_Do['Variables']

print("Best subset in Do:", best_set_Do)
print("Probability for the best subset in Do:", max_row_Do['P(data|Z)'])


# ===========================================================================
# Cross-validation over the experimental data
# ===========================================================================

results = []
PHz_results = []
best_sets_results = []
all_preds = []        # filled by bma_predict_and_evaluate
all_preds_DEU = []    # filled by evaluate_expected_outcome

kf = KFold(n_splits=N_SPLITS, shuffle=False)

for fold, (train_idx, test_idx) in enumerate(kf.split(De_all)):

    print(f"\nFold {fold+1}")

    De = De_all.iloc[train_idx]
    De_test = De_all.iloc[test_idx]

    ATEforRx0 = (De_test.loc[De_test[treatment] == 0, outcome] == 0).mean()
    ATEforRx1 = (De_test.loc[De_test[treatment] == 1, outcome] == 0).mean()

    Do_De = pd.concat([Do, De], ignore_index=True)

    df_Score_De = greedy_search_single_dataset_forward(
        De, treatment, covariates_without_T, outcome, threshold=THRESHOLD)
    best_set_De = df_Score_De.loc[df_Score_De['P(data|Z)'].idxmax(), 'Variables']

    df_Score_Do_De = greedy_search_single_dataset_forward(
        Do_De, treatment, covariates_without_T, outcome, threshold=THRESHOLD)
    best_set_Do_De = df_Score_Do_De.loc[df_Score_Do_De['P(data|Z)'].idxmax(), 'Variables']

    df_scores, T = greedy_search_FindIMB_forward(
        Do, De, treatment, covariates_without_T, outcome, threshold=THRESHOLD)

    best_c = df_scores.loc[df_scores['P_HZ_c'].idxmax()]
    best_c_bar = df_scores.loc[df_scores['P_HZ_c_bar'].idxmax()]

    best_idx = (df_scores['P_HZ_c'] + df_scores['P_HZ_c_bar']).idxmax()
    best_set_FindIMB = df_scores.loc[best_idx, 'Variables']

    metrics = bma_predict_and_evaluate(
        Do, De, De_test, df_scores,
        treatment, outcome,
        df_Score_De, df_Score_Do, df_Score_Do_De
    )

    fold_result = {"fold": fold + 1, "ATE_Rx0": ATEforRx0, "ATE_Rx1": ATEforRx1}
    arms = {"alg": "algorithm", "exp": "experimental", "obs": "observational", "all": "all"}
    for key, metric in [("DEU", "DEU"), ("AUC", "auc"), ("logloss", "logloss"), ("ece", "ece")]:
        for short, arm in arms.items():
            fold_result[f"{key}_{short}"] = metrics[arm][metric]
    for short, arm in arms.items():
        fold_result[f"T0_{short}"] = metrics[arm]['T0']
        fold_result[f"T1_{short}"] = metrics[arm]['T1']

    results.append(fold_result)

    PHz_results.append({
        "fold": fold + 1,
        "best_var_P_HZ_c": best_c["Variables"],
        "P_HZ_c": best_c["P_HZ_c"],
        "best_var_P_HZ_c_bar": best_c_bar["Variables"],
        "P_HZ_c_bar": best_c_bar["P_HZ_c_bar"],
        "total_P_HZ_c": df_scores['P_HZ_c'].sum(),
        "total_P_HZ_c_bar": df_scores['P_HZ_c_bar'].sum()
    })

    best_sets_results.append({
        "fold": fold + 1,
        "FindIMB": best_set_FindIMB,
        "Do": best_set_Do,
        "De": best_set_De,
        "Do_De": best_set_Do_De
    })

    # Saved after every fold so a long run can be inspected or resumed
    pd.DataFrame(results).to_csv(out_path("cv_metrics"), index=False)
    pd.DataFrame(PHz_results).to_csv(out_path("PHz_max_table"), index=False)
    pd.DataFrame(best_sets_results).to_csv(out_path("best_sets"), index=False)

    print("Progress saved.")


# ===========================================================================
# Variable importance: fraction of folds in which each covariate is in the
# best subset of each method
# ===========================================================================

df_best_sets = pd.DataFrame(best_sets_results)

all_covariates = sorted({cov for col in df_best_sets.columns[1:]
                         for row in df_best_sets[col] for cov in row})

importance_df = pd.DataFrame(0, index=all_covariates,
                             columns=df_best_sets.columns[1:], dtype=float)

for method in df_best_sets.columns[1:]:
    for cov in all_covariates:
        count = sum(cov in fold_set for fold_set in df_best_sets[method])
        importance_df.loc[cov, method] = count / N_SPLITS

importance_df.to_csv(out_path("Variable_importance"), index=True)


# ===========================================================================
# Pooled predictions and bootstrap summaries
# ===========================================================================
# All probabilities below are P(Y = 0), pooled over every test fold.

predictions_df  = pd.concat(all_preds, ignore_index=True)       # for LogLoss, ECE, AUC
predictions_DEU = pd.concat(all_preds_DEU, ignore_index=True)   # for DEU

predictions_df.to_csv(out_path("Preds"), index=False)
predictions_DEU.to_csv(out_path("Preds_DEU"), index=False)

model_cols = ['P_Y_alg', 'P_Y_exp', 'P_Y_obs', 'P_Y_all']

summary_table = bootstrap_metrics_summary(
    predictions_df, predictions_DEU, y_col='y_true', model_cols=model_cols,
    n_bins=10, n_boot=N_BOOT, random_state=SEED)

df_sorted = summary_table.sort_values(by=["metric", "model"]).reset_index(drop=True)
print(df_sorted)
df_sorted.to_csv(out_path("mean_of_metrics_results"), index=False)

summary = paired_bootstrap_summary_from_df(
    df=predictions_df,
    df_DEU=predictions_DEU,
    base_model='P_Y_alg',
    model_cols=model_cols,
    n_bins=10,
    n_boot=N_BOOT,
    random_state=SEED
)

print(summary)
summary.to_csv(out_path("hypothesis_testing_results"), index=False)

print("Wall-clock time: {:.2f} seconds".format(time.perf_counter() - start_time))
print("CPU time:        {:.2f} seconds".format(time.process_time() - start_cpu))
