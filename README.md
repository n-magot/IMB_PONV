# IMB_PONV

Code for the Postoperative Nausea and Vomiting (PONV) experiments in the paper:

> A Bayesian Approach for Combining EHR and RCT Data to Predict Conditional Average Treatment Effects: Methodology and Evaluation

The method predicts the effect of a treatment by combining **observational** (EHR) and **experimental** (RCT) data.

For each set of covariates Z, the method weighs two hypotheses:

- the observational and experimental data agree given Z, so both can be pooled;
- they disagree, so only the experimental data are used.

Predictions of P(Y | do(T), Z) are averaged over the sets Z and the two hypotheses. They are compared with three baselines:

- experimental data only;
- observational data only;
- both datasets pooled.

## Requirements

```
pip install numpy pandas scipy scikit-learn matplotlib
```

## Data

Two CSV files with the same columns:

- `observational_data.csv`
- `experimental_data.csv`

The data must satisfy these conditions:

- All variables are discrete. Discretize continuous ones first.
- The treatment is coded 0/1.
- The outcome is coded 0/1, and **0 is the good outcome**.
- Every other column is used as a candidate covariate, so remove IDs and dates first.

## Run

Set the paths and column names at the top of `run_experiments.py`, then:

```
python run_experiments.py
```

To plot the bootstrap comparisons, set `RESULTS_FILE` at the top of `plot_bootstrap_results.py` to the `hypothesis_testing_results_*.csv` file, then:

```
python plot_bootstrap_results.py
```

## Output

All results are written to `results/`.

| File | Contents |
|---|---|
| `cv_metrics_*.csv` | Per-fold log-loss, ECE, AUC and DEU for each method |
| `mean_of_metrics_results_*.csv` | Bootstrap means and 95% CIs. Used for Table 3 (mean performance of each method with 95% CIs) |
| `hypothesis_testing_results_*.csv` | Paired bootstrap comparisons with the baselines. Used for Table 4 (difference in performance of each method to IMB) |
| `best_sets_*.csv`, `Variable_importance_*.csv` | Selected covariate sets |
| `Preds_*.csv`, `Preds_DEU_*.csv` | Per-patient predictions |
| `bootstrap_results.pdf` | Figure of the paired bootstrap comparisons |

## Citation

To be added.
