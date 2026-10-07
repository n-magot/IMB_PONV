# -*- coding: utf-8 -*-
"""
Figure of the paired bootstrap comparisons produced by run_experiments.py.

For each metric (AUROC, ECE, DEU) and each comparison method, the figure
shows the mean performance difference relative to IMB, with its 95%
bootstrap CI. Differences are oriented so that a POSITIVE value favours IMB.
A star marks comparisons whose CI excludes 0 and whose p-value is below
1 - CI_LEVEL.

Input : hypothesis_testing_results_<timestamp>.csv from run_experiments.py
"""

import pandas as pd
import matplotlib.pyplot as plt


# ===========================================================================
# USER INPUT
# ===========================================================================

RESULTS_FILE  = "results/hypothesis_testing_results.csv"   # output of run_experiments.py
OUTPUT_FIGURE = "results/bootstrap_results.pdf"            # set to None to only show
CI_LEVEL      = 0.95


# ===========================================================================
# Methods, metrics and display settings
# ===========================================================================

# Comparison methods, renamed for display. ATE_A1 / ATE_A2 are the observed
# P(Y = 0) in the trial arms T = 1 and T = 0.
method_names = {
    'P_Y_exp': 'MBe',
    'P_Y_obs': 'MBo',
    'P_Y_all': 'MBeo',
    'ATE_A1':  'ATE_A1',
    'ATE_A0':  'ATE_A2'
}

# Metrics, top to bottom
metric_order = ['AUROC', 'ECE', 'DEU']

# Methods within each metric group, top to bottom
method_order = ['MBeo', 'MBe', 'MBo', 'ATE_A1', 'ATE_A2']

# Paul Tol "bright" qualitative scheme (colour-blind safe)
colors = {
    'MBeo':   '#4477AA',  # blue
    'MBe':    '#228833',  # green
    'MBo':    '#AA3377',  # purple
    'ATE_A1': '#CCBB44',  # yellow
    'ATE_A2': '#66CCEE'   # cyan
}

# y-position of each metric group (first metric at the top)
y_base = {m: len(metric_order) - 1 - i for i, m in enumerate(metric_order)}

# Vertical offset of each method within a group
OFFSET_STEP = 0.15
y_offsets = {
    m: ((len(method_order) - 1) / 2 - i) * OFFSET_STEP
    for i, m in enumerate(method_order)
}


# ===========================================================================
# Data
# ===========================================================================

df = pd.read_csv(RESULTS_FILE)

df = df[df['other_model'].isin(method_names)].copy()
df['other_model_display'] = df['other_model'].replace(method_names)

df = df[df['metric'].isin(['AUC', 'ECE', 'DEU'])].copy()
df['metric'] = df['metric'].replace({'AUC': 'AUROC'})


# ===========================================================================
# Plot
# ===========================================================================

def plot_hypothesis_testing(df, show_dotted=False, capped_ci=True,
                            savepath=None, ci_level=0.95):
    """
    show_dotted : draw a dotted line from 0 to each mean difference.
    capped_ci   : draw the CI with end caps.
    savepath    : if given, the figure is also saved to this file.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for row in df.itertuples():

        y = y_base[row.metric] + y_offsets[row.other_model_display]
        color = colors[row.other_model_display]

        # Confidence interval
        if capped_ci:
            ax.plot([row.ci_lower, row.ci_upper], [y, y],
                    color=color, lw=2, alpha=0.6,
                    marker='|', markersize=10, markeredgewidth=1.5)
        else:
            ax.plot([row.ci_lower, row.ci_upper], [y, y],
                    color=color, lw=2, alpha=0.9)

        # Mean difference
        ax.plot(row.delta_obs, y, 'o', color=color, markersize=8)

        if show_dotted:
            ax.plot([0, row.delta_obs], [y, y],
                    linestyle=':', color='gray', linewidth=1)

        # Significance star: CI excludes 0 and p < 1 - ci_level
        if (row.ci_lower > 0 or row.ci_upper < 0) and row.p_value < 1 - ci_level:
            ax.text(row.delta_obs, y + 0.03, '*',
                    color='black', fontsize=14, ha='center')

    # y-axis: one tick per metric, light separators between groups
    ax.set_yticks([y_base[m] for m in metric_order])
    ax.set_yticklabels(metric_order)
    for k in range(len(metric_order) - 1):
        ax.axhline(k + 0.5, color='lightgray', linewidth=0.8)
    ax.set_ylim(-0.5, len(metric_order) - 0.5)

    # Zero line and x-axis label
    ax.axvline(0, color='black', linestyle='--')
    ax.set_xlabel("Performance difference relative to IMB", fontsize=12, labelpad=10)

    # Legend
    handles = [
        plt.Line2D([0], [0], marker='o', color=colors[m], lw=2,
                   markerfacecolor=colors[m], markersize=8, label=m)
        for m in method_order
    ]
    ax.legend(handles=handles, title='Method',
              loc='upper right', bbox_to_anchor=(0.98, 0.98))

    plt.tight_layout()

    if savepath:
        fig.savefig(savepath)

    plt.show()


plot_hypothesis_testing(df, savepath=OUTPUT_FIGURE, ci_level=CI_LEVEL)
