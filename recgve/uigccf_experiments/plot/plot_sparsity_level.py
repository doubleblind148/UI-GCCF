#!/usr/bin/env python
__author__ = "XXX XXX"
__email__ = "XXX"

import wandb
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import matplotlib.ticker as ticker
from matplotlib.ticker import FormatStrFormatter

from utils.plot_utils import setup_plot

BASE_S = "density_{}_conv{}"
PROJECT = "ml1m_dat"
METRICS = ["Recall", "NDCG"]
CUTOFFS = [5, 20]

CONV_DEPTHS = [0, 1, 2, 3]
SPARSITY_LEVEL = [0.01, 0.005, 0.0001]
if __name__ == "__main__":
    setup_plot(241, style_sheet="ablation_dropout", fig_ratio=0.8)
    api = wandb.Api()
    project = f"{PROJECT}_density_performance"
    runs_path = f"XXX/{project}"
    runs = api.runs(runs_path)

    sps = []
    cd = []
    metric = []
    score = []
    for r in runs:
        for cdepth in CONV_DEPTHS:
            for sparsity in SPARSITY_LEVEL:
                if BASE_S.format(sparsity, cdepth) in r.name:
                    for k, v in r.summary.items():
                        if "@" in k:
                            sps.append(sparsity)
                            cd.append(cdepth)
                            metric.append(k)
                            score.append(v)

    res_df = pd.DataFrame(
        zip(map(str, sps), map(int, cd), metric, score),
        columns=["sparsity", "convolution depth", "metric", "score"],
    )

    for m in METRICS:
        for c in CUTOFFS:
            grp_df = res_df.groupby("metric").get_group(f"{m}@{c}")
            fix, ax = plt.subplots()

            palette = sns.color_palette("Blues_d", n_colors=4)
            sns.barplot(
                data=grp_df,
                y="score",
                x = "sparsity",
                order = ["0.0001", "0.005", "0.01"],
                dodge=True,
                hue="convolution depth",
                palette=palette,
            )

            ax.tick_params(direction="in")
            ax.set_ylabel(f"{m}@{c}")
            ax.set_xlabel("Density")
            #ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            ax.legend().set_title('cdepth')
            plt.tight_layout(pad=0.05)
            plt.gca().xaxis.grid(False)

            #plt.xticks([0.1, 0.3, 0.5, 0.7, 0.9])
            TITLE = f"density_{m}@{c}"
            print(TITLE)

            plt.savefig("{}/Desktop/{}_{}.pdf".format(os.environ["HOME"], PROJECT, TITLE))
            plt.show()
