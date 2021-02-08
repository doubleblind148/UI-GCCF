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

BASE_UPD = "valid_user_profile_dropout_"
BASE_ED = "valid_edge_dropout_"
PROJECTS = ["lastfm_dat", "ml1m_dat", "Amaz_dat", "gowalla_dat"]
PROJECT = "Amaz_dat"
METRICS = ["Recall", "NDCG"]
CUTOFFS = [5, 20]

if __name__ == "__main__":
    setup_plot(241, style_sheet="ablation_dropout", fig_ratio=0.8)
    for p in PROJECTS:
        api = wandb.Api()
        project = f"ablation_{p}"
        runs_path = f"XXX/{project}"
        runs = api.runs(runs_path)

        kind = []
        dropout = []
        metric = []
        score = []
        for r in runs:
            if BASE_UPD in r.name or BASE_ED in r.name:
                run_name = r.name.split("_")
                for k, v in r.summary.items():
                    if "@" in k:
                        kind.append(str.join("_", run_name[:-1]))
                        dropout.append(run_name[-1])
                        metric.append(k)
                        score.append(v)

        res_df = pd.DataFrame(
            zip(kind, map(float, dropout), metric, score),
            columns=["kind", "dropout", "metric", "score"],
        )

        line_style = {
            "valid_edge_dropout": [1, 0],
            "valid_user_profile_dropout": [1, 0.7],
        }

        colors = {
            "valid_edge_dropout": "darkblue",
            "valid_user_profile_dropout": "orange",
        }

        markers = {
            "valid_edge_dropout": ".",
            "valid_user_profile_dropout": ".",
        }

        for m in METRICS:
            for c in CUTOFFS:
                grp_df = res_df.groupby("metric").get_group(f"{m}@{c}")
                fix, ax = plt.subplots()

                sns.lineplot(
                    data=grp_df,
                    x="dropout",
                    y="score",
                    hue="kind",
                    sort=True,
                    marker="o",
                    dashes=line_style,
                    style="kind",
                    palette=colors,
                    markers=markers,
                )
                ax.tick_params(direction="in")
                ax.set_ylabel(f"{m}@{c}")
                ax.set_xlabel("Dropout")
                #ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                ax.legend().set_title('')
                plt.tight_layout(pad=0.05)
                plt.gca().xaxis.grid(False)
                ax.set(xticks=grp_df.dropout.values)
                for t in ax.legend_.texts: t.set_text(t.get_text().replace("_", " ").replace("valid", ""))
                if p != "lastfm_dat":
                    ax.set_ylabel("")
                    ax.get_legend().remove()

                #plt.xticks([0.1, 0.3, 0.5, 0.7, 0.9])
                TITLE = f"ablation_dropout_{m}@{c}"
                print(TITLE)

                plt.savefig("{}/Desktop/{}_{}.pdf".format(os.environ["HOME"], p, TITLE))
                plt.show()
