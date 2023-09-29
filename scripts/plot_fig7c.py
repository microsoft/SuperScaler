import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import csv 
from plot_fig7a import read_thpt

src_data_path = "results"
figure_save_path = "figures"

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

if __name__ == "__main__":
    fig, ax = plt.subplots(figsize=(7, 3))
    plt.grid(color='lightgrey', linestyle='--', linewidth=0.5, zorder=0)

    width = 0.15
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    blue, orange, green, red = colors[:4]


    index = ['0.77', '3', '6', '11', '22']
    nbars = 2
    inds = np.arange(len(index))
    width = 0.4
    line_config = dict(edgecolor='white')

    # new results for eurosys
    # bar1 = ('Megatron-LM', [1, 1, 1, 1, 1])
    # bar3 = ('Aceso', [1.00, 1.47, 1.39, 1.50, 1.35])

    norm_megatron_thpt, _, norm_aceso_thpt = read_thpt("e2e_norm_thpt_large_t5.csv")

    bar1 = ('Megatron-LM', norm_megatron_thpt)
    bar3 = ('Aceso', norm_aceso_thpt)

    ax.bar(inds - width / 2, np.array(bar1[1]), width, label=bar1[0], hatch='+', color=orange, **line_config, zorder=100)
    ax.bar(inds + width / 2, np.array(bar3[1]), width, label=bar3[0], color=blue, **line_config, zorder=100)
    ax.set_xlabel('Model Size (Billion)', dict(size=23))
    ax.set_xticks(list(range(len(index))))
    ax.set_xticklabels(index)
    ax.tick_params(axis='x', labelsize=23)
    ax.set_ylabel(r"Normalized Thpt.  ", dict(size=23))
    ax.set_yticks(np.arange(0, 1.7, 0.4))
    ax.tick_params(axis='y', labelsize=23)

    plt.legend(bbox_to_anchor=(0.14, 1.001, 1.2, .12), loc='lower left', ncol=5,
               frameon=False, handlelength=0.7, handletextpad=0.2, columnspacing=0.6,
               borderaxespad=0., prop={'size': 23})

    fig.tight_layout()
    plt.subplots_adjust(left=0.15, bottom=0.256, right=0.985, top=0.82)

    if not os.path.exists(figure_save_path):
        os.makedirs(figure_save_path)
    fig.savefig(os.path.join(figure_save_path, "fig7c.pdf"), dpi=600)