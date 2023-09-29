import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import csv
from plot_fig15a import read_perf_model_time

src_data_path = "results"
figure_save_path = "figures"

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

if __name__ == "__main__":
    fig, ax = plt.subplots(figsize=(5, 3))
    plt.grid(color='lightgrey', linestyle='--', linewidth=0.5, zorder=0)
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    blue, orange, green, red = colors[:4]

    xaxis = ('# of GPUs', [1, 4, 8, 16, 32])
    inds = np.arange(len(xaxis[1]))
    width = 0.4
    line_config = dict(edgecolor='white')

    # bar2 = ('Actual', np.array([71.28, 53.15, 62.3, 48.68, 51.72]))
    # bar3 = ('Prediction', np.array([67.13, 49.33, 56.52, 45.67, 47.60]))

    actual, predict = read_perf_model_time("perf_model_time_large_resnet.csv")
    bar2 = ('Actual', np.array(actual))
    bar3 = ('Prediction', np.array(predict))

    bar2_plot = ax.bar(inds - width / 2, np.array(bar2[1]), width, label=bar2[0], hatch='/', color=green, **line_config, zorder=100)
    bar3_plot = ax.bar(inds + width / 2, np.array(bar3[1]), width, label=bar3[0], color=blue, **line_config, zorder=100)
    ax.set_xlabel(xaxis[0], dict(size=22))
    ax.set_xticks(list(range(len(xaxis[1]))))
    ax.set_xticklabels(xaxis[1])
    ax.tick_params(axis='x', labelsize=22)
    ax.set_ylabel('Time (s)', dict(size=22))
    ax.set_yticks([0, 20, 40, 60],
            ['0',  '20', '40', '  60'])
    ax.tick_params(axis='y', labelsize=22)

    plt.legend(bbox_to_anchor=(0.03, 0.97, 1.2, .12), loc='lower left', ncol=5,
               frameon=False, handlelength=1.5, handletextpad=0.2, columnspacing=0.6,
               borderaxespad=0., prop={'size': 21})
    fig.tight_layout()
    plt.subplots_adjust(left=0.22, bottom=0.24, right=0.98, top=0.85)

    # plt.show()
    if not os.path.exists(figure_save_path):
        os.makedirs(figure_save_path)
    fig.savefig(os.path.join(figure_save_path, "fig15b.pdf"), dpi=600)
