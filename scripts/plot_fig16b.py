import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import csv
from plot_fig16a import read_perf_model_mem

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
    nbars = 2  # 3
    inds = np.arange(len(xaxis[1]))
    width = 0.4
    line_config = dict(edgecolor='white')

    # bar2 = ('Actual', np.array([16516, 29558, 29130, 29004, 29004]) / 1024)
    # bar3 = ('Prediction', np.array([16519, 22355, 21536, 22889, 24923]) / 1024)
    # bar4 = ('Prediction-extra-mem', np.array([3432, 4933, 5412, 4380, 2989]) / 1024)

    actual, predict_norm, predict_extra = read_perf_model_mem("perf_model_mem_large_resnet.csv")
    bar2 = ('Actual', np.array(actual) / 1024)
    bar3 = ('Prediction', np.array(predict_norm) / 1024)
    bar4 = ('Prediction-extra-mem', np.array(predict_extra) / 1024)

    bar2_plot = ax.bar(inds - width / 2, np.array(bar2[1]), width, hatch='/', color=green, **line_config, zorder=100)
    bar3_plot = ax.bar(inds + width / 2, np.array(bar3[1]), width, color=blue, **line_config, zorder=100)
    bar4_plot = ax.bar(inds + width / 2, np.array(bar4[1]), width, label=bar4[0], bottom=bar3[1], color=orange, zorder=100,
                       **line_config)
    ax.set_xlabel(xaxis[0], dict(size=22))
    ax.set_xticks(list(range(len(xaxis[1]))))
    ax.set_xticklabels(xaxis[1])
    ax.set_ylabel('Memory (GB)  ', dict(size=22))
    ax.tick_params(labelsize=22)
    plt.legend(bbox_to_anchor=(-0.31, 0.97, 1.2, .12), loc='lower left', ncol=5,
               frameon=False, handlelength=1.5, handletextpad=0.2, columnspacing=0.6,
               borderaxespad=0., prop={'size': 21})
    ax.set_ylim(0, 32)
    plt.yticks([0, 5, 10, 15, 20, 25, 30, ],
               ['0', '', '10', '', '20', '', '  30'])
    fig.tight_layout()
    plt.subplots_adjust(left=0.22, bottom=0.24, right=0.98, top=0.85)

    # plt.show()
    if not os.path.exists(figure_save_path):
        os.makedirs(figure_save_path)
    fig.savefig(os.path.join(figure_save_path, "fig16b.pdf"), dpi=600)