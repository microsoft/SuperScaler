import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import csv

src_data_path = "results"
figure_save_path = "figures"

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

def read_perf_model_time(_file_name):
    file_name = os.path.join(src_data_path, _file_name)
    with open(file_name, 'r') as f:
        csv_reader = csv.reader(f)
        lines = list(csv_reader)
        _actual = lines[1][1:]
        _predict = lines[2][1:]
    actual = [float(i) for i in _actual]
    predict = [float(i) for i in _predict]
    return actual, predict

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

    # bar2 = ('Actual', np.array([190.58, 100.28, 84.6, 85.82, 86.57]))
    # bar3 = ('Prediction', np.array([200.71, 101.18, 87.36, 87.32, 84.61]))

    actual, predict = read_perf_model_time("perf_model_time_large_gpt.csv")
    bar2 = ('Actual', np.array(actual))
    bar3 = ('Prediction', np.array(predict))

    bar2_plot = ax.bar(inds - width / 2, np.array(bar2[1]), width, label=bar2[0], hatch='/', color=green, **line_config, zorder=100)
    bar3_plot = ax.bar(inds + width / 2, np.array(bar3[1]), width, label=bar3[0], color=blue, **line_config, zorder=100)
    ax.set_xlabel(xaxis[0], dict(size=22))
    ax.set_xticks(list(range(len(xaxis[1]))))
    ax.set_xticklabels(xaxis[1])
    ax.set_ylabel('Time (s)', dict(size=22))
    ax.set_yticks(np.arange(0, 250, 50))
    ax.tick_params(labelsize=22)

    plt.legend(bbox_to_anchor=(0.01, 0.97, 1.2, .12), loc='lower left', ncol=5,
               frameon=False, handlelength=1.5, handletextpad=0.2, columnspacing=0.6,
               borderaxespad=0., prop={'size': 21})
    fig.tight_layout()
    plt.subplots_adjust(left=0.22, bottom=0.24, right=0.98, top=0.85)

    # plt.show()
    if not os.path.exists(figure_save_path):
        os.makedirs(figure_save_path)
    fig.savefig(os.path.join(figure_save_path, "fig15a.pdf"), dpi=600)