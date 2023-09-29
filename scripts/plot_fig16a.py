import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import csv

src_data_path = "results"
figure_save_path = "figures"

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

def read_perf_model_mem(_file_name):
    file_name = os.path.join(src_data_path, _file_name)
    with open(file_name, 'r') as f:
        csv_reader = csv.reader(f)
        lines = list(csv_reader)
        _actual = lines[1][1:]
        _predict_norm = lines[2][1:]
        _predict_extra = lines[3][1:]
    actual = [float(i) for i in _actual]
    predict_norm = [float(i) for i in _predict_norm]
    predict_extra = [float(i) for i in _predict_extra]
    return actual, predict_norm, predict_extra

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

    # bar2 = ('Actual', np.array([14820, 22082, 24748, 26858, 27058]) / 1024)
    # bar3 = ('Prediction', np.array([14592, 25152, 26308, 25281, 26340]) / 1024)
    # bar4 = ('Prediction-extra-mem', np.array([5120, 1280, 1280, 2560, 1650]) / 1024)

    actual, predict_norm, predict_extra = read_perf_model_mem("perf_model_mem_large_gpt.csv")
    bar2 = ('Actual', np.array(actual) / 1024)
    bar3 = ('Prediction', np.array(predict_norm) / 1024)
    bar4 = ('Prediction-extra-mem', np.array(predict_extra) / 1024)

    bar2_plot = ax.bar(inds - width / 2, np.array(bar2[1]), width, label=bar2[0], hatch='/', color=green, **line_config, zorder=100)
    bar3_plot = ax.bar(inds + width / 2, np.array(bar3[1]), width, label=bar3[0], color=blue, **line_config, zorder=100)
    bar4_plot = ax.bar(inds + width / 2, np.array(bar4[1]), width, bottom=bar3[1], color=orange, zorder=100,
                       **line_config)
    ax.set_xlabel(xaxis[0], dict(size=22))
    ax.set_xticks(list(range(len(xaxis[1]))))
    ax.set_xticklabels(xaxis[1])
    ax.set_ylabel('Memory (GB)  ', dict(size=22))
    ax.tick_params(labelsize=22)
    plt.legend(bbox_to_anchor=(0.06, 0.97, 1.2, .12), loc='lower left', ncol=5,
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
    fig.savefig(os.path.join(figure_save_path, "fig16a.pdf"), dpi=600)
