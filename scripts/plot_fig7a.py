import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import csv 

src_data_path = "results"
figure_save_path = "figures"

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

def read_thpt(_file_name):
    file_name = os.path.join(src_data_path, _file_name)
    with open(file_name, 'r') as f:
        csv_reader = csv.reader(f)
        lines = list(csv_reader)
        _megatron_thpt = lines[1][1:]
        _alpa_thpt = lines[2][1:]
        _aceso_thpt = lines[3][1:]
    megatron_thpt = [float(i) for i in _megatron_thpt]
    alpa_thpt = [float(i) for i in _alpa_thpt]
    aceso_thpt = [float(i) for i in _aceso_thpt]

    return megatron_thpt, alpa_thpt, aceso_thpt

if __name__ == "__main__":

    fig, ax = plt.subplots(figsize=(7, 3))
    plt.grid(color='lightgrey', linestyle='--', linewidth=0.5, zorder=0)

    width = 0.15
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    blue, orange, green, red = colors[:4]

    index = ['0.35', '1.3', '2.6', '6.7', '13']
    nbars = 3
    inds = np.arange(len(index))
    width = 0.8 / nbars
    line_config = dict(edgecolor='white')

    norm_megatron_thpt, norm_alpa_thpt, norm_aceso_thpt = read_thpt("e2e_norm_thpt_large_gpt.csv")

    # new results for eurosys
    # bar1 = ('Megatron-LM', [1, 1, 1, 1, 1])
    # bar2 = ('Alpa', [0.93, 0.97, 1.01, 1.06, 1.10])
    # bar3 = ('Aceso', [1.00, 1.22, 1.28, 1.31, 1.28])

    bar1 = ('Megatron-LM', norm_megatron_thpt)
    bar2 = ('Alpa', norm_alpa_thpt)
    bar3 = ('Aceso', norm_aceso_thpt)

    ax.bar(inds - width, np.array(bar1[1]), width, label=bar1[0], hatch='+', color=orange, **line_config, zorder=100)
    ax.bar(inds, np.array(bar2[1]), width, label=bar2[0], hatch='/', color=green, **line_config, zorder=100)
    ax.bar(inds + width, np.array(bar3[1]), width, label=bar3[0], color=blue, **line_config, zorder=100)
    ax.set_xlabel('Model Size (Billion)', dict(size=23))
    ax.set_xticks(list(range(len(index))))
    ax.set_xticklabels(index)
    ax.tick_params(axis='x', labelsize=23)
    ax.set_ylabel(r"Normalized Thpt.  ", dict(size=23))
    ax.set_yticks(np.arange(0, 1.6, 0.5))
    ax.tick_params(axis='y', labelsize=23)

    plt.legend(bbox_to_anchor=(0.06, 1.001, 1.2, .12), loc='lower left', ncol=5,
               frameon=False, handlelength=0.7, handletextpad=0.2, columnspacing=0.6,
               borderaxespad=0., prop={'size': 23})

    fig.tight_layout()
    plt.subplots_adjust(left=0.15, bottom=0.256, right=0.985, top=0.82)

    # plt.show()
    if not os.path.exists(figure_save_path):
        os.makedirs(figure_save_path)
    fig.savefig(os.path.join(figure_save_path, "fig7a.pdf"), dpi=600)