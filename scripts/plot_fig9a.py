import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import csv
from plot_fig8a import read_search_cost

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

    xaxis = ('# of GPU', [4, 8, 16, 32])
    nbars = 2
    inds = np.arange(len(xaxis[1]))
    width = 0.38
    line_config = dict(edgecolor='white')

    # alpa = np.array([1245.68, 2330.36, 4417.7059, 10504.64]) / 60 /60
    # aceso = np.array([23.59, 36.78, 200.13, 200.37, 200.33, 203.22, 219.61, 246.05]) / 60 / 60

    alpa_cost, aceso_cost = read_search_cost("search_cost_scale_scale-layer.csv")
    alpa = np.array(alpa_cost[:4]) / 60 /60
    aceso = np.array(aceso_cost) / 60 /60

    alpa_data = dict(name='Alpa',
                     search_time=np.array(alpa),  # to seconds
                     size=[0, 1, 2, 3])

    aceso_data = dict(name='Aceso',
                      search_time=np.array(aceso),  # to seconds
                      size=[0, 1, 2, 3, 4, 5, 6, 7])

    labels = ["Alpa", 'Aceso']
    line_config = dict(linewidth=4)
    line1 = plt.plot(alpa_data['size'], alpa_data['search_time'], linestyle='solid', marker='.', markersize=25,
                     color=green, **line_config, label='Alpa')
    line2 = plt.plot(aceso_data['size'], aceso_data['search_time'], linestyle='solid', marker='s', markersize=12,
                     color=blue, **line_config, label='Aceso')

    plt.legend(fontsize=22, loc="upper right", ncol=1, labels=labels,
               frameon=False)  # loc='lower right', #bbox_to_anchor=(0.9, 0.90)
    plt.xlabel('# of Layers', dict(size=22))

    plt.ylabel('Search Time(Hour)   ', dict(size=22))
    plt.grid(axis="y", linewidth=0.1, linestyle='--')
    name = ["8", "16", "32", "64", "128", "256", "512", "1024"]
    ors = np.array([0, 1, 2, 3, 4, 5, 6, 7])
    plt.xticks(ors, name, fontsize=22)

    plt.legend(bbox_to_anchor=(0.17, 0.97, 1.2, .12), loc='lower left', ncol=5,
               frameon=False, handlelength=1.0, handletextpad=0.2, columnspacing=0.6,
               borderaxespad=0., prop={'size': 21})

    ax.tick_params(axis='x', labelsize=22)
    ax.tick_params(axis='y', labelsize=22)

    plt.ylim((-0.2, 3.2))
    plt.yticks([0, 1, 2, 3],
               ['0', '1', '2', '3'])
    plt.xticks([0, 1, 2, 3, 4, 5, 6, 7],
               ['$2^3$', '$2^4$', '$2^5$', '$2^6$', '$2^7$', '$2^8$', '$2^9$', '$2^{10}$'])

    fig.tight_layout()
    plt.subplots_adjust(left=0.135, bottom=0.255, right=0.985, top=0.87)

    if not os.path.exists(figure_save_path):
        os.makedirs(figure_save_path)
    fig.savefig(os.path.join(figure_save_path, "fig9a.pdf"), dpi=600)
