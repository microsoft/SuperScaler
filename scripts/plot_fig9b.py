import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import csv

src_data_path = "results"
figure_save_path = "figures"

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

def read_data(_file_name):
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
    fig, ax = plt.subplots(figsize=(5, 3))
    plt.grid(color='lightgrey', linestyle='--', linewidth=0.5, zorder=0)
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    blue, orange, green, red = colors[:4]

    baseline = ["layer number", "Megatron", "Alpa", 'Aceso']

    layer_number = [8, 16, 32, 64, 128, 256, 512, 1024]
    # megatron = [201.50, 118.89, 73.34, 34.95, 16.30, 8.56, 4.38, 2.21]
    # alpa = [210.61, 109.93, 67.77, 34.25, 0, 0, 0, 0]
    # aceso = [227.08, 143.51, 80.76, 41.37, 20.15, 9.64, 4.57, 2.23]
    megatron, alpa, aceso = read_data("e2e_thpt_scale_scale-layer.csv")

    gpus = ('Layer Number', layer_number)
    megatron = ('Megatron-LM', megatron)
    alpa = ("Alpa", alpa)
    aceso = ('Aceso', aceso)

    nbars = 4
    inds = np.arange(len(gpus[1]))

    width = 1 / nbars
    line_config = dict(edgecolor='white')

    plt.bar(inds - (width / 2 + width), np.array(megatron[1]), width, hatch='+', color=orange, **line_config,label="Megatron-LM")
    plt.bar(inds - (width / 2), np.array(alpa[1]), width, hatch='/', color=green, **line_config, label='Alpa')
    plt.bar(inds + (width / 2), np.array(aceso[1]), width, color=blue, **line_config, label='Aceso')

    plt.text(4 - (width / 2) - 0.15, 1.2, 'X', dict(font='DejaVu Sans', size=17, color=green))
    plt.text(5 - (width / 2) - 0.15, 1.2, 'X', dict(font='DejaVu Sans', size=17, color=green))
    plt.text(6 - (width / 2) - 0.15, 1.2, 'X', dict(font='DejaVu Sans', size=17, color=green))
    plt.text(7 - (width / 2) - 0.15, 1.2, 'X', dict(font='DejaVu Sans', size=17, color=green))

    plt.xlabel('# of Layers', dict(size=22))
    plt.xticks(inds, gpus[1], fontsize=22)
    plt.subplots_adjust(left=0.2, right=0.8, top=0.7, bottom=0.3)
    plt.ylabel('Thpt. (samples/s)       ', dict(size=22))
    plt.grid(axis="y", linewidth=0.15, linestyle='--')
    plt.yticks(np.arange(0, 210, 100), fontsize=22)
 
    plt.legend(bbox_to_anchor=(-0.27, 0.97, 1.2, .12), loc='lower left', ncol=5,
               frameon=False, handlelength=1.0, handletextpad=0.2, columnspacing=0.6,
               borderaxespad=0., prop={'size': 21})

    ax.tick_params(axis='x', labelsize=22)
    ax.tick_params(axis='y', labelsize=22)

    plt.yticks([0, 50, 100, 150, 200],
               ['0', '', '100', '', '200'])
    plt.xticks([0, 1, 2, 3, 4, 5, 6, 7],
               ['$2^3$', '$2^4$', '$2^5$', '$2^6$', '$2^7$', '$2^8$', '$2^9$', '$2^{10}$'])

    fig.tight_layout()
    plt.subplots_adjust(left=0.21, bottom=0.255, right=0.985, top=0.87)

    if not os.path.exists(figure_save_path):
        os.makedirs(figure_save_path)
    fig.savefig(os.path.join(figure_save_path, "fig9b.pdf"), dpi=600)

