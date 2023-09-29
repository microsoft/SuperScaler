import sys
import os 
import csv
from get_common import get_model_sizes, get_search_cost, get_alpa_search_cost, _format

exp_setting = sys.argv[1]
result_path = "results/"
if not os.path.exists(result_path):
    os.makedirs(result_path)

if exp_setting == "small":
    log_path = "logs"
    model_names = ["gpt", "t5", "resnet"]
elif exp_setting == "large":
    log_path = "logs-large"
    model_names = ["gpt", "t5", "resnet"]
elif exp_setting == "scale":
    log_path = "logs-large"
    model_names = ["scale-layer"]

for model_name in model_names:
    all_time = {}
    model_sizes = get_model_sizes(model_name, exp_setting)
    if exp_setting == "large":
        model_sizes = model_sizes[1:]
    alpa_time, aceso_time = [], []
    for model_size in model_sizes:
        tmp_result = ['-', '-']
        add_result = False
        ## Alpa
        cost = get_alpa_search_cost(log_path, model_name, model_size)
        alpa_time.append(cost)
        if cost > 0:
            add_result = True
            tmp_result[0] = f"{cost:.2f}"

        ## Aceso
        cost = get_search_cost(log_path, "aceso", model_name, model_size)
        aceso_time.append(cost)
        if cost > 0:
            add_result = True
            tmp_result[1] = f"{cost:.2f}"

        if add_result:
            all_time[model_size] = list(tmp_result)

    if len(all_time) > 0:
        print(f"-------- [{model_name}] Search Cost (s) --------")
        print(f"Size\t Alpa\t Aceso")

        for model_size in model_sizes:
            if model_size in all_time:
                print(f"{model_size}\t{all_time[model_size][0]}\t {all_time[model_size][1]}")

    ## Save into csv files for plotting figures
    file_name = os.path.join(result_path, f"search_cost_{exp_setting}_{model_name}.csv")
    info_to_csv = [["size"] + model_sizes, ["alpa"] + _format(alpa_time), ["aceso"] + _format(aceso_time)]
    with open(file_name, mode="w", newline="") as file:
        writer = csv.writer(file)
        for row in info_to_csv:
            writer.writerow(row)