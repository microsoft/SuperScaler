import os
import csv
from datetime import datetime

def _format(src_list):
    new_list = []
    for i in src_list:
        new_list.append("{:.2f}".format(i))
    
    return new_list

def get_model_sizes(model_name, exp_setting):
    #### small-scale
    if exp_setting == "small":
        if model_name == "gpt":
            model_sizes = ["1_3B"]
        elif model_name == "t5":
            model_sizes = ["770M"]
        elif model_name == "resnet":
            model_sizes = ["1B"]
    #### large-scale
    elif exp_setting == "large":
        if model_name == "gpt":
            model_sizes = ["350M", "1_3B", "2_6B", "6_7B", "13B"]
        elif model_name == "t5":
            model_sizes = ["770M", "3B", "6B", "11B", "22B"]
        elif model_name == "resnet":
            model_sizes = ["500M", "2B", "4B", "6_8B", "13B"]
    #### scale-to-1K-layer 
    elif exp_setting == "scale":
        assert model_name == "scale-layer"
        model_sizes = [f"{n}layers" for n in [8, 16, 32, 64, 128, 256, 512, 1024]]

    return model_sizes #, num_gpus

def get_thpt(log_path, runtime, model_name, model_size):

    file_path = os.path.join(log_path, runtime, "runtime", model_name, model_size, "csv")
    max_thpt = 0
    min_time = 0
    if os.path.exists(file_path):
        files = [f for f in os.listdir(file_path) if os.path.isfile(os.path.join(file_path, f))]
        for file_name in files:
            with open(os.path.join(file_path, file_name), 'r') as f:
                csv_reader = csv.reader(f)
                next(csv_reader)
                for row in csv_reader:
                    global_bs = float(row[0])
                    time = float(row[1])/1000
                    thpt = global_bs/time
                    if thpt > max_thpt:
                        max_thpt = thpt
                        min_time = time

    return max_thpt, min_time

def get_alpa_thpt(log_path, model_name, model_size):
    if model_name == "resnet":
        model_name = "wresnet"
    file_path = os.path.join(log_path, "alpa", "runtime", model_name, model_size)
    max_thpt = 0
    min_time = 0
    if os.path.exists(file_path):
        files = [f for f in os.listdir(file_path) if os.path.isfile(os.path.join(file_path, f))]
        for file_name in files:
            with open(os.path.join(file_path, file_name), 'r') as f:
                csv_reader = csv.reader(f, delimiter='\t')
                for row in csv_reader:
                    time = float(row[5].split("s")[0])
                    global_bs = int(row[12])
                    thpt = global_bs/time
                    if thpt > max_thpt:
                        max_thpt = thpt
                        min_time = time
    return max_thpt, min_time

def get_normalized_thpt(log_path, model_name, exp_setting):

    model_sizes = get_model_sizes(model_name, exp_setting)

    megatron_thpt = []
    for model_size in model_sizes:
        _thpt, _ = get_thpt(log_path, "megatron", model_name, model_size)
        megatron_thpt.append(_thpt)
    alpa_thpt = []
    for model_size in model_sizes:
        _thpt, _ = get_alpa_thpt(log_path, model_name, model_size)
        alpa_thpt.append(_thpt)
    aceso_thpt = []
    for model_size in model_sizes:
        _thpt, _ = get_thpt(log_path, "aceso", model_name, model_size)
        aceso_thpt.append(_thpt)

    norm_megatron_thpt = [0 for i in range(len(megatron_thpt))]
    norm_alpa_thpt = [0 for i in range(len(megatron_thpt))]
    norm_aceso_thpt = [0 for i in range(len(megatron_thpt))]
    for i in range(len(megatron_thpt)):
        if megatron_thpt[i] > 0:
            norm_megatron_thpt[i] = 1
            norm_alpa_thpt[i] = alpa_thpt[i]/megatron_thpt[i]
            norm_aceso_thpt[i] = aceso_thpt[i]/megatron_thpt[i]
    
    return norm_megatron_thpt, norm_alpa_thpt, norm_aceso_thpt

def get_latest_file(folder_path, keyword=None):
    latest_file = None
    latest_modification_time = datetime.min

    for file_name in os.listdir(folder_path):
        if keyword is None or keyword in file_name:
            file_path = os.path.join(folder_path, file_name)

            if os.path.isfile(file_path):
                modification_time = os.path.getmtime(file_path)
                modification_datetime = datetime.fromtimestamp(modification_time)
                if modification_datetime > latest_modification_time:
                    latest_modification_time = modification_datetime
                    latest_file = file_path

    return latest_file

def get_alpa_search_cost(log_path, model_name, model_size):
    if model_name == "resnet":
        model_name = "wresnet"

    file_path = os.path.join(log_path, "alpa", "search", model_name, model_size)
    if os.path.exists(file_path):
        latest_summary_file = get_latest_file(file_path, "summary")
        with open(latest_summary_file, 'r') as f:
            csv_reader = csv.reader(f, delimiter='\t')
            lines = list(csv_reader)
            return float(lines[1][0])
    
    return 0

def get_search_cost(log_path, runtime, model_name, model_size):

    file_path = os.path.join(log_path, runtime, "search", model_name, model_size)
    if os.path.exists(file_path):
        latest_summary_file = get_latest_file(file_path, "summary")
        with open(latest_summary_file, 'r') as f:
            csv_reader = csv.reader(f, delimiter='\t')
            lines = list(csv_reader)
            return float(lines[1][0])
    return 0

def get_all_search_cost(log_path, model_name):

    model_sizes = get_model_sizes(model_name, "small")

    if len(model_sizes) > 1:
        model_sizes = model_sizes[1:]

    alpa_all_cost = []
    aceso_all_cost = []

    for model_size in model_sizes:
        alpa_cost = get_alpa_search_cost(log_path, model_name, model_size)
        if alpa_cost is not None:
            alpa_all_cost.append(float(alpa_cost))
        else:
            alpa_all_cost.append(0)
        aceso_cost = get_search_cost(log_path, "aceso", model_name, model_size)
        if aceso_cost is not None:
            aceso_all_cost.append(float(aceso_cost))
        else:
            aceso_all_cost.append(0)

    return alpa_all_cost, aceso_all_cost

def get_best_config(log_path, runtime, model_name, model_size):
    file_path = os.path.join(log_path, runtime, "runtime", model_name, model_size, "csv")
    best_config = None
    if os.path.exists(file_path):
        files = [f for f in os.listdir(file_path) if os.path.isfile(os.path.join(file_path, f))]
        max_thpt = 0
        for file_name in files:
            config_name = file_name.split("_stage")[0]
            with open(os.path.join(file_path, file_name), 'r') as f:
                csv_reader = csv.reader(f)
                next(csv_reader)
                for row in csv_reader:
                    global_bs = float(row[0])
                    time = float(row[1])/1000
                    thpt = global_bs/time
                    if thpt > max_thpt:
                        best_config = config_name
                        max_thpt = thpt
    return best_config