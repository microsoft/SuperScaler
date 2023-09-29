# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import sys
import os 
import csv 

model_name = sys.argv[1]
src_data_path = sys.argv[2]

all_data = []

for model_size in os.listdir(os.path.join(src_data_path, "runtime", model_name)):
    file_path = os.path.join(src_data_path, "runtime", model_name, model_size, "csv")
    if os.path.exists(file_path):
        files = [f for f in os.listdir(file_path) if os.path.isfile(os.path.join(file_path, f))]
        best_result = [None, 0,0,0, None]
        max_thpt = 0
        for file_name in files:
            with open(os.path.join(file_path, file_name), 'r') as f:
                csv_reader = csv.reader(f)
                next(csv_reader)
                for row in csv_reader:
                    global_bs = float(row[0])
                    time = float(row[1])/1000
                    thpt = global_bs/time
                    config_name = os.path.join(src_data_path, "configs", model_name, model_size, file_name.split("_stage")[0] + ".json")
                    if thpt > max_thpt:
                        best_result = [model_size, global_bs, time, thpt, config_name]
                        max_thpt = thpt
        all_data.append(best_result)
    else:
        print(f"{file_path} not exist")
        all_data.append([None, 0,0,0, None])

print(f"-------- {model_name} End-to-end throughput --------")
print(f"Size\t Batch Size\t Time(s)\t Thpt(samples/s)")
for data in all_data:
    if data[0] is not None:
        print(f"{data[0]}\t {data[1]}\t\t {data[2]:.2f}\t\t {data[3]:.2f}")

print(f"Config")
for data in all_data:
    if data[0] is not None:
        print(f"{data[4]}")