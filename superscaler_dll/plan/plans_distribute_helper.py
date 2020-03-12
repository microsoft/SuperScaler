import os
import shutil
import plans_distribute

mode = 0  # 0 for append, 1 for override

mapping_hostname_file_path = "./mapping_hostname.cfg"
mapping_port_file_path = "./mapping_port.cfg"

plan_output_dir = "/data/data5/ftp/files/users/v-guanx/models_test/"
model_test_dir = "/data/data5/ftp/files/users/v-guanx/models_test/"

for root, dirs, files in os.walk(model_test_dir):
    for dir_name in dirs:
        # DataParallelismPlan2GPUsIn1Hosts  21
        # DataParallelismPlan2GPUsIn2Hosts  21 25
        # DataParallelismPlan4GPUsIn2Hosts  21 25
        # DataParallelismPlan4GPUsIn4Hosts  21 22 24 25
        if dir_name == "ModelParallelismPlan2GPUsIn2Hosts":
            plan_dir = os.path.join(root, dir_name)
            json_file_path = os.path.join(plan_dir, "exec_plan", "run.json")

            print (plan_dir)
            plans_distribute.plans_distribute(mapping_hostname_file_path, mapping_port_file_path, json_file_path, plan_dir)

print ("plan distribute Done")
# cd path/to/plan
# python plans_distribute_helper.py
