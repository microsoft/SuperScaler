import json

def split_hostname_gpu(str, mapping_hostname_dict, mapping_port_dict):
    strs = str.split('/')
    host_name = strs[1]
    host_name = mapping_hostname_dict[host_name]
    gpu_name = strs[3]
    host_port = mapping_port_dict[gpu_name]
    return host_name, gpu_name, host_port

def plans_distribute(mapping_hostname_file_path, mapping_port_file_path, plans_file_path, plan_out_dir):
    mapping_hostname_dict = {}
    with open(mapping_hostname_file_path, 'r') as mapping:
        all_mappings = mapping.readlines()
        for item in all_mappings:
            items = item.split(' ')
            host_name = items[0]
            host_IP = items[1].replace('\n', '')
            mapping_hostname_dict[host_name] = host_IP

    mapping_port_dict = {}
    with open(mapping_port_file_path, 'r') as mapping:
        all_mappings = mapping.readlines()
        for item in all_mappings:
            items = item.split(' ')
            host_gpu = items[0]
            host_port = items[1].replace('\n', '')
            mapping_port_dict[host_gpu] = host_port

    with open(plans_file_path, 'r') as plans:
        all_plans = json.load(plans)

    for host_key, host_value in all_plans.items():
        host_name, gpu_name, host_port = split_hostname_gpu(host_key, mapping_hostname_dict, mapping_port_dict)

        for device_key, device_value in host_value.items():
            device_name = device_key[7:]

            cfg_file_path = plan_out_dir + "/" + device_name + "/plan.cfg"

            plan_file = open(cfg_file_path, 'w')
            plan_file.write(host_name + "\n")
            plan_file.write(gpu_name + "\n")
            plan_file.write(host_port + "\n")
            plan_file.write(str(len(device_value)) + "\n")

            for tensor_key, tensor_value in device_value.items():

                plan_file.write(tensor_key + "\n")
                plan_file.write(tensor_value["name"] + "\n")
                plan_file.write(tensor_value["type"] + "\n")

                plan_file.write(str(len(tensor_value["endpoints"])) + "\n")
                for endpoints in tensor_value["endpoints"]:
                    host_name, gpu_name, host_port = split_hostname_gpu(endpoints, mapping_hostname_dict, mapping_port_dict)
                    plan_file.write(host_name + " " + gpu_name + " " + host_port + "\n")

            plan_file.close()

if __name__ == "__main__":

    mapping_hostname_file_path = "./mapping_hostname.cfg"
    mapping_port_file_path = "./mapping_port.cfg"

    plans_file_path = "./execution_plan/run.json"
    plan_out_dir = "./execution_plan"

    plans_distribute(mapping_hostname_file_path, mapping_port_file_path, plans_file_path, plan_out_dir)

    print("Distribute Done")

# cd path/to/plan
# python plans_distribute.py
