# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import random 
from aceso_utils import *

random.seed(2)
args = parse_args()

config_target_visited = {}
config_action_target_partner_visited = {}

NUM_CASES_HIT = []

def reset_hit_resources():
    global NUM_CASES_HIT
    NUM_CASES_HIT = [0, 0, 0]

def get_hit_resources():
    global NUM_CASES_HIT
    return NUM_CASES_HIT

def get_target_stage(config, metric=None, other_info="", adaptive_flag=False):
    global config_target_visited
    if not adaptive_flag and num_visited(config_target_visited, hash_str=config_details(config, get_string=True) + other_info) >= config.num_stages:
        return None

    if metric is None and max(config.memory_list) > args.memory_limit:
        values = config.memory_list
    else:
        values = config.time_list

    target_stage = None
    max_value = 0
    for i in range(config.num_stages):
        if values[i] > max_value and (adaptive_flag or not is_visited(config_target_visited, hash_str=config_details(config, get_string=True) + other_info, target=i)):
            target_stage = i
            max_value = values[i]

    if target_stage is not None:
        mark_visited(config_target_visited, hash_str=config_details(config, get_string=True) + other_info, target=target_stage)
    return target_stage

def get_partner_stage(config, action, partner_action, target_stage, metric="time_with_efficiency", other_info=""):
    """
    Currently, only action_mig_workloads() calls this function to get partner stage (which accepts the moved-in workloads).
    By default, the metric is "time_with_efficiency" when there is no memory pressure.
    """
    global config_action_target_partner_visited
    hash_str = config_details(config, get_string=True) + action + partner_action + str(target_stage) + str(other_info)
    if num_visited(config_action_target_partner_visited, hash_str) >= (config.num_stages - 1):
        return None

    partner_stage = None
    if metric == "time_with_efficiency":
        values = config.efficient_time_list
        max_value = 0
        for i in range(len(values)):
            if values[i] > max_value \
            and not is_visited(config_action_target_partner_visited, hash_str=hash_str, target=i) \
            and i != target_stage:
                partner_stage = i 
                max_value = values[i]
    elif metric == "memory":
        values = config.memory_list
        min_value = MAX_VALUE
        for i in range(len(values)):
            if values[i] < min_value \
            and not is_visited(config_action_target_partner_visited, hash_str=hash_str, target=i) \
            and i != target_stage:
                partner_stage = i 
                min_value = values[i]
    else:
        raise RuntimeError(f"metric {metric} for get_partner_stage() is not supported.")

    if partner_stage is not None:
        mark_visited(config_action_target_partner_visited, hash_str=hash_str, target=partner_stage)
    
    return partner_stage

def reset_visited_partners():
    global config_action_target_partner_visited
    config_action_target_partner_visited = {}

def get_actions_by_filters(action_resource_table, memory_choice=["+","-","0"], efficiency_choice=["+","-","0"], time_choice=["+","-","0"], comm_choice=["+","-","0"], gpu_choice=["+","-","0"], workloads_choice=["+","-","0"], exclude=[]):
    prims = []
    for prim in action_resource_table:
        if prim not in exclude and prim.memory in memory_choice and \
            prim.time in time_choice and prim.efficiency in efficiency_choice and \
            prim.comm in comm_choice and prim.num_devices in gpu_choice and \
            prim.workloads in workloads_choice:
                prims.append(prim)

    return prims    

def get_actions_with_policy(config, bottleneck, action_resource_table, adaptive_flag=False):
    """
    get_actions v5:
    Order actions according to the RATIO of ideal_time, eff_loss_time, recompute_time.
    [The ratio is calculated using the per-gpu value.]
    Consider memory at the first place.
    To decrease ideal_time: time_choice=["-"]
        But also consider comm and memory with additional comm_choice and memory_choice.
    To decrease eff_loss_time: comm_choice=["-"]
        But also consider ideal_time and memory with additional time_choice and memory_choice.
    To decrease recompute_time: memory_choice=["-"]
        But also consider ideal_time and comm with additional time_choice and comm_choice.
    """
    global NUM_CASES_HIT

    actions, turn_back_actions = [], []
    stage_type = ""

    if max(config.memory_list) > args.memory_limit:
        _actions = get_actions_by_filters(action_resource_table, memory_choice=["-"])
        actions.append(_actions)    
        stage_type = "OOM"
    else:
        comp_time = config.breakdown_ideal_time_per_gpu[bottleneck]
        avg_comp_time = sum(config.breakdown_ideal_time_per_gpu)/config.num_stages
        if comp_time > 0:
            comp_time_ratio = config.breakdown_ideal_time_per_gpu[bottleneck] / sum(config.breakdown_ideal_time_per_gpu)
        else:
            comp_time_ratio = 0
        eff_loss_time = config.breakdown_eff_loss_time_per_gpu[bottleneck]
        avg_eff_loss_time = sum(config.breakdown_eff_loss_time_per_gpu)/config.num_stages
        if eff_loss_time > 0:
            eff_loss_time_ratio = config.breakdown_eff_loss_time_per_gpu[bottleneck] / sum(config.breakdown_eff_loss_time_per_gpu)
        else:
            eff_loss_time_ratio = 0
        recomp_time = config.breakdown_recomp_time_per_gpu[bottleneck]
        avg_recomp_time = sum(config.breakdown_recomp_time_per_gpu)/config.num_stages
        if recomp_time > 0:
            recomp_time_ratio = config.breakdown_recomp_time_per_gpu[bottleneck] / sum(config.breakdown_recomp_time_per_gpu)
        else:
            recomp_time_ratio = 0
        tmp_list = [comp_time_ratio, eff_loss_time_ratio, recomp_time_ratio]
        index = 0
        visited_actions = []
        while index < 3:
            ## - comp_time
            if comp_time_ratio == max(tmp_list):
                if eff_loss_time > avg_eff_loss_time:
                    comm_choice = ["-", "0"]
                else:
                    comm_choice = ["+", "-", "0"]
                if recomp_time > 0: 
                    memory_choice = ["-", "0"]
                else:
                    memory_choice = ["+", "-", "0"]
                _actions = get_actions_by_filters(action_resource_table, time_choice=["-"], comm_choice=comm_choice, memory_choice=memory_choice, exclude=visited_actions)
                actions.append(_actions)
                visited_actions += _actions

                _actions = get_actions_by_filters(action_resource_table, time_choice=["-"], exclude=visited_actions)
                actions.append(_actions)
                visited_actions += _actions

                tmp_list.remove(comp_time_ratio)
                if index == 0:
                    NUM_CASES_HIT[0] += 1
            ## - comm_time
            elif eff_loss_time_ratio == max(tmp_list):
                if comp_time > avg_comp_time:
                    time_choice = ["-", "0"]
                else:
                    time_choice = ["+", "-", "0"]
                if recomp_time > 0: 
                    memory_choice = ["-", "0"]
                else:
                    memory_choice = ["+", "-", "0"]
                _actions = get_actions_by_filters(action_resource_table, time_choice=time_choice, comm_choice=["-"], memory_choice=memory_choice, exclude=visited_actions)
                actions.append(_actions)
                visited_actions += _actions

                _actions = get_actions_by_filters(action_resource_table, comm_choice=["-"], exclude=visited_actions)
                actions.append(_actions)
                visited_actions += _actions                

                tmp_list.remove(eff_loss_time_ratio)        
                if index == 0:
                    NUM_CASES_HIT[1] += 1            
            ## - memory
            elif recomp_time_ratio == max(tmp_list):
                if comp_time > avg_comp_time:
                    time_choice = ["-", "0"]
                else:
                    time_choice = ["+", "-", "0"]
                if eff_loss_time > avg_eff_loss_time:
                    comm_choice = ["-", "0"]
                else:
                    comm_choice = ["+", "-", "0"]        
                _actions = get_actions_by_filters(action_resource_table, time_choice=time_choice, comm_choice=comm_choice, memory_choice=["-"], exclude=visited_actions)
                actions.append(_actions)
                visited_actions += _actions

                _actions = get_actions_by_filters(action_resource_table, memory_choice=["-"], exclude=visited_actions)
                actions.append(_actions)
                visited_actions += _actions      

                tmp_list.remove(recomp_time_ratio)   
                if index == 0:
                    NUM_CASES_HIT[2] += 1            
            index += 1
        _actions = get_actions_by_filters(action_resource_table, exclude=visited_actions)
        actions.append(_actions)

    return actions, turn_back_actions, stage_type

def get_actions_random_order(config, bottleneck, action_resource_table, adaptive_flag=False):
    actions, turn_back_actions = [], []
    stage_type = ""
    all_actions = ["move_out", "inc_dp", "dec_dp", "inc_tp", "dec_tp", "inc_mbs", "dec_mbs"]

    random_order = random.sample(range(7), 7)

    for index in random_order:
        actions.append([all_actions[index]])

    return actions, turn_back_actions, stage_type

def get_actions(config, bottleneck, action_resource_table, adaptive_flag):
    if args.random_order_actions:
        return get_actions_random_order(config, bottleneck, action_resource_table, adaptive_flag)
    else:
        return get_actions_with_policy(config, bottleneck, action_resource_table, adaptive_flag)