import json
import os
import pytest
from superscaler.plan_gen.plan.resources.resource_pool import ResourcePool
from superscaler.plan_gen.plan.plan_mapper import GPURoundRobinMapper
from superscaler.plan_gen.plan.plan_pool import PlanPool
from superscaler.plan_gen.plan.plan import Plan
from superscaler.plan_gen.plan.plan_manager import PlanManager


def test_plan_manager():

    def Init_PlanMapper():
        # Init resouce_pool
        resource_yaml_path = os.path.join(
            os.path.dirname(__file__), 'data', 'resource_pool.yaml')
        rp = ResourcePool()
        rp.init_from_yaml(resource_yaml_path)

        # Init PlanMapper by resouce_pool
        mapper = GPURoundRobinMapper(rp)
        return mapper

    def Init_PlanPool():
        # Init PlanPool by resouce_pool
        pool = PlanPool()
        plan_test = Plan(plan_type='Default', plan_name='Default_plan')
        pool.add_plan(plan_test)
        return pool

    # Init PlanMapper and PlanPool
    mapper = Init_PlanMapper()
    pool = Init_PlanPool()

    # Wrong init test
    with pytest.raises(Exception):
        planmanager = PlanManager(None, None)

    # Init PlanManager by PlanPool and PlanMapper
    planmanager = PlanManager(pool, mapper)

    # get input node_list
    path = "data/DataParallelismPlan2GPUsIn2Hosts"
    plan_path = os.path.join(
        os.path.dirname(__file__), os.path.join(path, "Nodes.json"))
    plans = json.load(open(plan_path, "r"))

    # Test wrong input of get_execution_plan function
    # wrong node_list
    None_output = planmanager.get_execution_plan(node_list=None,
                                                 plan_type='Default',
                                                 plan_name='Default_plan')
    assert(None_output is None)

    # wrong plan_type
    None_output = planmanager.get_execution_plan(node_list=plans,
                                                 plan_type='wrong_type',
                                                 plan_name='Default_plan')
    assert(None_output is None)

    # wrong plan_name
    None_output = planmanager.get_execution_plan(node_list=plans,
                                                 plan_type='Default',
                                                 plan_name='wrong_name')
    assert(None_output is None)

    # Get execution plan and compare
    plan_output = planmanager.get_execution_plan(plans,
                                                 'Default',
                                                 'Default_plan')
    output_path = os.path.join(
        os.path.dirname(__file__), os.path.join(path, "Default.json"))
    plan_output_ref = json.load(open(output_path, "r"))
    assert(plan_output.to_json() == plan_output_ref)
