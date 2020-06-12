import json
import os
from plan import plan


def test_plan():

    Default_plan = plan.Plan(plan_type="Default", plan_name="Default_Plan")
    # Test get_plan_name() function
    assert(Default_plan.get_plan_name() == 'Default_Plan')
    # Test get_plan_type() function
    assert(Default_plan.get_plan_type() == 'Default')
    # Test get_plan_info() function
    assert(Default_plan.get_plan_info() == ('Default', 'Default_Plan'))
    # Test generate_plan() function
    assert(hasattr(Default_plan, 'generate_plan'))

    # Test None input
    Default_plan.reset_plan(None)
    assert(Default_plan.generate_plan() is None)

    # load input node_list and reset plan
    nodes = json.load(open(os.path.join(os.path.dirname(__file__),
                           "data/test_input_nodes.json"), 'r'))
    Default_plan.reset_plan(nodes)

    # Test generate_plan() function
    assert(Default_plan.generate_plan() == nodes)

    node = {
            'device': 'device_2',
            'name': 'test',
            'op': 'Allreduce',
            'output_shapes': [[1, 100]],
            'tensor_name': 'test',
            'tensor_type': 1,
            'input': []
           }
    # Test _add_node() function
    Default_plan._add_node(node)
    assert(Default_plan._get_node_index(node) == 2)

    # Test _get_node() function
    assert(Default_plan._get_node(2) == node)

    # Test _add_node() function
    Default_plan._remove_node(node)
    assert(Default_plan._get_node_index(node) is None)


def test_allreduce_plan():

    Allreduce_plan = plan.AllreducePlan(plan_name='Allreduce_Plan')
    # Test get_plan_name() function
    assert(Allreduce_plan.get_plan_name() == 'Allreduce_Plan')
    # Test get_plan_type() function
    assert(Allreduce_plan.get_plan_type() == 'Allreduce')
    # Test get_plan_type() function
    assert(Allreduce_plan.get_plan_info() == ('Allreduce', 'Allreduce_Plan'))
