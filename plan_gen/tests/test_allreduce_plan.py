import os
import json
from plan.allreduce_plan import AllreducePlan


def test_allreduce_plan():

    Allreduce_plan = AllreducePlan(plan_name='Allreduce_Plan')
    # Test get_plan_name() function
    assert(Allreduce_plan.get_plan_name() == 'Allreduce_Plan')
    # Test get_plan_type() function
    assert(Allreduce_plan.get_plan_type() == 'Allreduce')
    # Test get_plan_type() function
    assert(Allreduce_plan.get_plan_info() == ('Allreduce', 'Allreduce_Plan'))

    # Test None input
    Allreduce_plan.reset_plan(None)
    assert(Allreduce_plan.generate_plan() is None)

    # load input node_list and reset plan
    nodes = json.load(open(os.path.join(os.path.dirname(__file__),
                           "data/test_input_nodes.json"), 'r'))
    Allreduce_plan.reset_plan(nodes)

    # Test generate_plan() function
    assert(Allreduce_plan.generate_plan() == nodes)

    # Test find_all_allreduce_nodes() function
    assert(Allreduce_plan.find_all_allreduce_nodes(nodes) == nodes)

    # Test find_endpoints() function
    node = nodes[0]
    assert(Allreduce_plan.find_endpoints(node, nodes) == nodes)
