import json
import os
from plan.reduce_broadcast_allreduce_plan import ReduceBroadcastAllreducePlan


def test_reduce_broadcast_allreduce_plan():

    RBPlan = ReduceBroadcastAllreducePlan(plan_name='ReduceBroadcast')
    # Test get_plan_name() function
    assert(RBPlan.get_plan_name() == 'ReduceBroadcast')
    # Test get_plan_type() function
    assert(RBPlan.get_plan_type() == 'Allreduce')
    # Test get_plan_info() function
    assert(RBPlan.get_plan_info() == ('Allreduce', 'ReduceBroadcast'))

    # Test None input
    RBPlan.reset_plan(None)
    output_plan = RBPlan.generate_plan()
    assert(output_plan is None)

    # load input node_list
    path_input = os.path.join(os.path.dirname(__file__),
                              "data/test_input_nodes.json")
    nodes = json.load(open(path_input, 'r'))

    node = {
            'device': 'device_0',
            'name': 'test',
            'op': 'Allreduce',
            'output_shapes': [[1, 100]],
            'tensor_name': 'test',
            'tensor_type': 1,
            'input': []
            }

    # Test find_all_allreduce_nodes
    assert(RBPlan.find_all_allreduce_nodes(nodes) == nodes)

    # Test find_endpoints
    assert(RBPlan.find_endpoints(node, nodes) == nodes)

    # Test generate_plan() function
    RBPlan.reset_plan(nodes)
    output_plan = RBPlan.generate_plan()
    path_output = "data/test_reduce_broadcast_allreduce_output_ref.json"
    path_output = os.path.join(os.path.dirname(__file__),
                               path_output)
    output_ref = json.load(open(path_output, 'r'))
    assert(output_plan == output_ref)
