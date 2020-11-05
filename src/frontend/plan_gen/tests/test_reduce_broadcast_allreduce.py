import json
import os
from frontend.plan_gen.plan.reduce_broadcast_allreduce_plan import \
     ReduceBroadcastAllreducePlan


def test_reduce_broadcast_allreduce_plan():

    RBPlan = ReduceBroadcastAllreducePlan(plan_name='ReduceBroadcast')
    # Test get_plan_name() function
    assert(RBPlan.get_plan_name() == 'ReduceBroadcast')
    # Test get_plan_type() function
    assert(RBPlan.get_plan_type() == 'Allreduce')
    # Test get_plan_info() function
    assert(RBPlan.get_plan_info() == ('Allreduce', 'ReduceBroadcast'))

    # Test None input
    RBPlan.reset_node_list(None)
    assert(RBPlan.generate_plan() is None)

    # load input node_list
    path_input = os.path.join(os.path.dirname(__file__),
                              "data/test_input_nodes.json")
    nodes = json.load(open(path_input, 'r'))

    # Test generate_plan() function
    RBPlan.reset_node_list(nodes)
    output_plan = RBPlan.generate_plan()
    path_output = "data/test_reduce_broadcast_allreduce_output_ref.json"
    path_output = os.path.join(os.path.dirname(__file__),
                               path_output)
    output_ref = json.load(open(path_output, 'r'))
    assert(output_plan.to_json() == output_ref)
