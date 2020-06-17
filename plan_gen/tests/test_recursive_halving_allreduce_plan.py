import json
import os
from plan.recursive_halving_plan import RecursiveHalvingAllreducePlan


def test_recursive_halving_plan():

    RH = RecursiveHalvingAllreducePlan(plan_name='recursive_halving')
    # Test get_plan_name() function
    assert(RH.get_plan_name() == 'recursive_halving')
    # Test get_plan_type() function
    assert(RH.get_plan_type() == 'Allreduce')
    # Test get_plan_info() function
    assert(RH.get_plan_info() == ('Allreduce', 'recursive_halving'))

    # Load input node_list
    path_input = os.path.join(os.path.dirname(__file__),
                              "data/test_input_nodes.json")
    nodes = json.load(open(path_input, 'r'))

    # Test generate_plan() function
    RH.reset_plan(nodes)
    output_plan = RH.generate_plan()
    path_output = "data/test_recursive_halving_allreduce_output_ref.json"
    path_output = os.path.join(os.path.dirname(__file__),
                               path_output)
    output_ref = json.load(open(path_output, 'r'))
    assert(output_plan == output_ref)
