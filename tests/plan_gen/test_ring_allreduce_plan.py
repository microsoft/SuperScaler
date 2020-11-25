# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json
import os
from superscaler.plan_gen.plan import ring_allreduce_plan


def test_ring_allreduce_plan():

    ring = ring_allreduce_plan.RingAllreducePlan(plan_name='ring')
    # Test get_plan_name() function
    assert(ring.get_plan_name() == 'ring')
    # Test get_plan_type() function
    assert(ring.get_plan_type() == 'Allreduce')
    # Test get_plan_info() function
    assert(ring.get_plan_info() == ('Allreduce', 'ring'))

    # Test None input
    ring.reset_node_list(None)
    assert(ring.generate_plan() is None)

    # Load input node_list
    path_input = os.path.join(os.path.dirname(__file__),
                              "data/test_input_nodes.json")
    nodes = json.load(open(path_input, 'r'))

    # Test generate_plan() function
    ring.reset_node_list(nodes)
    output_plan = ring.generate_plan()
    path_output = os.path.join(os.path.dirname(__file__),
                               "data/test_ring_allreduce_output_ref.json")
    output_ref = json.load(open(path_output, 'r'))
    assert(output_plan.to_json() == output_ref)
