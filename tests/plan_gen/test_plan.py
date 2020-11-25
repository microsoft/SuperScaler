# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json
import os
from superscaler.plan_gen.plan import plan


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
    Default_plan.reset_node_list(None)
    assert(Default_plan.generate_plan() is None)

    # load input node_list and reset
    nodes = json.load(open(os.path.join(os.path.dirname(__file__),
                           "data/test_input_nodes.json"), 'r'))
    Default_plan.reset_node_list(nodes)

    # Test generate_plan() function
    node_list = Default_plan.generate_plan()
    assert(node_list.to_json() == nodes)
