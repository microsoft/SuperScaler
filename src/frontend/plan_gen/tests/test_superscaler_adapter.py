import json
import os
from plan.adapter.superscaler_adapter import SuperScalerAdapter


def test_plan_adapter_for_superscaler():
    # load input plan
    path_input = os.path.join(
        os.path.dirname(__file__), "data/ring_simple.json")
    plan_example = json.load(open(path_input, 'r'))

    # Init adapter
    adapter = SuperScalerAdapter()
    adapter.set_plan(plan_example)

    # Adapt plan and check dumped plan
    output_plan = adapter.adapt_plan()
    output_path = os.path.join(os.path.dirname(__file__), "data")
    adapter.dump_plan(output_path, 'superscaler')
    output_ref = []
    for i in range(len(output_plan)):
        path_output = os.path.join(os.path.dirname(__file__),
                                   "data/superscaler" + str(i) + ".json")
        output_ref.append(json.load(open(path_output, 'r')))
    assert(output_plan == output_ref)
