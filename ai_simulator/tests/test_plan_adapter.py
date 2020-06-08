import json
import os

from adapter.plan_adapter import PlanAdapter


def test_plan_adapter():

    # Init adapter class
    adapter = PlanAdapter()

    # Set input plan
    path_input = os.path.join(
        os.path.dirname(__file__),
        "test_plan_gen_adapter/test_plan_adapter_plan.json")
    plan = json.load(open(path_input, 'r'))
    assert adapter.set_plan(plan) is True

    # Test plan_adapter function
    adapted_plan = adapter.get_plan()
    path_output = os.path.join(
        os.path.dirname(__file__),
        "test_plan_gen_adapter/test_plan_adapter_plan_output.json")
    adapted_plan_ref = json.load(open(path_output, 'r'))
    assert(adapted_plan == adapted_plan_ref)

    # Test wrong input
    assert adapter.set_plan(None) is False
    assert adapter.get_plan() is None
    # No 'device' in node
    wrong_input_node = {
        "name": "test_Send_0",
        "op": "Send",
        "output_shapes": [[1, 100]],
        "tensor_name": "test",
        "tensor_type": 1,
        "offset": 0,
        "size": 50,
        "reduction": "",
        "target": "device_1",
        "related_op": "test_Recv_1",
        "parent": "test",
        "input": []}
    assert adapter.set_plan([wrong_input_node]) is False
    assert adapter.get_plan() is None
    # Wrong 'device' type (should be str)
    wrong_input_node = {
        'device': 1024,
        "name": "test_Send_0",
        "op": "Send",
        "output_shapes": [[1, 100]],
        "tensor_name": "test",
        "tensor_type": 1,
        "offset": 0,
        "size": 50,
        "reduction": "",
        "target": "device_1",
        "related_op": "test_Recv_1",
        "parent": "test",
        "input": []}
    assert adapter.set_plan([wrong_input_node]) is False
    assert adapter.get_plan() is None
    # No (test_Send_0, device_1) in node list
    wrong_input_node = {
        "device": "device_0",
        "name": "test_Recv_1",
        "op": "Recv",
        "output_shapes": [[1, 100]],
        "tensor_name": "test",
        "tensor_type": 1,
        "offset": 50,
        "size": 50,
        "reduction": "sum",
        "target": "device_1",
        "related_op": "test_Send_0",
        "parent": "test",
        "input": ["test_Send_0"]
    }
    assert adapter.set_plan([wrong_input_node]) is False
    assert adapter.get_plan() is None
    # 'parent' attr is not the same
    wrong_input_nodes = [
        {
            "device": "device_0",
            "name": "test_Send_0",
            "op": "Send",
            "output_shapes": [[1, 100]],
            "tensor_name": "test",
            "tensor_type": 1,
            "offset": 0,
            "size": 50,
            "reduction": "",
            "target": "device_1",
            "related_op": "test_Recv_1",
            "parent": "NOT_SAME_PARENT",
            "input": []
        },
        {
            "device": "device_1",
            "name": "test_Recv_1",
            "op": "Recv",
            "output_shapes": [[1, 100]],
            "tensor_name": "test",
            "tensor_type": 1,
            "offset": 0,
            "size": 50,
            "reduction": "sum",
            "target": "device_0",
            "related_op": "test_Send_0",
            "parent": "test",
            "input": ["test_Send_0"]
        }
    ]
    assert adapter.set_plan(wrong_input_nodes) is False
    assert adapter.get_plan() is None
    # (related_op, target) not in node list
    wrong_input_nodes = [
        {
            "device": "device_0",
            "name": "test_Send_0",
            "op": "Send",
            "output_shapes": [[1, 100]],
            "tensor_name": "test",
            "tensor_type": 1,
            "offset": 0,
            "size": 50,
            "reduction": "",
            "target": "device_0",
            "related_op": "test_Recv_1",
            "parent": "test",
            "input": []
        },
        {
            "device": "device_1",
            "name": "test_Recv_1",
            "op": "Recv",
            "output_shapes": [[1, 100]],
            "tensor_name": "test",
            "tensor_type": 1,
            "offset": 0,
            "size": 50,
            "reduction": "sum",
            "target": "device_0",
            "related_op": "test_Send_0",
            "parent": "test",
            "input": ["test_Send_0"]
        }
    ]
    assert adapter.set_plan(wrong_input_nodes) is False
    assert adapter.get_plan() is None
