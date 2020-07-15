import json
import os
import copy

from simulator.adapter.plan_adapter import PlanAdapter


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
    # Need to make sure the input/dependency and successor_ids are coherent
    for node in adapted_plan:
        for check_node in adapted_plan:
            if check_node["index"] in node['input_ids'] \
                    or check_node['index'] in node['dependency_ids']:
                assert node['index'] in check_node['successor_ids']
            elif check_node["index"] in node['successor_ids']:
                assert node['index'] in check_node['input_ids'] \
                    or node['index'] in check_node['dependency_ids']

    # Check adapted_plan, excluding the output_tensors
    adapted_plan_without_output_tensors = copy.deepcopy(adapted_plan)
    for node in adapted_plan_without_output_tensors:
        node['output_tensors'] = []
    path_output = os.path.join(
        os.path.dirname(__file__),
        "test_plan_gen_adapter/test_plan_adapter_plan_output.json")
    adapted_plan_without_output_tensors_ref = json.load(open(path_output, 'r'))

    assert(adapted_plan_without_output_tensors ==
           adapted_plan_without_output_tensors_ref)

    # Check output_tensors
    for node in adapted_plan:
        node_tensor_byte_size = 0
        for tensor in node['output_tensors']:
            node_tensor_byte_size += tensor.get_bytes_size()
        if node['op'] == 'Send':
            assert node_tensor_byte_size == 200
        elif node['op'] == 'Recv':
            assert node_tensor_byte_size == 0

    # Test wrong input
    assert adapter.set_plan(None) is False
    assert adapter.get_plan() is None
    # No 'device' in node
    wrong_input_node = {
        "name": "test_Send_0",
        "op": "Send",
        "output_shapes": [[1, 100]],
        "tensor_name": "test",
        "tensor_type": "DT_FLOAT",
        "offset": 0,
        "size": 50,
        "reduction": "",
        "target": "/server/hostname1/GPU/1/",
        "related_op": "test_Recv_1",
        "parent": "test",
        "input": [],
        "route_index": 0,
        "route_type": "PCIE"
    }
    assert adapter.set_plan([wrong_input_node]) is False
    assert adapter.get_plan() is None
    # Wrong 'device' type (should be str)
    wrong_input_node = {
        "device": 1024,
        "name": "test_Send_0",
        "op": "Send",
        "output_shapes": [[1, 100]],
        "tensor_name": "test",
        "tensor_type": "DT_FLOAT",
        "offset": 0,
        "size": 50,
        "reduction": "",
        "target": "/server/hostname1/GPU/1/",
        "related_op": "test_Recv_1",
        "parent": "test",
        "input": [],
        "route_index": 0,
        "route_type": "PCIE"
    }
    assert adapter.set_plan([wrong_input_node]) is False
    assert adapter.get_plan() is None
    # No (test_Send_0, /server/hostname1/GPU/1/) in node list
    wrong_input_node = {
        "device": "/server/hostname1/GPU/0/",
        "name": "test_Recv_1",
        "op": "Recv",
        "output_shapes": [[1, 100]],
        "tensor_name": "test",
        "tensor_type": "DT_FLOAT",
        "offset": 50,
        "size": 50,
        "reduction": "sum",
        "target": "/server/hostname1/GPU/1/",
        "related_op": "test_Send_0",
        "parent": "test",
        "input": ["test_Send_0"],
        "route_index": 0,
        "route_type": "PCIE"
    }
    assert adapter.set_plan([wrong_input_node]) is False
    assert adapter.get_plan() is None
    # 'parent' attr is not the same
    wrong_input_nodes = [
        {
            "device": "/server/hostname1/GPU/0/",
            "name": "test_Send_0",
            "op": "Send",
            "output_shapes": [[1, 100]],
            "tensor_name": "test",
            "tensor_type": "DT_FLOAT",
            "offset": 0,
            "size": 50,
            "reduction": "",
            "target": "/server/hostname1/GPU/1/",
            "related_op": "test_Recv_1",
            "parent": "NOT_SAME_PARENT",
            "input": [],
            "route_index": 0,
            "route_type": "PCIE"
        },
        {
            "device": "/server/hostname1/GPU/1/",
            "name": "test_Recv_1",
            "op": "Recv",
            "output_shapes": [[1, 100]],
            "tensor_name": "test",
            "tensor_type": "DT_FLOAT",
            "offset": 0,
            "size": 50,
            "reduction": "sum",
            "target": "/server/hostname1/GPU/0/",
            "related_op": "test_Send_0",
            "parent": "test",
            "input": ["test_Send_0"],
            "route_index": 0,
            "route_type": "PCIE"
        }
    ]
    assert adapter.set_plan(wrong_input_nodes) is False
    assert adapter.get_plan() is None
    # (related_op, target) not in node list
    wrong_input_nodes = [
        {
            "device": "/server/hostname1/GPU/0/",
            "name": "test_Send_0",
            "op": "Send",
            "output_shapes": [[1, 100]],
            "tensor_name": "test",
            "tensor_type": "DT_FLOAT",
            "offset": 0,
            "size": 50,
            "reduction": "",
            "target": "NO_THIS_TARGET",
            "related_op": "test_Recv_1",
            "parent": "test",
            "input": [],
            "route_index": 0,
            "route_type": "PCIE"
        },
        {
            "device": "/server/hostname1/GPU/1/",
            "name": "test_Recv_1",
            "op": "Recv",
            "output_shapes": [[1, 100]],
            "tensor_name": "test",
            "tensor_type": "DT_FLOAT",
            "offset": 0,
            "size": 50,
            "reduction": "sum",
            "target": "/server/hostname1/GPU/0/",
            "related_op": "test_Send_0",
            "parent": "test",
            "input": ["test_Send_0"],
            "route_index": 0,
            "route_type": "PCIE"
        }
    ]
    assert adapter.set_plan(wrong_input_nodes) is False
    assert adapter.get_plan() is None
