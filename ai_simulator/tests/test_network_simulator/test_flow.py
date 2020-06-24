import pytest

from simulator.tensor import Tensor
from simulator.fifo_device import FIFODevice
from simulator.node import Node, NodeMetadata
from simulator.network_simulator.flow import Flow


def test_flow():
    tensor_unit = Tensor('int32', 1)
    tensor_size = 5
    # Init flow with same time_now and capacity
    flow_0 = create_test_flow(0, 'send', '/switch/switch0/', tensor_size, 10)
    flow_0.set_available_bandwidth(tensor_unit.get_bytes_size() * 8, 10)
    flow_1 = create_test_flow(0, 'send', '/switch/switch0/',
                              tensor_size + 1, 10)
    flow_1.set_available_bandwidth(tensor_unit.get_bytes_size() * 8, 10)
    # Test Flow comparision operation
    assert flow_0 < flow_1
    # Test Flow methods
    with pytest.raises(ValueError):
        # Wrong available_bandwidth: negative number
        flow_0.set_available_bandwidth(-100, 10)
    with pytest.raises(ValueError):
        # Wrong time_now: time should not be reversed
        flow_0.set_available_bandwidth(100, -999)
    assert flow_0.get_available_bandwidth() == \
        tensor_unit.get_bytes_size()*8
    assert flow_0.get_estimated_finish_time() == 10 + tensor_size
    flow_0.set_available_bandwidth(tensor_unit.get_bytes_size() * 8 * 2, 11)
    assert flow_0.get_estimated_finish_time() == 10 + 1 + (tensor_size - 1)/2
    assert flow_0.get_available_bandwidth() == \
        tensor_unit.get_bytes_size() * 8 * 2
    flow_0.set_available_bandwidth(0, 14)
    assert flow_0.get_estimated_finish_time() == float('inf')
    # Test float precision tolerant
    assert flow_1.get_estimated_finish_time() == 16.0
    flow_1.set_available_bandwidth(tensor_unit.get_bytes_size() * 8, 16.0)
    assert flow_1.get_estimated_finish_time() == 16.0


def create_test_flow(index, node_name, device_name, tensor_size, time_now):
    # tensor_unit = Tensor('int32', 1)
    # Initialize output_tensors
    o_tensors = [Tensor('int32', 1) for _ in range(tensor_size)]
    # Initialize node metadata
    nodes_metadata = []
    nodes_metadata.append(NodeMetadata(index=index, op="send", name=node_name,
                                       device_name=device_name,
                                       output_tensors=o_tensors,
                                       execution_time=10, input_ids=[],
                                       dependency_ids=[], successor_ids=[]))
    device = FIFODevice(device_name)
    node = Node(nodes_metadata[-1], device)
    return Flow(node, time_now)
