import pytest
from simulator.tensor import *
from simulator.fifo_device import FIFODevice
from simulator.fairsharing_device import *
from simulator.node import Node, NodeMetadata


def test_flow():
    tensor_unit = Tensor('int32', 1)
    tensor_size = 5
    # Initialize output_tensors
    o_tensors = [Tensor('int32', 1) for _ in range(tensor_size)]
    # Initialize node metadata
    nodes_metadata = []
    nodes_metadata.append(NodeMetadata(index=0, op="add_op", name="add",
                                       device_name="GPU:0",
                                       output_tensors=o_tensors,
                                       execution_time=10, input_ids=[],
                                       dependency_ids=[], successor_ids=[]))
    device = FIFODevice("GPU:0")
    node = Node(nodes_metadata[-1], device)
    # Init flow with same time_now and capacity
    flow_0 = Flow(node, 10, tensor_unit.get_bytes_size()*8)

    nodes_metadata.append(
        NodeMetadata(index=1, op="add_op", name="add",
                     device_name="GPU:0",
                     output_tensors=[
                      Tensor('int32', 1) for _ in range(tensor_size+1)],
                     execution_time=10, input_ids=[],
                     dependency_ids=[], successor_ids=[]))
    node = Node(nodes_metadata[-1], device)
    flow_1 = Flow(node, 10, tensor_unit.get_bytes_size()*8)

    # Test Flow comparision operation
    assert flow_0 < flow_1

    # Test Flow methods
    assert flow_0.get_estimated_finish_time() == 10 + tensor_size
    flow_0.change_available_bandwidth(
        11,
        tensor_unit.get_bytes_size()*8*2)
    assert flow_0.get_estimated_finish_time() == 10 + 1 + (tensor_size-1)/2


def test_fairsharing_device_simple():
    tensor_unit = Tensor('int32', 1)
    device = FairSharingDevice("network: 0",
                               str(tensor_unit.get_bytes_size()*8)+'bps')
    nodes_metadata = []
    nodes_metadata.append(
        NodeMetadata(index=0,
                     op="send",
                     name="send", device_name="network: 0",
                     output_tensors=[
                         Tensor('int32', 1) for _ in range(4)],
                     execution_time=1, input_ids=[],
                     dependency_ids=[], successor_ids=[]))
    node = Node(nodes_metadata[-1], device)
    device.enqueue_node(node, 0)
    next_node, next_time = device.get_next_node()
    assert next_node.get_index() == 0
    assert next_time == 4

    nodes_metadata.append(
        NodeMetadata(index=1,
                     op="send",
                     name="send", device_name="network: 0",
                     output_tensors=[
                         Tensor('int32', 1) for _ in range(1)],
                     execution_time=1, input_ids=[],
                     dependency_ids=[], successor_ids=[]))
    node = Node(nodes_metadata[-1], device)
    device.enqueue_node(node, 0)
    next_node, next_time = device.get_next_node()
    assert next_node.get_index() == 1
    assert next_time == 2

    device.dequeue_node()
    next_node, next_time = device.get_next_node()
    assert next_node.get_index() == 0
    assert next_time == 5

    device.dequeue_node()
    assert device.is_idle()


def test_fairsharing_device_complex():
    tensor_unit = Tensor('int32', 1)
    # Index     Arrival_time    normalized len
    # 0         0               1
    # 1         0               1
    # 2         1               2
    # 3         1               2
    # 4         6               8
    # 5         7               4
    # 6         8               1
    param_arr = [(0, 1), (0, 1), (1, 2), (1, 2), (6, 8), (7, 4), (8, 1)]
    correct_departure_index = [0, 1, 2, 3, 6, 5, 4]
    correct_finish_time = [3, 3, 6, 6, 11, 16, 19]

    device = FairSharingDevice("network: 0",
                               str(tensor_unit.get_bytes_size()*8)+'bps')

    nodes = []
    nodes_enqueue_time = []
    index_id = 0
    for arrival_time, norm_len in param_arr:
        node_metadata = NodeMetadata(
                         index=index_id,
                         op="send",
                         name="send", device_name="network: 0",
                         output_tensors=[
                          Tensor('int32', 1) for _ in range(norm_len)],
                         execution_time=1, input_ids=[],
                         dependency_ids=[], successor_ids=[])
        index_id += 1
        nodes.append(Node(node_metadata, device))
        nodes_enqueue_time.append(arrival_time)

    node_index = 0
    t_now = nodes_enqueue_time[node_index]
    device.enqueue_node(nodes[node_index], nodes_enqueue_time[node_index])
    node_index += 1

    results = []
    while(node_index < len(nodes)):
        next_node, next_time = device.get_next_node()
        if(nodes_enqueue_time[node_index] <= next_time):
            device.enqueue_node(nodes[node_index],
                                nodes_enqueue_time[node_index])
            node_index += 1
        else:
            device.dequeue_node()
            results.append((next_node.get_index(), next_time))

    while not device.is_idle():
        next_node, next_time = device.get_next_node()
        device.dequeue_node()
        results.append((next_node.get_index(), next_time))

    for i in range(len(results)):
        ind, t = results[i]
        print("index is {}, leave time is {}".format(ind, t))
        assert ind == correct_departure_index[i]
        assert t == correct_finish_time[i]
