import pytest
from simulator.node import NodeMetadata, Node, NodeException, NodeStatus
from simulator.tensor import Tensor
from simulator.fifo_device import FIFODevice


def test_node_module():
    # Initialize output_tensors
    o_tensors = [Tensor("DT_INT32", 5), Tensor("DT_INT16", 6)]
    # Test node initialization failure
    # The dependency_ids has duplicated elements
    node_metadata = NodeMetadata(
        index=0, op="add_op", name="add",
        device_name="GPU:0", output_tensors=o_tensors,
        execution_time=10, input_ids=[], dependency_ids=[0, 1, 1],
        successor_ids=[4, 5])
    device = FIFODevice("GPU:0")
    with pytest.raises(NodeException):
        Node(node_metadata, device)
    # The conflict between input_ids and dependency_ids in metadata
    node_metadata = NodeMetadata(
        index=0, op="add_op", name="add",
        device_name="GPU:0", output_tensors=o_tensors,
        execution_time=10, input_ids=[0, 1], dependency_ids=[1],
        successor_ids=[4, 5])
    device = FIFODevice("GPU:0")
    with pytest.raises(NodeException):
        Node(node_metadata, device)
    # The mismatch between metadata and device
    node_metadata = NodeMetadata(
        index=0, op="add_op", name="add",
        device_name="GPU:0", output_tensors=o_tensors,
        execution_time=10, input_ids=[], dependency_ids=[],
        successor_ids=[4, 5])
    device = FIFODevice("GPU:1")
    with pytest.raises(NodeException):
        Node(node_metadata, device)

    testnodes = {}
    # Generate independent node
    node_metadata = NodeMetadata(
        index=3, op="add_op", name="add_2",
        device_name="GPU:0", output_tensors=o_tensors,
        execution_time=10, input_ids=[], dependency_ids=[],
        successor_ids=[4, 5])
    node = Node(node_metadata, FIFODevice("GPU:0"))
    testnodes['independency'] = node

    # Generate dependent node
    node_metadata = NodeMetadata(
        index=3, op="add_op", name="add_1",
        device_name="GPU:0", output_tensors=o_tensors,
        execution_time=10, input_ids=[0, 1], dependency_ids=[2],
        successor_ids=[4, 5])
    node = Node(node_metadata, FIFODevice("GPU:0"))
    testnodes['dependency'] = node

    # Test independent node
    test_node = testnodes['independency']
    # Test is_ready()
    assert test_node.is_ready()
    # Test get_remain_dependency_cnt()
    assert test_node.get_remain_dependency_cnt() == 0
    # Test decrease_remain_dependency_cnt()
    with pytest.raises(NodeException):
        test_node.decrease_remain_dependency_cnt(1)
    # Test execute()
    test_node.execute(time_now=20)
    assert test_node.get_status() == NodeStatus.executing
    # Test reset()
    test_node.reset()
    assert test_node.get_status() == NodeStatus.waiting
    assert test_node.get_remain_dependency_cnt() == 0
    # Test finish()
    test_node.execute(time_now=30)
    test_node.finish()
    assert test_node.get_status() == NodeStatus.done

    # Test dependent node
    test_node = testnodes['dependency']
    # Test is_ready()
    assert not test_node.is_ready()
    # Test get_index()
    assert test_node.get_index() == 3
    # Test execute() when raising an exception
    with pytest.raises(NodeException):
        test_node.execute(time_now=5)
    # Test finish() when raising an exception
    with pytest.raises(NodeException):
        test_node.finish()
    # Test decrease_remain_dependency_cnt()
    test_node.decrease_remain_dependency_cnt(3)
    assert test_node.is_ready()
    # Test execute()
    test_node.execute(time_now=5)
    assert test_node.get_status() == NodeStatus.executing
    # Test finish()
    test_node.finish()
    assert test_node.get_status() == NodeStatus.done
    # Test reset()
    test_node.reset()
    assert test_node.get_status() == NodeStatus.waiting
    assert test_node.get_remain_dependency_cnt() == 3
