from superscaler.ai_simulator.simulator.device import Device
from superscaler.ai_simulator.simulator.fifo_device import FIFODevice
from superscaler.ai_simulator.simulator.node import Node, NodeMetadata


def test_device_module():
    # Mock the input data
    test_device = Device("GPU:0")
    assert test_device.is_idle() is True

    # Test enqueue_node()
    test_FIFO_device = FIFODevice("GPU:0")
    assert test_device.is_idle() is True
    test_enqueue_node_metadata = NodeMetadata(
        index=0,
        op='Add',
        name='test_node_1',
        device_name='GPU:0',
        execution_time=2.0,
        input_ids=[],
        dependency_ids=[],
        successor_ids=[]
    )
    test_enqueue_node = Node(test_enqueue_node_metadata, test_device)
    test_FIFO_device.enqueue_node(node=test_enqueue_node, time_now=1.0)
    assert test_FIFO_device.is_idle() is False
    assert test_FIFO_device.name() == "GPU:0"

    # Test head_node()
    actual_next_node, actual_next_finish_time \
        = test_FIFO_device.get_next_node()
    assert actual_next_node == test_enqueue_node
    assert actual_next_finish_time == 3.0

    # Test dequeue_node()
    test_FIFO_device.dequeue_node()
    assert test_FIFO_device.is_idle() is True
