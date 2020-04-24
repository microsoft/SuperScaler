import pytest

from simulator.network_device import NetworkSwitch
from simulator.node import Node, NodeMetadata
from simulator.tensor import Tensor


def test_network_switch_error_handling():
    # Init tensor unit
    tensor_unit = Tensor('int32', 1)
    # Parameter invalid: duplicated dest_device_name
    with pytest.raises(ValueError):
        NetworkSwitch(
            "/switch/switch0",
            [('/server/hostname1/GPU/0', '1bps')]
            + [('/server/hostname1/GPU/0', '1bps')])
    correct_switch = NetworkSwitch(
            "/switch/switch0",
            [('/server/hostname1/GPU/0', '1bps')]
            + [('/server/hostname1/GPU/1', '1bps')])
    nodes_metadata = []
    nodes_metadata.append(
        NodeMetadata(
            # op example: ":send:outbound_dest:...."
            # e.g. ":send:/server/hostname1/GPU/0:"
            # Wrong outbound_dest
            index=0, op=":send:/server/hostname1/GPU/2312312:", name="send",
            device_name="/switch/switch0",
            output_tensors=[tensor_unit for _ in range(10)],
            execution_time=5233, input_ids=[],
            dependency_ids=[], successor_ids=[]))
    nodes_list = []
    for metadata in nodes_metadata:
        nodes_list.append(Node(metadata, correct_switch))
    # Wrong op outbound_dest
    with pytest.raises(ValueError):
        correct_switch.enqueue_node(nodes_list[0], 0)


def test_network_switch_functionality():
    # Init a 4 port switch
    # legend: port_number(port_normalized_rate)
    #       ---------------
    #    ---|0(1)     2(2)|---
    #       |             |
    #    ---|1(1)     3(2)|---
    #       ---------------
    # Index     outbound_port   normalized len
    # 0         0               10
    # 1         0               20
    # 2         1               10
    # 3         2               10
    # 4         3               20
    correct_departure_index = [3, 2, 4, 0, 1]
    correct_finish_time = [5, 10, 10, 20, 30]
    # Init tensor unit
    tensor_unit = Tensor('int32', 1)
    tensor_bit_size = tensor_unit.get_bytes_size() * 8
    pcie_switch = NetworkSwitch(
        "/switch/switch0",
        [('/server/hostname1/GPU/0', str(tensor_bit_size)+'bps')]
        + [('/server/hostname1/GPU/1', str(tensor_bit_size)+'bps')]
        + [('/server/hostname1/GPU/2', str(tensor_bit_size*2)+'bps')]
        + [('/server/hostname1/GPU/3', str(tensor_bit_size*2)+'bps')])
    nodes_metadata = []
    nodes_metadata.append(
        NodeMetadata(
            index=0, op=":send:/server/hostname1/GPU/0:", name="send",
            device_name="/switch/switch0",
            output_tensors=[tensor_unit for _ in range(10)],
            execution_time=5233, input_ids=[],
            dependency_ids=[], successor_ids=[]))
    nodes_metadata.append(
        NodeMetadata(
            index=1, op=":send:/server/hostname1/GPU/0:", name="send",
            device_name="/switch/switch0",
            output_tensors=[tensor_unit for _ in range(20)],
            execution_time=5233, input_ids=[],
            dependency_ids=[], successor_ids=[]))
    nodes_metadata.append(
        NodeMetadata(
            index=2, op=":send:/server/hostname1/GPU/1:", name="send",
            device_name="/switch/switch0",
            output_tensors=[tensor_unit for _ in range(10)],
            execution_time=5233, input_ids=[],
            dependency_ids=[], successor_ids=[]))
    nodes_metadata.append(
        NodeMetadata(
            index=3, op=":send:/server/hostname1/GPU/2:", name="send",
            device_name="/switch/switch0",
            output_tensors=[tensor_unit for _ in range(10)],
            execution_time=5233, input_ids=[],
            dependency_ids=[], successor_ids=[]))
    nodes_metadata.append(
        NodeMetadata(
            index=4, op=":send:/server/hostname1/GPU/3:", name="send",
            device_name="/switch/switch0",
            output_tensors=[tensor_unit for _ in range(20)],
            execution_time=5233, input_ids=[],
            dependency_ids=[], successor_ids=[]))
    nodes_list = []
    for metadata in nodes_metadata:
        nodes_list.append(Node(metadata, pcie_switch))

    # Test NetworkSwitch
    assert pcie_switch.is_idle()
    pcie_switch.dequeue_node()
    # Enqueue all nodes in time 0
    for node in nodes_list:
        pcie_switch.enqueue_node(node, 0)
    assert not pcie_switch.is_idle()
    sim_results = []
    assert pcie_switch.get_next_node()[1] == 5
    # Dequeue all nodes
    for i in range(5):
        sim_results.append(pcie_switch.get_next_node())
        pcie_switch.dequeue_node()

    assert pcie_switch.is_idle()
    # Test sim results
    for i in range(5):
        node, t = sim_results[i]
        ind = node.get_index()
        assert ind == correct_departure_index[i]
        assert t == correct_finish_time[i]
