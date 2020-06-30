import pytest

from simulator.tensor import Tensor
from simulator.node import Node, NodeMetadata
from simulator.network_simulator.network_simulator import NetworkSimulator


def test_network_simulator_functionality():
    # Topology:
    # GPU0 --10G-→ PCIeSW --6G-→ GPU2
    # GPU1 --2G--↗
    tensor_unit = Tensor('int32', 1)
    net_sim = NetworkSimulator(
        'network_sim',
        [
            {
                'link_id': 0, 'source_name': '/server/hostname/GPU/0/',
                'dest_name': '/switch/switch0/',
                'capacity': str(tensor_unit.get_bytes_size()*8*10)+'bps'
            },
            {
                'link_id': 1, 'source_name': '/server/hostname/GPU/1/',
                'dest_name': '/switch/switch0/',
                'capacity': str(tensor_unit.get_bytes_size()*8*2)+'bps'
            },
            {
                'link_id': 2, 'source_name': '/switch/switch0/',
                'dest_name': '/server/hostname/GPU/2/',
                'capacity': str(tensor_unit.get_bytes_size()*8*6)+'bps'
            }
        ],
        {
            ('/server/hostname/GPU/0/', '/server/hostname/GPU/2/', 0): [0, 2],
            ('/server/hostname/GPU/1/', '/server/hostname/GPU/2/', 0): [1, 2]
        }
    )
    assert net_sim.is_idle()
    next_node, next_time = net_sim.get_next_node()
    assert next_node is None and next_time is -1

    # Test enqueue node
    # A flow from GPU0 to GPU2
    send_0, recv_0 = create_netsim_nodes(
        0,
        '/server/hostname/GPU/0/',
        '/server/hostname/GPU/2/',
        0,
        net_sim,
        30
    )

    net_sim.enqueue_node(send_0, 0)
    next_node, next_time = net_sim.get_next_node()
    assert next_node == send_0 and next_time == 5
    # A flow from GPU1 to GPU2
    send_1, recv_1 = create_netsim_nodes(
        2,
        '/server/hostname/GPU/1/',
        '/server/hostname/GPU/2/',
        0,
        net_sim,
        30
    )
    net_sim.enqueue_node(send_1, 0)
    next_node, next_time = net_sim.get_next_node()
    assert next_node == send_0 and next_time == 30/4
    # Test dequeue node
    net_sim.dequeue_node()
    # Test enqueue a Recv node
    net_sim.enqueue_node(recv_0, 30/4)
    next_node, next_time = net_sim.get_next_node()
    assert next_node == recv_0 and next_time == 30/4
    net_sim.dequeue_node()
    next_node, next_time = net_sim.get_next_node()
    assert next_node == send_1 and next_time == 15

    # Test error handling
    with pytest.raises(ValueError):
        # wrong enqueue parameter: send_1 will dequeue in time: 15, so the
        # input time_now should be less than 15
        net_sim.enqueue_node(recv_1, 999999)

    with pytest.raises(ValueError):
        wrong_recv_node = Node(NodeMetadata(
            index=100, op="Recv",
            name=":recv:{0}:{1}:{2}:".format(
                '/server/hostname/GPU/0/',
                '/server/hostname/GPU/2/',
                0
            ),
            device_name=net_sim.name(),
            output_tensors=[Tensor('int32', 1)],  # should be []
            execution_time=10, input_ids=[],
            dependency_ids=[], successor_ids=[101]),
            net_sim)
        net_sim.enqueue_node(wrong_recv_node, 10)

    with pytest.raises(ValueError):
        # Test wrong Node
        wrong_send_2, wrong_recv_2 = create_netsim_nodes(
            4,
            '/server/hostname/GPU/1/',
            '/server/hostname/GPU/2/',
            999,  # wrong route_index
            net_sim,
            30
        )
        net_sim.enqueue_node(wrong_send_2, 10)


def create_netsim_nodes(index, src_name, dst_name,
                        route_index, device_obj, tensor_size):
    # Initialize Nodes
    send_node = Node(NodeMetadata(
        index=index, op="Send",
        name=":send:{0}:{1}:{2}:".format(
            src_name, dst_name, route_index
        ),
        device_name=device_obj.name(),
        output_tensors=[Tensor('int32', tensor_size)],
        execution_time=10, input_ids=[],
        dependency_ids=[], successor_ids=[index+1]),
        device_obj)
    recv_node = Node(NodeMetadata(
        index=index + 1, op="Recv",
        name=":recv:{0}:{1}:{2}:".format(
            src_name, dst_name, route_index
        ),
        device_name=device_obj.name(),
        output_tensors=[],  # The recv node's output_tensors should be []
        execution_time=10, input_ids=[],
        dependency_ids=[index], successor_ids=[]),
        device_obj)
    return (send_node, recv_node)
