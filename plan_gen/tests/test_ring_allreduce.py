import json
import os
from plan import ring_allreduce_plan


def test_ring_allreduce_plan():

    ring = ring_allreduce_plan.RingAllreducePlan(plan_name='ring')
    # Test get_plan_name() function
    assert(ring.get_plan_name() == 'ring')
    # Test get_plan_type() function
    assert(ring.get_plan_type() == 'Allreduce')

    # load input node_list
    path_input = os.path.join(os.path.dirname(__file__),
                              "data/test_ring_allreduce_input_nodes.json")
    nodes = json.load(open(path_input, 'r'))

    node = {
            'device': 'device_0',
            'name': 'test',
            'op': 'Allreduce',
            'output_shapes': [[1, 100]],
            'tensor_name': 'test',
            'tensor_type': 1,
            'input': []
            }

    # Test find_all_allreduce_nodes
    assert(ring.find_all_allreduce_nodes(nodes) == nodes)

    # Test find_endpoints
    assert(ring.find_endpoints(node, nodes) == nodes)

    # Test _generate_sr_send_node function
    sr_send_node = ring._generate_sr_send_node(node=node,
                                               index=0,
                                               sendAddress=0,
                                               sendTarget="device_1",
                                               nelem=0,
                                               device='device_0')
    sr_send_node_ref = {'name': 'test_Send_0',
                        'offset': 0,
                        'size': 0,
                        'op': 'Send',
                        'reduction': '',
                        'target': "device_1",
                        'related_op': 'test_Recv_1',
                        'device': 'device_0',
                        'output_shapes': [[1, 100]],
                        'tensor_name': 'test',
                        'tensor_type': 1,
                        'parent': 'test',
                        'input': []}
    assert(sr_send_node == sr_send_node_ref)

    # Test _generate_sr_recv_node function
    sr_recv_node = ring._generate_sr_recv_node(node=node,
                                               index=1,
                                               receiveAddress=0,
                                               recvTarget="device_1",
                                               nelem=0,
                                               device='device_0')
    sr_recv_node_ref = {'name': 'test_Recv_1',
                        'offset': 0,
                        'size': 0,
                        'op': 'Recv',
                        'reduction': 'sum',
                        'target': "device_1",
                        'related_op': 'test_Send_0',
                        'device': 'device_0',
                        'output_shapes': [[1, 100]],
                        'tensor_name': 'test',
                        'tensor_type': 1,
                        'parent': 'test',
                        'input': ['test_Send_0']}
    assert(sr_recv_node == sr_recv_node_ref)

    # Test _generate_ag_send_node function
    ag_send_node = ring._generate_ag_send_node(node=node,
                                               index=2,
                                               sendAddress=0,
                                               sendTarget="device_1",
                                               nelem=0,
                                               device='device_0')
    ag_send_node_ref = {'name': 'test_Send_2',
                        'offset': 0,
                        'size': 0,
                        'op': 'Send',
                        'reduction': '',
                        'target': "device_1",
                        'related_op': 'test_Recv_3',
                        'device': 'device_0',
                        'output_shapes': [[1, 100]],
                        'tensor_name': 'test',
                        'tensor_type': 1,
                        'parent': 'test',
                        'input': ['test_Recv_1']}
    assert(ag_send_node == ag_send_node_ref)

    # Test _generate_sr_recv_node function
    ag_recv_node = ring._generate_ag_recv_node(node=node,
                                               index=3,
                                               receiveAddress=0,
                                               recvTarget="device_1",
                                               nelem=0,
                                               device='device_0')
    ag_recv_node_ref = {'name': 'test_Recv_3',
                        'offset': 0,
                        'size': 0,
                        'op': 'Recv',
                        'reduction': 'copy',
                        'target': "device_1",
                        'related_op': 'test_Send_2',
                        'device': 'device_0',
                        'output_shapes': [[1, 100]],
                        'tensor_name': 'test',
                        'tensor_type': 1,
                        'parent': 'test',
                        'input': ['test_Send_2']}
    assert(ag_recv_node == ag_recv_node_ref)

    # Test generate_plan() function
    ring.reset_plan(nodes)
    output_plan = ring.generate_plan()
    path_output = os.path.join(os.path.dirname(__file__),
                               "data/test_ring_allreduce_output_ref.json")
    output_ref = json.load(open(path_output, 'r'))
    assert(output_plan == output_ref)
