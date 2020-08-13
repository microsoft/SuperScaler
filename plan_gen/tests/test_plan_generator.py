import os
import json
from plan.adapter.tf_parser import TFParser
from plan.resources.resource_pool import ResourcePool
from plan.plan_generator import PlanGenerator


def test_plan_generator():

    def get_device(device_count):
        # get virtual device names
        return ["device_%d" % (i) for i in range(device_count)]

    def get_graph_paths(path, device_count):
        # Parser graph from paths
        graph_paths = []
        for i in range(device_count):
            sub_path = os.path.join(path, str(i) + "/graph.pbtxt")
            graph_paths.append(sub_path)
        return graph_paths

    # Init nodelist
    device_count = 2
    parser = TFParser(
        db_file_path='tests/data/tf_parser_testbench/db_test.json')
    devices = get_device(device_count)
    graph_paths = get_graph_paths(
        "examples/CNN_vgg16_imagenet/PureDataParallelismPlan2GPUsIn1Hosts/",
        device_count)
    parser = TFParser(
        db_file_path='tests/data/tf_parser_testbench/db_test.json')
    nodelist = parser.parse_graphs(graph_paths, devices)

    # Init ResourcePool
    resource_yaml_path = os.path.join(
        os.path.dirname(__file__), 'data', 'resource_pool.yaml')
    rp = ResourcePool()
    rp.init_from_yaml(resource_yaml_path)

    # Init PlanManager by PlanPool and PlanMapper
    plan_generator = PlanGenerator(nodelist, rp)

    # Check the correctness of output
    plan_ring = plan_generator.get_execution_plan('Allreduce', 'ring')
    output_path = os.path.join(
        os.path.dirname(__file__), "data/ring_vgg16.json")

    output_ref = json.load(open(output_path, 'r'))
    assert(plan_ring.to_json() == output_ref)

    route_info = plan_generator.get_routing_info()
    assert(len(route_info) == 2)
    assert(route_info == {('/server/hostname1/GPU/0/',
                           '/server/hostname1/GPU/1/', 0): [2, 9],
                          ('/server/hostname1/GPU/1/',
                           '/server/hostname1/GPU/0/', 0): [4, 8]})

    device_info = plan_generator.get_device_info()
    assert device_info == [
        {'performance': '12884901888.0bps',
         'name': '/server/hostname1/CPU/0/',
         'type': 'CPU'},
        {'performance': '12884901888.0bps',
         'name': '/server/hostname1/CPU/1/',
         'type': 'CPU'},
        {'performance': '13194139533312.0bps',
         'name': '/server/hostname1/GPU/0/',
         'type': 'GPU'},
        {'performance': '13194139533312.0bps',
         'name': '/server/hostname1/GPU/1/',
         'type': 'GPU'},
        {'performance': '13194139533312.0bps',
         'name': '/server/hostname1/GPU/2/',
         'type': 'GPU'},
        {'performance': '13194139533312.0bps',
         'name': '/server/hostname1/GPU/3/',
         'type': 'GPU'}
    ]
