import os

from plan_gen import ResourcePool, TFParser, PlanGenerator
from ai_simulator import Simulator, PlanAdapter


def init_nodelist(graph_dir):
    '''get an example nodelist
    '''
    # change relative path to absolute path
    graph_path = os.path.join(
        os.path.dirname(__file__), graph_dir)

    # get all graphs and devices
    device_ids = os.listdir(graph_path)
    graph_paths = [
        os.path.join(graph_path, i + "/graph.pbtxt") for i in device_ids]
    devices = ["device_" + i for i in device_ids]

    # parse the graph into nodelist
    parser = TFParser()
    nodelist = parser.parse_graphs(graph_paths, devices)

    return nodelist


def init_resource_pool(rp_path):
    '''get an example resource pool
    '''
    resource_yaml_path = os.path.join(os.path.dirname(__file__), rp_path)
    rp = ResourcePool()
    rp.init_from_yaml(resource_yaml_path)

    return rp


def init_simulator(node_list, rp):
    # Init the PlanGenerator
    plan_generator = PlanGenerator(node_list, rp)

    # get plan, links, routing and devices info from plan_gen
    mapped_plan = plan_generator.\
        get_execution_plan('Allreduce', 'ring').to_json()
    links_info = plan_generator.get_links_info()
    routing_info = plan_generator.get_routing_info()
    compu_device_spec = plan_generator.get_device_info()

    # use PlanAdapter to get json node_list
    adapter = PlanAdapter()
    assert adapter.set_plan(mapped_plan) is True
    sim_nodes_list = adapter.get_plan()

    # packages computing devices and network info into device_list
    device_list = [
        ('NetworkSimulator', ['NetworkSimulator', links_info, routing_info])
    ]
    for device_spec in compu_device_spec:
        device_list.append(
            (device_spec['type'],
             [device_spec['name'], device_spec['performance']])
        )

    # generate Simulator
    return Simulator(sim_nodes_list, device_list)


def test_integration():
    # Init nodelist
    graph_dir = '../plan_gen/examples/CNN_vgg16_imagenet/' + \
        'PureDataParallelismPlan2GPUsIn1Hosts/'
    nodelist = init_nodelist(graph_dir)

    # Init reasourse pool
    rp_file = '../plan_gen/tests/data/resource_pool.yaml'
    rp = init_resource_pool(rp_file)

    # Init Simulator
    simulator = init_simulator(nodelist, rp)

    # Run Simulator
    time_use, start_time, finish_time = simulator.run()

    print("total time use:", time_use)
    print('--------start time---------')
    print(start_time)
    print('--------finish time---------')
    print(finish_time)

    assert time_use == 55343017.6
