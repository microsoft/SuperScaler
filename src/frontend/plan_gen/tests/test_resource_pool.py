import os
import pytest

from frontend.plan_gen.plan.resources.resource_pool import ResourcePool


def test_resource_pool_functionality():
    resource_yaml_path = os.path.join(
        os.path.dirname(__file__), 'data', 'resource_pool.yaml')
    rp = ResourcePool()
    # Test yaml parser
    rp.init_from_yaml(resource_yaml_path)

    # Test get_servers
    servers = rp.get_servers()
    assert len(servers) == 1
    for server_name, server in servers.items():
        hardware_list = server.get_hardware_dict()
        assert len(hardware_list) == 6
        for hardware in server.get_hardware_dict():
            print(hardware)

    # Test get_switches
    switches = rp.get_switches()
    assert len(switches) == 1
    for switch_name, switch in switches.items():
        assert len(switch.get_inbound_links()) == 4
        assert len(switch.get_outbound_links()) == 4
        print(switch)

    # Test get_links
    links = rp.get_links()
    assert len(links) == 12
    ref_links_src_dest = \
        [("/server/hostname1/CPU/1/", "/server/hostname1/CPU/0/"),
         ("/server/hostname1/CPU/0/", "/server/hostname1/CPU/1/")] \
        + [("/server/hostname1/GPU/{0}/".format(i),
            "/switch/switch0/") for i in range(4)] \
        + [("/switch/switch0/",
            "/server/hostname1/GPU/{0}/".format(i)) for i in range(4)] \
        + [("/server/hostname1/GPU/0/", "/server/hostname1/GPU/1/"),
           ("/server/hostname1/GPU/1/", "/server/hostname1/GPU/0/")]
    unique_link_id_set = set()
    for link in links:
        assert (link.source_hardware, link.dest_hardware) in ref_links_src_dest
        assert link.link_id not in unique_link_id_set
        unique_link_id_set.add(link.link_id)

    # Test get_resource_from_name
    cpu_0 = rp.get_resource_from_name('/server/hostname1/CPU/0/')
    assert cpu_0.get_performance() == 12*2**30
    assert rp.get_resource_from_name('NO_THIS_RESOURCE') is None

    # Test get_hardware_list_from_type
    with pytest.raises(ValueError):
        rp.get_resource_list_from_type('NO_THIS_TYPE')
    # Test CPUHardware type
    cpu_1 = rp.get_resource_from_name('/server/hostname1/CPU/1/')
    assert rp.get_resource_list_from_type('CPU') == [cpu_0, cpu_1]
    # Test GPUHardware type
    gpu_list = rp.get_resource_list_from_type("GPU")
    correct_gpu_name = [
        "/server/hostname1/GPU/{0}/".format(i) for i in range(4)]
    assert len(gpu_list) == len(correct_gpu_name)
    for gpu in gpu_list:
        assert gpu.get_name() in correct_gpu_name
    # Test Server type
    server_list = rp.get_resource_list_from_type("Server")
    assert len(server_list) == 1 \
        and server_list[0].get_name() == '/server/hostname1/'

    # Test get_route_info
    cpu_1 = rp.get_resource_from_name('/server/hostname1/CPU/1/')
    cpu_0 = rp.get_resource_from_name('/server/hostname1/CPU/0/')
    cpu_route = rp.get_route_info(cpu_0.get_name(), cpu_1.get_name())
    assert cpu_route == [
        ([cpu_0.get_outbound_links()[cpu_1.get_name()][0]],
         'RDMA')]
    gpu_route = rp.get_route_info(
        '/server/hostname1/GPU/0/', '/server/hostname1/GPU/1/')
    gpu_0 = rp.get_resource_from_name('/server/hostname1/GPU/0/')
    switch = rp.get_resource_from_name('/switch/switch0/')
    assert len(gpu_route) == 2
    assert gpu_route[0] == (
        [gpu_0.get_outbound_links()[switch.get_name()][0],
         switch.get_outbound_links()['/server/hostname1/GPU/1/'][0]],
        "PCIE"
    )
    assert rp.get_route_info(
        '/server/hostname1/CPU/1/', '/server/hostname1/GPU/0/'
    ) == []

    # Test get_links_as_list()
    all_links_info = []
    for link in links:
        all_links_info.append(link.to_dict())
    assert rp.get_links_as_list() == all_links_info

    # Test get_computational_hardware_as_list()
    print(rp.get_computational_hardware_as_list())
    assert rp.get_computational_hardware_as_list() == [
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
