import pytest

from superscaler.plan_gen.plan.resources.router import Router
from superscaler.plan_gen.plan.resources.link import PCIE, RDMA
from superscaler.plan_gen.plan.resources.hardware import CPUHardware,\
    GPUHardware, NetworkSwitchHardware


def test_router():
    # Init two CPUs: cpu0 --RDMA--> cpu1
    cpu_0 = CPUHardware('/server/hostname1/CPU/0/')
    cpu_1 = CPUHardware('/server/hostname1/CPU/1/')
    link_cpu0_cpu1 = RDMA(0, cpu_0.get_name(), cpu_1.get_name())
    cpu_0.add_link(link_cpu0_cpu1)
    cpu_1.add_link(link_cpu0_cpu1)
    hardware_dict = {cpu_0.get_name(): cpu_0, cpu_1.get_name(): cpu_1}
    # Init router
    router = Router(hardware_dict)
    # Test get_route_info
    cpu_route = router.get_route_info(cpu_0.get_name(), cpu_1.get_name())
    assert cpu_route == [([link_cpu0_cpu1], 'RDMA')]

    # Init GPUs: gpu_0 --PCIE--> SW --PCIE--> gpu_1
    gpu_0 = GPUHardware('/server/hostname1/GPU/0/')
    gpu_1 = GPUHardware('/server/hostname1/GPU/1/')
    switch = NetworkSwitchHardware('/switch/switch0/')
    link_gpu0_sw = PCIE(1, gpu_0.get_name(), switch.get_name())
    link_sw_gpu1 = PCIE(2, switch.get_name(), gpu_1.get_name())
    link_gpu0_gpu1 = RDMA(3, gpu_0.get_name(), gpu_1.get_name())
    gpu_0.add_link(link_gpu0_sw)
    switch.add_link(link_gpu0_sw)
    gpu_1.add_link(link_sw_gpu1)
    switch.add_link(link_sw_gpu1)
    gpu_0.add_link(link_gpu0_gpu1)
    gpu_1.add_link(link_gpu0_gpu1)
    new_hw_dict = {gpu_0.get_name(): gpu_0, gpu_1.get_name(): gpu_1,
                   switch.get_name(): switch}
    router.update_hardware_dict(new_hw_dict)

    gpu_route = router.get_route_info(
        '/server/hostname1/GPU/0/', '/server/hostname1/GPU/1/')
    assert len(gpu_route) == 2
    assert gpu_route == [
        ([link_gpu0_sw, link_sw_gpu1], "PCIE"),
        ([link_gpu0_gpu1], "RDMA")]
    assert router.get_route_info(
        '/server/hostname1/GPU/1/', '/server/hostname1/GPU/0/'
    ) == []

    # Test error handling
    with pytest.raises(ValueError):
        # Wrong init param: init router with a list
        Router([0, 1, 2])

    with pytest.raises(ValueError):
        # Wrong init param: Init router without switch hardware
        Router({gpu_0.get_name(): gpu_0, gpu_1.get_name(): gpu_1})
