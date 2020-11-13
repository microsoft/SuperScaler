import pytest

from superscaler.plan_gen.plan.resources.server import Server
from superscaler.plan_gen.plan.resources.hardware import GPUHardware,\
    CPUHardware, Hardware


def test_server():
    # Wrong server name
    with pytest.raises(ValueError):
        Server('/not_a_server/win/')
    server = Server("/server/win/")
    # Test server.add_hardware
    gpu = GPUHardware('/server/win/GPU/0/')
    cpu = CPUHardware('/server/win/CPU/0/')
    server.add_hardware(gpu)
    server.add_hardware(cpu)
    raw_hw = Hardware('/server/win/not_computational_hardware/0/')
    # Test adding a non computational hardware to the server
    with pytest.raises(ValueError):
        server.add_hardware(raw_hw)
    # Test get_hardware()
    hw_dict = server.get_hardware_dict()
    assert hw_dict == {gpu.get_name(): gpu, cpu.get_name(): cpu}
    another_host_gpu = GPUHardware('/server/linux/GPU/0/')
    # Test adding a computational hardware that is not connected to the server
    with pytest.raises(ValueError):
        server.add_hardware(another_host_gpu)
    # Test adding two computational hardware with same name
    gpu_1 = GPUHardware('/server/win/GPU/1/')
    server.add_hardware(gpu_1)
    with pytest.raises(ValueError):
        server.add_hardware(GPUHardware('/server/win/GPU/1/'))
    # Test get_hardware_list_from_type
    cpu_list = server.get_hardware_list_from_type('CPU')
    assert cpu_list == [cpu]
    # Test get_hardware_from_name
    my_gpu = server.get_hardware_from_name('/server/win/GPU/1/')
    assert my_gpu == gpu_1
