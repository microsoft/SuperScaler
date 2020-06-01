import pytest

from resources.link import PCIE
from resources.hardware import Hardware, ComputationHardware, CPUHardware
from resources.hardware import GPUHardware, NetworkSwitchHardware


def test_hardware():
    hw_0 = Hardware("/server/wind4/GPU/0/")
    hw_1 = Hardware("/server/wind4/GPU/1/")
    assert hw_0.get_name() == "/server/wind4/GPU/0/"
    assert hw_0.get_name() != hw_1.get_name()
    # Add correct link
    hw_0.add_link(PCIE(
        hw_0.get_name(), hw_1.get_name(), '600bps', '3us'))
    assert len(hw_0.get_outbound_links()) == 1

    hw_2 = Hardware("/server/wind4/GPU/2/")
    # Add error link
    with pytest.raises(ValueError):
        hw_0.add_link(PCIE(
            hw_1.get_name(), hw_2.get_name(), '600bps', '5us'))
    with pytest.raises(ValueError):
        hw_0.add_link(PCIE(
            hw_0.get_name(), hw_0.get_name(), '600bps', '5us'))

    # Add correct 2 links
    hw_0.add_link(PCIE(
        hw_0.get_name(), hw_2.get_name(), '600bps', '5us'))
    hw_0.add_link(PCIE(
        hw_0.get_name(), hw_2.get_name(), '600bps', '1us'))
    assert len(hw_0.get_outbound_links()) == 2
    # Test get_outbound_links() function
    links = hw_0.get_outbound_links()
    assert len(links["/server/wind4/GPU/1/"]) == 1
    assert len(links["/server/wind4/GPU/2/"]) == 2

    # Test get_inbound_links() function
    inbound_pcie_link = PCIE(hw_2.get_name(), hw_0.get_name(), '600bps', '5us')
    hw_0.add_link(inbound_pcie_link)
    assert hw_0.get_inbound_links() == {
        inbound_pcie_link.source_hardware: [inbound_pcie_link]}
    # No need to test __str__() function. It only servers for display purpose.


def test_computation_hardware():
    compute_hw = ComputationHardware('/server/my_computer/GPU/0/', '100bps')
    assert compute_hw.get_performance() == 100
    # Test hardware name parser
    with pytest.raises(ValueError):
        ComputationHardware.get_computation_hardware_description(
            "/server/wind4/GPU/")
    with pytest.raises(ValueError):
        ComputationHardware.get_computation_hardware_description(
            "/switch/wind4/GPU/0/")
    assert ComputationHardware.get_computation_hardware_description(
        "/server/my_computer/GPU/0/") == ("my_computer", "GPU", "0", [''])


def test_CPU_hardware():
    cpu_correct = CPUHardware('/server/hostname/CPU/0/', '100bps')
    assert cpu_correct.get_name() == '/server/hostname/CPU/0/'
    # Init a cpu hardware with wrong names
    with pytest.raises(ValueError):
        CPUHardware('/server/hostname/GPU/0/', '100bps')
    with pytest.raises(ValueError):
        CPUHardware('/not_a_server/hostname/CPU/0/', '100bps')
    with pytest.raises(ValueError):
        CPUHardware('/server/hostname/CPU', '100bps')


def test_GPU_hardware():
    gpu_correct = GPUHardware('/server/hostname/GPU/0/', '100bps')
    assert gpu_correct.get_name() == '/server/hostname/GPU/0/'
    # Init a cpu hardware with wrong names
    with pytest.raises(ValueError):
        GPUHardware('/server/hostname/CPU/0/', '100bps')
    with pytest.raises(ValueError):
        GPUHardware('/not_a_server/hostname/GPU/0/', '100bps')
    with pytest.raises(ValueError):
        GPUHardware('/server/hostname/GPU', '100bps')


def test_network_device():
    # Init a NetworkSwitch hardware with wrong names
    with pytest.raises(ValueError):
        NetworkSwitchHardware('/not_switch/swname/')
    pcie_sw = NetworkSwitchHardware('/switch/pciesw/')
    assert pcie_sw.get_name() == '/switch/pciesw/'
