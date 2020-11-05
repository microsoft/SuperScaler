import pytest
from frontend.ai_simulator.simulator.device_factory import DeviceFactory


def test_device_factory():
    df = DeviceFactory()
    # Wrong device_type
    with pytest.raises(ValueError):
        df.generate_device("NOT_VALID_TYPE", "/server/hostname1/GPU/0/")
    # Test functionality
    cpu_device = df.generate_device("CPU", "/server/hostname1/CPU/0/", '12bps')
    assert cpu_device.name() == "/server/hostname1/CPU/0/"
    assert cpu_device.get_performance() == 12
    net_sim = df.generate_device(
        'NetworkSimulator',
        'NetworkSimulator',
        [
            {
                'link_id': 0, 'source_name': '/server/hostname/GPU/0/',
                'dest_name': '/switch/switch0/',
                'capacity': '80bps'
            },
            {
                'link_id': 2, 'source_name': '/switch/switch0/',
                'dest_name': '/server/hostname/GPU/2/',
                'capacity': '80bps'
            }
        ],
        {('/server/hostname/GPU/0/', '/server/hostname/GPU/2/', 0): [0, 2]}
    )
    assert net_sim.name() == 'NetworkSimulator'
    assert net_sim.is_idle()
    next_node, next_time = net_sim.get_next_node()
    assert next_node is None and next_time == -1
