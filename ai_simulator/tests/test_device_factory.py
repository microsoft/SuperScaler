import pytest
from simulator.device_factory import DeviceFactory


def test_device_factory():
    df = DeviceFactory()
    # Wrong device_type
    with pytest.raises(ValueError):
        df.generate_device("NOT_VALID_TYPE", "/server/hostname1/GPU/0/")
    # Test functionality
    cpu_device = df.generate_device("CPU", "/server/hostname1/CPU/0/", '12bps')
    assert cpu_device.name() == "/server/hostname1/CPU/0/"
    assert cpu_device.get_performance() == 12
