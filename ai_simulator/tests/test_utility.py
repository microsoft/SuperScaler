import random
import pytest
import humanreadable

from simulator.utility import transfer_rate_to_bps, data_size_to_bit


def test_transfer_rate_converter():
    # Test bps --> bps
    for i in range(10):
        # Generate a random transfer rate [0, 1000)
        rand_rate = random.random() * 1000
        rate_str = str(rand_rate) + 'bps'
        assert rand_rate == transfer_rate_to_bps(rate_str)
    # Test kibps --> bps
    for i in range(10):
        rand_rate = random.random() * 1000
        rate_str = str(rand_rate) + 'Kibit/s'
        assert rand_rate * 1024 == transfer_rate_to_bps(rate_str)
    # Test error handler
    with pytest.raises(humanreadable.error.UnitNotFoundError):
        transfer_rate_to_bps('12ki/s')


def test_data_size_converter():
    # Test bit --> bit
    for i in range(10):
        # Generate a random data size [0, 1000)
        rand_size = random.random() * 1000
        size_str = str(rand_size) + 'b'
        assert rand_size == data_size_to_bit(size_str)
    # Test kibit --> bit
    for i in range(10):
        rand_size = random.random() * 1000
        size_str = str(rand_size) + 'Kib'
        assert rand_size * 1024 == data_size_to_bit(size_str)
    # Test error handler
    with pytest.raises(ValueError):
        data_size_to_bit('10kib')
