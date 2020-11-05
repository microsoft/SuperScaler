from frontend.ai_simulator.simulator.computation_device import CPU, GPU


def test_calculation_device():
    test_CPU = CPU("/server/hostname1/CPU/0", '12bps')
    test_GPU = GPU("/server/hostname1/GPU/0", '120bps')
    assert test_CPU.get_performance() == 12
    assert test_GPU.get_performance() == 120
