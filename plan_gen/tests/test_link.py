from resources.link import Link, PCIE, RDMA


def test_link():
    hw_0 = "hardware_0"
    hw_1 = "hardware_1"
    test_link = Link(hw_0, hw_1, '199bps', '12s')
    test_pcie = PCIE(hw_0, hw_1, '123bps', '1s')
    test_rmda = RDMA(hw_0, hw_1, '1232bps', '1s', 'FairSharing')
    assert test_pcie.source_hardware == hw_0
    assert test_pcie.dest_hardware == hw_1
    assert test_pcie.latency == '1s'
    assert test_pcie.capacity == 123
    assert test_pcie.scheduler == 'FIFO'
    assert test_link.get_name() == '#link#BasicType#hardware_0#hardware_1'
    assert test_pcie.get_name() == '#link#PCIE#hardware_0#hardware_1'
    assert test_rmda.get_name() == '#link#RDMA#hardware_0#hardware_1'
