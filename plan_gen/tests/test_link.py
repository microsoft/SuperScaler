from plan.resources.link import Link, PCIE, RDMA


def test_link():
    hw_0 = "hardware_0"
    hw_1 = "hardware_1"
    # Test basic Link
    test_link = Link(0, hw_0, hw_1, '199bps', '12s', 'FIFO')
    assert test_link.get_name() == '#link#BasicType#hardware_0#hardware_1'
    assert test_link.source_hardware == hw_0
    assert test_link.dest_hardware == hw_1
    assert test_link.latency == '12s'
    assert test_link.capacity == 199
    assert test_link.scheduler == 'FIFO'
    assert test_link.link_id == 0
    assert test_link.to_dict() == {
        'source_name': hw_0,
        'dest_name': hw_1,
        'capacity': '199.0bps',
        'latency': '12s',
        'scheduler': 'FIFO',
        'link_type': 'BasicType',
        'link_id': 0
    }


def test_PCIE_link():
    hw_0 = "hardware_0"
    hw_1 = "hardware_1"
    # Test PCIE link
    test_pcie = PCIE(1, hw_0, hw_1, '123bps', '1s', 'FIFO')
    assert test_pcie.get_name() == '#link#PCIE#hardware_0#hardware_1'
    assert test_pcie.source_hardware == hw_0
    assert test_pcie.dest_hardware == hw_1
    assert test_pcie.latency == '1s'
    assert test_pcie.capacity == 123
    assert test_pcie.scheduler == 'FIFO'
    assert test_pcie.link_id == 1
    assert test_pcie.to_dict() == {
        'source_name': hw_0,
        'dest_name': hw_1,
        'capacity': '123.0bps',
        'latency': '1s',
        'scheduler': 'FIFO',
        'link_type': 'PCIE',
        'link_id': 1
    }


def test_RDMA_link():
    hw_0 = "hardware_0"
    hw_1 = "hardware_1"
    # Test RDMA link
    test_rmda = RDMA(2, hw_0, hw_1, '1232bps', '1s', 'FairSharing')
    assert test_rmda.get_name() == '#link#RDMA#hardware_0#hardware_1'
    assert test_rmda.source_hardware == hw_0
    assert test_rmda.dest_hardware == hw_1
    assert test_rmda.latency == '1s'
    assert test_rmda.capacity == 1232
    assert test_rmda.scheduler == 'FairSharing'
    assert test_rmda.link_id == 2
    assert test_rmda.to_dict() == {
        'source_name': hw_0,
        'dest_name': hw_1,
        'capacity': '1232.0bps',
        'latency': '1s',
        'scheduler': 'FairSharing',
        'link_type': 'RDMA',
        'link_id': 2
    }
