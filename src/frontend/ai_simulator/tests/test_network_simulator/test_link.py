from frontend.ai_simulator.simulator.network_simulator.link import Link
from test_flow import create_test_flow


def test_link():
    # Init link
    link_0 = Link(0, '/server/hostname/GPU/0/',
                  '/switch/switch0/',
                  '100bps')
    flow_0 = create_test_flow(0, 'send', '/switch/switch0/', 5, 1)
    flow_1 = create_test_flow(1, 'send', '/switch/switch0/', 5, 1)

    # Test link properties
    assert link_0.link_id == 0
    assert link_0.source_name == '/server/hostname/GPU/0/'
    assert link_0.dest_name == '/switch/switch0/'
    assert link_0.capacity == 100
    assert link_0.latency == '0s'
    assert link_0.flows == []

    # Test add_flow & delete_flow
    link_0.add_flow(flow_0)
    link_0.add_flow(flow_1)
    assert link_0.flows == [flow_0, flow_1]
    link_0.delete_flow(flow_1)
    assert link_0.flows == [flow_0]
