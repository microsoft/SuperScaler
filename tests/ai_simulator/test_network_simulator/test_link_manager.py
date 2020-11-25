# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import pytest

from superscaler.ai_simulator.simulator.network_simulator.link_manager import \
     LinkManager


def test_link_manager():
    links_spec = []
    for i in range(5):
        links_spec.append(
            {'link_id': i,
             'source_name': 'device' + str(i),
             'dest_name': 'device' + str(i+1),
             'capacity': str(i)+'bps'}
        )
    links_spec.append(
        {'link_id': 5,
         'source_name': 'device0',
         'dest_name': 'device5',
         'capacity': '10086bps'}
    )
    routing_info_dict = {
        ('device0', 'device5', 0): [0, 1, 2, 3, 4],
        ('device0', 'device5', 1): [5]}
    lm = LinkManager(links_spec, routing_info_dict)
    # Test get_link
    link0 = lm.get_link(0)
    assert link0.source_name == 'device0'
    assert link0.dest_name == 'device1'
    assert link0.capacity == 0
    # Test get_link with a wrong link_id
    assert lm.get_link(10086) is None
    # Test get_links_dict
    all_links = lm.get_links_dict()
    assert len(all_links.values()) == 6
    link_id_set = set()
    for link_id, link_obj in all_links.items():
        assert link_id not in link_id_set
        link_id_set.add(link_id)
        if link_id != 5:
            assert link_obj.source_name == 'device' + str(link_id)
            assert link_obj.dest_name == 'device' + str(link_id + 1)
            assert link_obj.capacity == link_id
        else:
            assert link_obj.source_name == 'device0'
            assert link_obj.dest_name == 'device5'
            assert link_obj.capacity == 10086

    # Test get_routing_path
    assert lm.get_routing_path('device0', 'device5', 1) == [all_links[5]]
    assert lm.get_routing_path('device0', 'device5', 0) == \
        [all_links[i] for i in range(5)]

    # Test get_routing
    assert lm.get_routing(":send:device0:device5:1:") == [all_links[5]]
    # Test get_routing: wrong input format
    assert lm.get_routing(":send:device0:device5:") is None
    # Test error handling
    with pytest.raises(ValueError):
        # links_spec should be a list
        LinkManager(
            {'link_id': 5,
             'source_name': 'device0',
             'dest_name': 'device5',
             'capacity': '10086bps'},
            routing_info_dict)
    with pytest.raises(ValueError):
        # link_id should be unique
        duplicated_links_spec = [
            {'link_id': 5,
             'source_name': 'device0',
             'dest_name': 'device5',
             'capacity': '10086bps'},
            {'link_id': 5,
             'source_name': 'device0',
             'dest_name': 'device5',
             'capacity': '10086bps'}
        ]
        LinkManager(duplicated_links_spec, routing_info_dict)

    with pytest.raises(ValueError):
        # routing_info_dict should be a dict
        LinkManager(links_spec, [0, 1, 2])
    with pytest.raises(ValueError):
        # the key should be (src_name, dst_name, route_index)
        wrong_routing_info_dict = {
            ('device0', 'device5'): [0, 1, 2, 3, 4]}
        LinkManager(links_spec, wrong_routing_info_dict)

    # Test wrong functionality:
    wrong_routing_info_dict = {
        ('device0', 'device5', 0): [0, 1, 4]}
    lm = LinkManager(links_spec, wrong_routing_info_dict)
    assert not lm.get_routing_path('device0', 'device5', 0)
