import pickle

import resource


def test_resource_pool():
    rp = resource.ResourcePool("tests/data/resource_pool.yaml")
    # # Dump target
    # pickle.dump(
    #     rp,
    #     open("tests/data/resource_pool.data", "wb")
    # )
    rp_expected = pickle.load(open("tests/data/resource_pool.data", "rb"))
    devices = rp.get_devices()
    devices_expected = rp_expected.get_devices()
    # Check device
    assert(devices == devices_expected)
    # Check links
    for id, device in devices.items():
        links = device.neighbors
        links_expected = devices_expected[id].neighbors
        assert(links == links_expected)
    # Check metadata
    assert(rp.get_metadata() == rp_expected.get_metadata())
