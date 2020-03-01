import pickle

import resource
import device_assigner


def test_gpu_round_robin_assign():
    rp = pickle.load(open("tests/data/resource_pool.data", "rb"))
    assigner = device_assigner.GPURoundRobin(rp)
    target_gpu_count = 4
    for i in range(target_gpu_count):
        assigner.add_graph(None)
    gpus = set()
    while True:
        device = assigner.assign_device(None)
        assert(isinstance(device, resource.GPU))
        if device in gpus:
            break
        gpus.add(device)
    assert(len(gpus) == target_gpu_count)
