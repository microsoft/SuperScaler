import pickle

import resource
import device_assigner


def test_gpu_round_robin_assign():
    rp = pickle.load(open("tests/data/resource_pool.data", "rb"))
    assigner = device_assigner.GPURoundRobin(rp)
    target_gpu_count = 4
    running_graphs = [{"Test": "Graph"} for i in range(target_gpu_count)]
    for graph in running_graphs:
        assigner.add_graph(graph)
    gpus = set()
    i = 0
    while i < len(running_graphs):
        device = assigner.assign_device(running_graphs[i])
        assert(isinstance(device, resource.GPU))
        if device in gpus:
            break
        gpus.add(device)
        i += 1
    assert(len(gpus) == target_gpu_count)
