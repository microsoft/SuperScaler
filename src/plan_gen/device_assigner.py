'''
Device Assigner
'''

import abc

import resource


class DeviceAssigner(abc.ABC):
    def __init__(self, resource_pool):
        self.resource_pool = resource_pool
        self.graphs = []

    def add_graph(self, graph):
        self.graphs.append(graph)

    @abc.abstractmethod
    def assign_device(self, graph):
        pass


class GPURoundRobin(DeviceAssigner):
    def __init__(self, resource_pool):
        super().__init__(resource_pool)
        self.pos = 0
        self.gpus = []
        for _, device in resource_pool.get_devices().items():
            if isinstance(device, resource.GPU):
                self.gpus.append(device)
        self.gpus.sort(key=lambda d: d.get_id())

    def assign_device(self, graph):
        if len(self.gpus) < len(self.graphs):
            raise ValueError("GPU count %s is less than graph count %s" % (
                                len(self.gpus),
                                len(self.graphs)))
        if len(self.gpus) < 1:
            raise ValueError("Resource Pool is empty")
        device = self.gpus[self.pos]
        self.pos += 1
        if self.pos >= len(self.gpus):
            self.pos = 0
        return device

        