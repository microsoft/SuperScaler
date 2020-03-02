#!/usr/bin/env python3

'''
Resource Pool
'''

import abc
import yaml



'''
# Servers
hostname1:
    CPUs:
        0:
            links:
                - "/hostname1/CPU/1/":
                    type: "RDMA"
        1:
    GPUs:
        0:
            links:
                - "/hostname1/GPU/1/":
                    type: "PCIE"
        1:
'''
class ResourcePool(object):
    def __init__(self, path):
        self.metadata = yaml.load(open(path, "r"), Loader=yaml.FullLoader)
        self.devices = {}
        self.__load_device()
        self.__load_link()

    def get_metadata(self):
        return self.metadata

    def get_devices(self):
        return self.devices

    def __load_device(self):
        for hostname, server in self.metadata.items():
            self.__create_server(hostname, server)
        # for _, cluster in self.metadata.items():
        #     self.create_cluster(cluster)
    
    # def create_cluster(self, cluster_metadata):
    #     for hostname, server in cluster_metadata.items():
    #         self.__create_server(hostname, server)

    def __create_server(self, hostname, server_metadata):
        server = Server(hostname)
        for device_id, device_metadata in server_metadata.get("CPU", {}).items():
            cpu = CPU(device_id)
            server.add_cpu(cpu)
            if not device_metadata:
                server_metadata["CPU"][device_id] = {}
                device_metadata = server_metadata["CPU"][device_id]
            self.__create_device(cpu, device_metadata)
        for device_id, device_metadata in server_metadata.get("GPU", {}).items():
            gpu = GPU(device_id)
            server.add_gpu(gpu)
            if not device_metadata:
                server_metadata["GPU"][device_id] = {}
                device_metadata = server_metadata["GPU"][device_id]
            self.__create_device(gpu, device_metadata)

    def __create_device(self, device, device_metadata):
        device.metadata = device_metadata
        if "links" not in device.metadata:
            device.metadata["links"] = []
        self.devices[device.get_id()] = device

    def __load_link(self):
        # Add default PCI link for all devices in the same machine
        for _, device in self.devices.items():
            device_list = device.host.get_devices()
            for target_device in device_list:
                if target_device == device:
                    continue
                device.metadata["links"].append({
                    target_device.get_id():
                    {
                        "type": "PCIE"
                    }
                })
            for link in device.metadata["links"]:
                self.__create_link(device, link)
    
    def __create_link(self, device, link_metadata):
        if len(link_metadata) != 1:
            raise ValueError("Error link format")
        path, link_metadata = list(link_metadata.items())[0]
        target_device = self.devices[path]
        link_type = link_metadata["type"]
        eval(link_type)(device, target_device)


class Resource(abc.ABC):
    @abc.abstractmethod
    def get_id(self):
        pass

    def __eq__(self, obj):
        return self.get_id() == obj.get_id()

    def __hash__(self):
        return id(self)


class Server(Resource):
    def __init__(self, hostname):
        self.hostname = hostname
        self.cpus = []
        self.gpus = []

    def add_cpu(self, cpu):
        cpu.host = self
        self.cpus.append(cpu)

    def add_gpu(self, gpu):
        gpu.host = self
        self.gpus.append(gpu)

    def get_devices(self):
        return self.cpus + self.gpus

    def get_id(self):
        return self.hostname
    
    def __str__(self):
        return "hostname : %s\ncpus : %s\ngpus : %s\n" % \
            (self.hostname, str(self.cpus), str(self.gpus))


class Device(Resource):
    def __init__(self, device_id):
        self.id  = device_id
        self.neighbors = {}

    def add_link(self, device, link):
        if device is self:
            raise ValueError("Cannot add self to its device")
        device_id = device.get_id()
        if device_id not in self.neighbors:
            self.neighbors[device_id] = []
        self.neighbors[device_id].append(link)

    def __str__(self):
        buffer = "id : " + self.get_id() + " [\n"
        for target, links in self.neighbors.items():
            buffer += " " + target + " : "
            for link in links:
                buffer += str(link.get_id()) + ","
            buffer += "\n"
        buffer += "]"
        return buffer


class CPU(Device):
    def get_id(self):
        return "/%s/CPU/%s/" % (
            self.host.hostname,
            self.id
        )


class GPU(Device):
    def get_id(self):
        return "/%s/GPU/%s/" % (
            self.host.hostname,
            self.id
        )


class Link(Resource):
    def __init__(self, device, target_device):
        self.device = device
        self.target_device = target_device
        self.device.add_link(self.target_device, self)
    
    def get_id(self):
        return (self.device.get_id(), self.target_device.get_id())


class RDMA(Link):
    def get_id(self):
        return ("RDMA", *super().get_id())


class PCIE(Link):
    def get_id(self):
        return ("PCIE", *super().get_id())
