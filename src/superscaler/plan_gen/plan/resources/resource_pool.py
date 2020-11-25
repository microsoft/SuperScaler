# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import yaml

from superscaler.plan_gen.plan.resources.hardware import CPUHardware,\
    GPUHardware, NetworkSwitchHardware
from superscaler.plan_gen.plan.resources.server import Server
from superscaler.plan_gen.plan.resources.link import PCIE, RDMA
from superscaler.plan_gen.plan.resources.router import Router


class ResourcePool():

    # Use a dict to store valid_hardware_type string instead of using
    # dangerous eval() function
    valid_computational_hardware_type = {'CPUHardware': CPUHardware,
                                         'CPU': CPUHardware,
                                         'GPUHardware': GPUHardware,
                                         'GPU': GPUHardware}
    valid_resource_type = dict(
        {'Server': Server, 'NetworkSwitchHardware': NetworkSwitchHardware},
        **valid_computational_hardware_type)

    def __init__(self):
        self.__servers = {}  # {server_name: obj}
        self.__computational_hardware = {}  # {hardware_name: obj}
        self.__switches = {}  # {switch_name: obj}
        self.__links = []
        self.__router = None  # A Router object

    def init_from_yaml(self, yaml_path):
        '''Reset resource pool and init via input yaml file

        Args:
            yaml_path: string, the path of the yaml file
        '''
        with open(yaml_path) as f:
            resources_yaml_data = yaml.load(f, Loader=yaml.FullLoader)

        # Create hardware and resources
        # Create servers
        self.__create_servers(resources_yaml_data)
        # Create computational devices
        self.__create_all_computational_devices(resources_yaml_data)
        # Create Links
        self.__create_all_links(resources_yaml_data)
        # Create network switches
        self.__create_switches(resources_yaml_data)

        # Create logical knowledge of server/hardware/link/switch
        self.__add_all_hardware_to_servers()
        self.__add_all_links_to_hardware()

        # Init routing_info
        self.__init_routing_info()

    def __create_servers(self, resources_yaml_data):
        '''Create servers with computational devices and outbound links

        Note: class Server is stored for future usage
        '''
        servers_metadata = resources_yaml_data["Server"]
        # Create Servers
        for hostname, server_hardware in servers_metadata.items():
            server_name = '/server/'+hostname+'/'
            self.__servers[server_name] = Server(server_name)

    def __create_all_computational_devices(self, resources_yaml_data):
        '''Create all computational devices
        '''
        for hostname, server_hardware in resources_yaml_data["Server"].items():
            for hardware_type, hardware_pool in server_hardware.items():
                for hardware_index, hardware_spec in hardware_pool.items():
                    new_hardware = self.__create_computational_device(
                        hostname, hardware_type,
                        hardware_index, hardware_spec)
                    self.__computational_hardware[new_hardware.get_name()] = \
                        new_hardware

    def __create_computational_device(self, hostname, hardware_type,
                                      hardware_index, hardware_spec):
        '''Create a computational device

        Args:
            hostname: string
            hardware_type: string, indicating the CPU or GPU or etc.
            hardware_index: int
            hardware_spec: a dict containing all essential info
        '''
        hardware_name = "/server/{0}/{1}/{2}/".format(
            hostname, hardware_type, hardware_index)
        if hardware_spec is None:
            return self.valid_computational_hardware_type[
                hardware_type](hardware_name)
        else:
            average_performance = \
                hardware_spec['properties']['average_performance']
            new_hardware = self.valid_computational_hardware_type[
                hardware_type](hardware_name, average_performance)
            return new_hardware

    def __create_all_links(self, resources_yaml_data):
        unique_link_id = 0
        for hostname, server_hardware in resources_yaml_data["Server"].items():
            for hardware_type, hardware_pool in server_hardware.items():
                for hardware_index, hardware_spec in hardware_pool.items():
                    if hardware_spec is not None:
                        for link_info in hardware_spec['links']:
                            # Each hardware could have multiple links
                            src_name = "/server/{0}/{1}/{2}/".format(
                                hostname, hardware_type, hardware_index)
                            new_link = self.__create_link(
                                src_name, link_info, unique_link_id)
                            unique_link_id += 1
                            self.__links.append(new_link)

        for switch_name, switch_spec in resources_yaml_data["Switch"].items():
            full_switch_name = '/switch/{0}/'.format(switch_name)
            for link_info in switch_spec['links']:
                new_link = self.__create_link(
                    full_switch_name, link_info, unique_link_id)
                unique_link_id += 1
                self.__links.append(new_link)

    def __create_link(self, source_hardware_name, link_info, link_id):
        '''Create a Link via link_info

        Args:
            link_info: a dict
            link_id: int
        '''
        valid_link_type = {'RDMA': RDMA, 'PCIe': PCIE, 'PCIE': PCIE}
        return valid_link_type[link_info['type']](
            link_id,
            source_hardware_name, link_info['dest'],
            link_info['rate'], link_info['propagation_latency'],
            link_info['scheduler'])

    def __create_switches(self, resources_yaml_data):
        '''Create switches via resources_yaml_data
        '''
        switches_metadata = resources_yaml_data["Switch"]
        for switch_name, switch_spec in switches_metadata.items():
            full_switch_name = '/switch/{0}/'.format(switch_name)
            self.__switches[full_switch_name] = NetworkSwitchHardware(
                full_switch_name)

    def __add_all_links_to_hardware(self):
        '''Add link to correct hardware, both link and hardware should be
        initialized first
        '''
        for link in self.__links:
            # Add Link to src_hw, the outbound link
            if link.source_hardware in self.__computational_hardware:
                self.__computational_hardware[link.source_hardware].add_link(
                    link)
            elif link.source_hardware in self.__switches:
                self.__switches[link.source_hardware].add_link(link)
            else:
                raise ValueError(
                    "Invalid Link Parameter. src{0}, dest{1}".format(
                        link.source_hardware, link.dest_hardware))

            # Add link to dest_hw, the inbound link
            if link.dest_hardware in self.__computational_hardware:
                self.__computational_hardware[link.dest_hardware].add_link(
                    link)
            elif link.dest_hardware in self.__switches:
                self.__switches[link.dest_hardware].add_link(link)
            else:
                raise ValueError(
                    "Invalid Link Parameter. src{0}, dest{1}".format(
                        link.source_hardware, link.dest_hardware))

    def __add_all_hardware_to_servers(self):
        '''Add hardware to its server, both hardware and server should be
        initialized first

        Note: class Server is stored for future usage
        '''
        for hardware_name, hw_obj in self.__computational_hardware.items():
            # Get server name from current hardware
            hw_name_arr = hardware_name.split('/')
            server_name = '/{0}/{1}/'.format(hw_name_arr[1], hw_name_arr[2])
            self.__servers[server_name].add_hardware(hw_obj)

    def __init_routing_info(self):
        sw_comput_hardware_dict = {
            **self.__computational_hardware,
            **self.__switches}
        # Init router
        self.__router = Router(sw_comput_hardware_dict)

    def get_servers(self):
        return self.__servers

    def get_computational_hardware(self):
        return self.__computational_hardware

    def get_switches(self):
        return self.__switches

    def get_links(self):
        return self.__links

    def get_links_as_list(self):
        '''Return a list of dict, representing links' info
        '''
        link_info = []
        for link in self.__links:
            link_info.append(link.to_dict())
        return link_info

    def get_computational_hardware_as_list(self):
        '''Return [{hardware0_spec}, {hardware1_spec}]
        '''
        hardware_info = []
        for hw in self.__computational_hardware.values():
            hardware_dict = hw.to_dict()
            _, hardware_dict['type'], _, _ = \
                hw.get_computation_hardware_description(hardware_dict['name'])
            hardware_info.append(hardware_dict)
        return hardware_info

    def get_resource_from_name(self, resource_name):
        '''Return the resource whose name is resource_name, return None if not
        found

        Args:
            resource_name: string, denoting the name of resource
        '''
        if resource_name in self.__servers:
            return self.__servers[resource_name]
        elif resource_name in self.__switches:
            return self.__switches[resource_name]
        elif resource_name in self.__computational_hardware:
            return self.__computational_hardware[resource_name]
        return None

    def get_resource_list_from_type(self, resource_type):
        '''Return a list of (resource_type) resource, including computational
        hardware and Server

        Args:
            resource_type: string, denoting the class type. e.g. 'CPUHardware'
        Return:
            a list of type=resource_type resource
        '''
        type_hardware_list = []
        if resource_type not in self.valid_resource_type:
            raise ValueError("Invalid input hardware_type!")
        if resource_type == 'NetworkSwitchHardware':
            type_hardware_list += self.__switches.values()
        elif resource_type == 'Server':
            type_hardware_list += self.__servers.values()
        else:
            for server_name in self.__servers:
                type_hardware_list +=\
                    self.__servers[server_name].get_hardware_list_from_type(
                        resource_type)
        return type_hardware_list

    def get_route_info(self, src_hw_name, dest_hw_name):
        '''Return the routing information from src_hw_name to dest_hw_name,
        return [] if no route found.

        Args:
            src_hw_name: string, source hardware name
            dest_hw_name: string, destination hardware name
        Returns:
            a list of tuple (path: list, path_type: string)
                list: [path0, path1..], each path is a list:[link0, link1]
                string, indicating the route type, used for SuperScalar
        '''
        return self.__router.get_route_info(src_hw_name, dest_hw_name)
