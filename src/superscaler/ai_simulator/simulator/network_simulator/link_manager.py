# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from superscaler.ai_simulator.simulator.network_simulator.link import Link


class LinkManager():
    link_essential_data = {'link_id': int, 'source_name': str,
                           'dest_name': str}
    link_extra_data = {'capacity': str, 'latency': str}

    def __init__(self, links_spec, routing_info_dict):
        '''Reconstruct routing information using links_with_id and routing_info,
        check the validity of input params

        Args:
            links_spec: list of dict, [{'link_id': int, 'source_name': str,
                            'dest_name': str, 'capacity': str,
                            'latency': str}]
                src/dest_name format:
                    /server/hostname/DeviceType/DeviceIndex/
                    /switch/switch_name/
            routing_info_dict: {(src_name, dst_name, route_index):[id0, id1..]}
        '''
        # A dict {link_id: Link obj} containing all links
        self.__links_dict = {}
        # Store routing path:
        # {(src_name, dst_name, route_index):[Link0, Link1..]}
        self.__routing_path = {}
        # Init all Links
        self.__init_links(links_spec)
        # Init routing info for easier access
        self.__init_routing_path(routing_info_dict)

    def get_routing_path(self, src_name, dst_name, route_index):
        '''Return a list containing the sequence of Link in this path,
        return None if not found

        Args:
            src_name: string, the source name of the device
            dst_name: string, the destination name of the device
            route_index: int, the routing index
        '''
        if (src_name, dst_name, route_index) not in self.__routing_path:
            return None
        return self.__routing_path[(src_name, dst_name, route_index)]

    def get_routing(self, node_name):
        '''Parse the node_name string and return routing path

        Args:
            node_name: string, format:
                ":send:src_name:dst_name:route_index:"
        '''
        # Check the validity of node_name
        name_split = node_name.split(':')
        if len(name_split) < 6:
            return None
        _, src_name, dst_name, route_index, *__ = name_split[1:]
        route_index = int(route_index)
        return self.get_routing_path(src_name, dst_name, route_index)

    def get_link(self, link_id):
        '''Return a Link with specific link_id
        '''
        if link_id not in self.__links_dict:
            return None
        return self.__links_dict[link_id]

    def get_links_dict(self):
        '''Return a dict {link_id: Link obj} containing all links
        '''
        return self.__links_dict

    def __init_links(self, links_spec):
        '''Check validity of links_spec, and init self.__links using links_spec
        '''
        # Check link_spec_dict
        if not isinstance(links_spec, list):
            raise ValueError("Input links_spec should be a list!")
        for link_info in links_spec:
            arg_dict = {}
            link_info_valid = True
            for key, ess_type in self.link_essential_data.items():
                if key in link_info and isinstance(link_info[key], ess_type):
                    arg_dict[key] = link_info[key]
                else:
                    # link_info does not contain all necessary information
                    link_info_valid = False
                    break
            if not link_info_valid:
                # current links_info is incorrect, ignore this link
                continue
            for key, extra_type in self.link_extra_data.items():
                if key in link_info and isinstance(link_info[key], extra_type):
                    arg_dict[key] = link_info[key]
            new_link = Link(**arg_dict)
            if link_info['link_id'] in self.__links_dict:
                raise ValueError("Invalid links_spec: duplicated link_id")
            else:
                self.__links_dict[link_info['link_id']] = new_link

    def __init_routing_path(self, routing_info_dict):
        '''Init self.__routing_path via routing_info_dict
        '''
        # Check the validity of input dict
        if not isinstance(routing_info_dict, dict):
            raise ValueError("Invalid routing_info_dict parameter!")
        for comm_info, route_path in routing_info_dict.items():
            # Check the validity of communication pair tuple
            # (src_name, dst_name, route_index)
            if len(comm_info) != 3:
                raise ValueError("Invalid routing_info_dict parameter!")
            current_route_path_links = []
            route_path_valid = True
            prev_src_name = comm_info[0]
            for link_id in route_path:
                if link_id not in self.__links_dict:
                    # No this link_id, ignore this comm_info tuple
                    route_path_valid = False
                    break
                current_link = self.__links_dict[link_id]
                if current_link.source_name != prev_src_name:
                    route_path_valid = False
                    break
                prev_src_name = current_link.dest_name
                current_route_path_links.append(current_link)

            if route_path_valid and prev_src_name == comm_info[1]:
                self.__routing_path[comm_info] = current_route_path_links
