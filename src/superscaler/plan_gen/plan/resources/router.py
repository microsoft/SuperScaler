import copy

from superscaler.plan_gen.plan.resources.hardware import Hardware,\
    ComputationHardware


class Router():
    def __init__(self, hardware_dict):
        '''Init a Router with {hardware_name: hardware_obj} dict

        Args:
            hardware_dict: dict, {hardware_name: hardware_obj}, containing both
                computational and switch hardware
        '''
        if not isinstance(hardware_dict, dict):
            raise ValueError("Input param should be a dict!")
        # Use routing log to store routing info
        self.__routing_log = {}  # {(src_name, dest_name): ([[Link0, ]], type)}
        self.__hardware_dict = {}
        self.update_hardware_dict(hardware_dict)

    def get_route_info(self, src_hw_name, dst_hw_name):
        '''Return the routing information from src_hw_name to dst_hw_name,
        return [] if no route found.

        Args:
            src_hw_name: string, source hardware name
            dst_hw_name: string, destination hardware name
        Returns:
            a list of tuple (path: list, path_type: string)
                list: [path0, path1..], each path is a list:[link0, link1]
                string, indicating the route type, used for SuperScalar
        '''
        if self.__hardware_is_computational(src_hw_name) \
                and self.__hardware_is_computational(dst_hw_name):
            if (src_hw_name, dst_hw_name) not in self.__routing_log:
                result_list = []
                result_list = self.__dfs_generate_route_info(
                    src_hw_name, dst_hw_name, [], 0)
                # Set routing logs of current communication pair
                self.__set_routing_log(src_hw_name, dst_hw_name, result_list)

            return self.__routing_log[(src_hw_name, dst_hw_name)]
        else:
            return []

    def update_hardware_dict(self, hardware_dict):
        '''Use a new hardware_dict, reset Router logs
        '''
        self.__routing_log = {}
        # Check the correctness of hardware_dict
        for hw_name, hw_obj in hardware_dict.items():
            o_links = hw_obj.get_outbound_links()
            if not isinstance(hw_obj, Hardware):
                raise ValueError("Invalid input!")
            else:
                for link_dst_name in o_links:
                    if link_dst_name not in hardware_dict:
                        # Cannot find the dest hardware in Link, fatal error
                        raise ValueError("Invalid input!")
        self.__hardware_dict = copy.deepcopy(hardware_dict)

    def __dfs_generate_route_info(self, src_hw_name, dst_hw_name,
                                  visited_hw_name: list,
                                  comp_hw_num_in_path):
        '''Main function of generating routing info, using DFS algorithm,
        return valid paths, return [] if not found

        Args:
            src_hw_name: string, source hardware name
            dst_hw_name: string, destination hardware name
            visited_hw_name: list of string, hardware that has been visited
            comp_hw_num_in_path: int, the computationhardware in current path

        Return:
            result_list: the result, a list of list [path0, path1..], each path
            is a list:[link0, link1...]

        e.g. a--Link0---b----Link1----c
                         ↘---Link2----c
        Return list is [[Link0, Link1], [Link0, Link2]]
        '''
        src_hw = self.__hardware_dict[src_hw_name]
        if comp_hw_num_in_path > 0:
            # GPU cannot forward traffic, therefore there should not be any
            # computational hardware in one path (beside src and dst)
            return []
        result_list = []
        # Log current hardware
        visited_hw_name.append(src_hw_name)
        o_links = src_hw.get_outbound_links()
        for link_dst_name, links in o_links.items():
            # There may be multiple links between src_hw and dst_hw
            if link_dst_name not in visited_hw_name:
                next_paths = []
                if link_dst_name == dst_hw_name:
                    # Reach destination in this link
                    next_paths = [[]]
                else:
                    # Search deeper
                    next_paths = self.__dfs_generate_route_info(
                        link_dst_name, dst_hw_name, visited_hw_name,
                        comp_hw_num_in_path
                        + self.__hardware_is_computational(link_dst_name)
                    )
                if next_paths:
                    # Found valid paths
                    for link in links:
                        for path in next_paths:
                            result_list.append([link] + path)
        # Remove current hardware log
        visited_hw_name.remove(src_hw_name)
        return result_list

    def __set_routing_log(self, src_hw_name, dst_hw_name, result_list):
        '''Set self.__routing_log[(src_hw_name, dst_hw_name)] = \
            [(path, path_type)]
        '''
        self.__routing_log[(src_hw_name, dst_hw_name)] = []
        for path in result_list:
            # Get path_type, used for SuperScalar
            path_type = self.__get_route_path_type(path)
            # A valid path, add it to routing log
            self.__routing_log[(src_hw_name, dst_hw_name)].append(
                (path, path_type)
            )

    def __hardware_is_computational(self, hw_name):
        '''Check whether the hardware is a ComputationHardware

        Args:
            hw_name: string, the name of Hardware
        '''
        if hw_name not in self.__hardware_dict:
            raise ValueError("Invalid input hw_name!")
        if isinstance(self.__hardware_dict[hw_name], ComputationHardware):
            return True
        else:
            return False

    def __get_route_path_type(self, path):
        '''Get the routing path type: PCIE or RDMA, used for SuperScalar

        Args:
            path: list of Link
        '''
        for link in path:
            if link.link_type == "RDMA":
                return 'RDMA'
        return 'PCIE'
