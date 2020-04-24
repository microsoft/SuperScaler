from .fairsharing_device import FairSharingDevice
from .device import Device


class NetworkSwitch(Device):
    INVALID_DEST = "INVALID_DEST"

    def __init__(self, name, links):
        '''Init a network switch with len(links) ports

        Args:
            name: string, the name of the switch
            links: a list of tuple (dest: string, rate: string/float).
                   dest is the name of the target device,
                   and rate denotes transmit rate in string, or float in bps
                   e.g. [('/server/hostname1/CPU/0', '16bps'),
                         ('/server/hostname1/GPU/0','32bps')]
        '''
        super().__init__(name)
        self.__port_num = len(links)
        self.__links = links
        self.__fairsharing_devices = {}  # a dict, key: device_name, val: rate
        for i in range(self.__port_num):
            if links[i][0] not in self.__fairsharing_devices:
                self.__fairsharing_devices[links[i][0]] \
                    = FairSharingDevice(name, links[i][1])
            else:
                raise ValueError("Invalid parameter: device_name duplicated")

    def get_port_num(self):
        '''Return a int denotes the port num'''
        return self.__port_num

    def get_links(self):
        '''Return links in a list of tuple'''
        return self.__links

    def is_idle(self):
        '''If all FS_Device is idle, then the switch is idle'''
        for dest_name, fs_device in self.__fairsharing_devices.items():
            if not fs_device.is_idle():
                return False
        return True

    def get_next_node(self):
        '''Return (next finish node, estimated finish time) in this switch'''
        _, next_node, estimated_finish_time = self.__get_next_finish()
        return (next_node, estimated_finish_time)

    def enqueue_node(self, node, time_now):
        '''Parse the op of the node, enqueue the node to specific port

        op example: ":send:outbound_dest:...."
        e.g. ":send:/server/hostname1/GPU/0:"
        '''
        dest, extra_info = NetworkSwitch.__get_outbound_dest(node)
        if dest in self.__fairsharing_devices:
            self.__fairsharing_devices[dest].enqueue_node(node, time_now)
        else:
            raise ValueError("NetworkSwitch isn't connected to device named "
                             + str(dest))

    def dequeue_node(self):
        '''Dequeue the node with minimum estimated finish time

        First, find the next port that will finish transmition, then dequeue
        node in this port
        '''
        dest_name, next_node, estimated_finish_time = self.__get_next_finish()
        if dest_name != NetworkSwitch.INVALID_DEST:
            self.__fairsharing_devices[dest_name].dequeue_node()

    def __get_next_finish(self):
        '''Find the next node that will finish transmit in this switch,
        return (dest, next finish node, estimated finish time)
        '''
        latest_dest = NetworkSwitch.INVALID_DEST
        latest_node = None
        latest_time = float('inf')
        if self.is_idle():
            return (latest_dest, latest_node, latest_time)
        for dest_name in self.__fairsharing_devices:
            if not self.__fairsharing_devices[dest_name].is_idle():
                next_node, next_time =\
                    self.__fairsharing_devices[dest_name].get_next_node()
                if next_time < latest_time:
                    latest_dest = dest_name
                    latest_node = next_node
                    latest_time = next_time
        return (latest_dest, latest_node, latest_time)

    @staticmethod
    def __get_outbound_dest(node):
        '''Parse the op of the node, return outbound_dest and extra_info

        op example: ":send:outbound_dest:...."
        e.g. ":send:/server/hostname1/GPU/0:"
        '''
        if 'send' not in node.get_op():
            raise ValueError("node does not have a valid outbound_port")
        _, dest, extra_info = node.get_op().split(':')[1:]
        return (dest, extra_info)
