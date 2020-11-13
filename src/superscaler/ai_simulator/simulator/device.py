class Device():
    def __init__(self, name):
        # Name of device
        self.__name = name
        # The finish time of current node.
        self.__next_finish_time = 0.0

    # Get the device name
    def name(self):
        return self.__name

    # Check whether the device is idle
    def is_idle(self):
        return True

    # Get the first completed node
    def get_next_node(self):
        return None, self.__next_finish_time

    # Enqueue a new node into this device
    def enqueue_node(self, node, time_now):
        return

    # Dequeue the first completed node from the device.
    # Do not modify the attribute of the node, just modify info of device.
    def dequeue_node(self):
        return
