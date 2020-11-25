# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from superscaler.ai_simulator.simulator.computation_device import CPU, GPU
from superscaler.ai_simulator.simulator.network_simulator.network_simulator \
    import NetworkSimulator


class DeviceFactory():
    # Use a dict to store valid device_type string instead of using
    # dangerous eval() function
    valid_device_type = {'CPU': CPU, 'GPU': GPU,
                         'NetworkSimulator': NetworkSimulator}

    def __init__(self):
        pass

    def generate_device(self, device_type, *spec_list):
        '''Return a (device_type) class obj based on spec_list

        Args:
            device_type: string, denoting the type of this device
            spec_list: list, the parameters to initialize the class
        '''

        if device_type not in self.valid_device_type:
            raise ValueError("device_type in device_info is invalid")
        device = self.valid_device_type[device_type](*spec_list)
        return device
