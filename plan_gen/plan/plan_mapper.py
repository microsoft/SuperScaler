import abc
import copy
from resources.resource_pool import ResourcePool


class PlanMapper(abc.ABC):
    """ An mapper class maps nodes on actual devices
    """
    def __init__(self, resource_pool):

        if not isinstance(resource_pool, ResourcePool):
            raise ValueError(
                "Input resource_pool must be ResourcePool instance")
        self.__resource_pool = resource_pool

    @property
    def resource_pool(self):
        return self.__resource_pool

    @abc.abstractmethod
    def map(self, plan):
        return None


class GPURoundRobinMapper(PlanMapper):
    """ Assign device as GPURoundRobin
    """
    def __init__(self, resource_pool):
        super().__init__(resource_pool)
        self.gpus = self.resource_pool.get_resource_list_from_type("GPU")

    def map(self, plan):
        if not isinstance(plan, list):
            return None
        else:
            mapped_plan = copy.deepcopy(plan)
            if not self.__assign_device(mapped_plan):
                return None
            else:
                return mapped_plan

    def __assign_device(self, plan):
        ''' This function assigns the virtual devices of plan
            as the real devices of resource_pool
        '''
        # Record all devices of plan
        devices = []

        for node in plan:
            if 'device' in node and node['device'] not in devices:
                devices.append(node['device'])

        # Check whether the plan can be assigned into resource_pool
        if len(self.gpus) < 1:
            # Resource Pool is empty
            return False
        if len(self.gpus) < len(devices):
            # GPU count in resource_pool can't meet the requirement of plan
            return False

        # Assign devices by RoundRobin order
        for node in plan:
            if 'device' in node:
                gpu = self.gpus[devices.index(node['device'])]
                node['device'] = gpu.get_name()
            if 'target' in node:
                gpu = self.gpus[devices.index(node['target'])]
                node['target'] = gpu.get_name()

        return True
