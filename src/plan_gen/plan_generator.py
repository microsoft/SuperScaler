#!/usr/bin/env python3

import device_assigner
import operator_assigner


class PlanGenerator(object):
    def __init__(
        self,        
        graph_visitor,
        resource_pool,
        device_assigner_class = device_assigner.GPURoundRobin):
        self.graph_visitor = graph_visitor
        self.resource_pool = resource_pool
        self.device_assigner = device_assigner_class(self.resource_pool)
        self.op_manager = operator_assigner.OperatorManager()

    def get_plan(self, superscaler_graphs):
        # Import graph to device assigner
        for graph in superscaler_graphs:
            self.device_assigner.add_graph(graph)

        # Assign device for each graph
        for graph in superscaler_graphs:
            device = self.device_assigner.assign_device(graph)
            for op in self.graph_visitor(graph):
                self.op_manager.add_op(op, device)

        self.op_manager.schedule()

        plan = {}
        for op in self.op_manager.get_op_iterator():
            device_id = op.device.get_id()
            tensor = op.tensor
            if device_id not in plan:
                plan[device_id] = {}
            if tensor in plan[device_id]:
                raise ValueError("Tensor %s has been assign in %s" % (tensor, plan[device_id]))
            plan[device_id][tensor] = op.dump()
        return plan
