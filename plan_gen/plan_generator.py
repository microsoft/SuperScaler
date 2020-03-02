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

    def get_plan(self, superscaler_graphs = {}):
        graphs = superscaler_graphs.items()
        # Import graph to device assigner
        for _, graph in graphs:
            self.device_assigner.add_graph(graph)

        # Assign device for each graph
        for graph_id, graph in graphs:
            device = self.device_assigner.assign_device(graph)
            for op in self.graph_visitor(graph):
                self.op_manager.add_op(op, str(graph_id), device)

        self.op_manager.schedule()

        plan = {}
        for op in self.op_manager.get_op_iterator():
            device_id = op.device.get_id()
            tensor = op.tensor
            if device_id not in plan:
                plan[device_id] = {}
            if op.graph_id not in plan[device_id]:
                plan[device_id][op.graph_id] = {}
            if tensor in plan[device_id][op.graph_id]:
                raise ValueError("Tensor %s has been assign in %s" % (tensor, plan[device_id]))
            plan[device_id][op.graph_id][tensor] = op.dump()
        return plan
