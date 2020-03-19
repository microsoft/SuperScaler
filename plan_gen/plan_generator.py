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

    def get_plan(self, running_graphs, init_graphs = None):
        """ Generate plan for running graphs and init graphs
        get_plan(running_graphs, init_graphs = None) -> (running_plan, init_plan)
        Keyword arguments:
            (running_graphs, init_graphs)
            running_graphs -- running graphs (required)
            init_graphs    -- init graphs (optional)
        Return Value:
            (running_plan, init_plan)
            running_plan -- running plan for running graph
            init_plan    -- init plan for init graph
        """
        if init_graphs is None:
            init_graphs = { key: None for key in running_graphs.keys() }
            
        if sorted(running_graphs.keys()) != sorted(init_graphs.keys()):
            raise ValueError("Running graphs isn't consistent with initial graphs")
        running_graphs = sorted(running_graphs.items())
        init_graphs = sorted(init_graphs.items())
        # Import graph to device assigner
        for i in range(len(running_graphs)):
            self.device_assigner.add_graph(running_graphs[i][1], init_graphs[i][1])
        return self._get_plan(running_graphs), self._get_plan(init_graphs)

    def _get_plan(self, graphs):
        op_manager = operator_assigner.OperatorManager()
        # Assign device for each graph
        for graph_id, graph in graphs:
            device = self.device_assigner.assign_device(graph)
            if device is None:
                continue
            for op in self.graph_visitor(graph):
                op_manager.add_op(op, str(graph_id), device)
        op_manager.schedule()
        plan = {}
        for op in op_manager.get_op_iterator():
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
       

