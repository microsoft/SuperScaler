'''
Operator Assigner
'''


import abc
import resource


class Operator(object):
    def __init__(self, op, graph_id, device):
        self.op = op
        self.graph_id = graph_id
        self.device = device
        self.name = op.name
        self.tensor = op.tensor
        self.type = None
        self.endpoints = []

    @abc.abstractmethod
    def assign(self, peer_operators):
        pass

    def dump(self):
        return {
            "name" : self.name,
            "type" : self.type,
            "tensor" : self.tensor,
            "endpoints" : self.endpoints
        }
    

class Operator2P2Communication(Operator):
    def __init__(self, op, graph_id, device):
        super().__init__(op, graph_id, device)
        self.type = "PCIE"

    def assign(self, peer_operators):
        if len(peer_operators) != 2 or self not in peer_operators:
            raise ValueError("%s is not met requirement" % (peer_operators,))
        peer_operator = peer_operators[1] if peer_operators[0] is self else peer_operators[0]
        if self.device.host == peer_operator.device.host:
            self.type = "PCIE"
        else:
            self.type = "RDMA"
        if len(self.endpoints) > 0:
            raise ValueError("This operator has been assigned.")
        self.endpoints.append(peer_operator.device.get_id())


class OperatorReceive(Operator2P2Communication):
    pass


class OperatorSend(Operator2P2Communication):
    pass


class OperatorAllReduce(Operator):
    def __init__(self, op, graph_id, device):
        super().__init__(op, graph_id, device)
        self.type = "DEFAULT"

    def assign(self, peer_operators):
        if self not in peer_operators:
            raise ValueError("%s is not met requirement" % (peer_operators,))
        if len(self.endpoints) > 0:
            raise ValueError("This operator has been assigned.")
        for op in peer_operators:
            self.endpoints.append(op.device.get_id())
        self.endpoints.sort()


class OperatorManager(object):
    operator_factory = {
        "_screcv": OperatorReceive,
        "_scsend": OperatorSend,
        "_scallreduce" : OperatorAllReduce,
    }

    def __init__(self):
        self.op_groups = {}
        self.assigned_ops = set()

    def add_op(self, op, graph_id, device):
        op_name = op.name.lower()
        if op_name not in OperatorManager.operator_factory:
            return
        
        # op.tensor is the unique key to identify a group of op
        # Group these ops according to the tensor name
        if op.tensor not in self.op_groups:
            self.op_groups[op.tensor] = []
        self.op_groups[op.tensor].append(
            OperatorManager.operator_factory[op_name](op, graph_id, device)
        )

    def get_op_iterator(self):
        # def _iterator(self):
        for op_group in self.op_groups.values():
            for op in op_group:
                yield op

    def schedule(self):
        for op_group in self.op_groups.values():
            for op in op_group:
                if op in self.assigned_ops:
                    continue
                op.assign(op_group)
                self.assigned_ops.add(op)

