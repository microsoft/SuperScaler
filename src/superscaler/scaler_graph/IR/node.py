# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

'''
Implementation of Node, CompositeNode
'''
from superscaler.scaler_graph.IR.tensor import Tensor
from superscaler.scaler_graph.IR.util import serialization
from superscaler.scaler_graph.util.log import logger
import json


class Node:
    '''Node
    '''
    _ID = 0

    def __init__(self, node_name, op, input_node_idxes, output_size, attrs):
        '''
        initilize node and its output tensors, set its input tensors
        dependencies.
        Args:
            op_name: node type
            input_node_idxes: input nodes and respective output index
            output_size: output tensors of this node
            attrs: attributions
        Returns:
            None
        '''
        self.id = self._ID
        self.__class__._ID += 1
        self.name = node_name
        self.op = op
        self.attrs = attrs
        # Graph will create Edges for each node.
        self.in_edges = []  # just reference
        self.out_edges = []  # just reference

        self._input_tensors = []  # reference to input_node's output tensors
        self._output_tensors = []

        for input_node_idx in input_node_idxes:
            (input_node, idx) = input_node_idx
            output_tensor = input_node.get_output_tensor(idx)
            self._input_tensors.append(
                output_tensor)  # we should keep them ordered

        self._output_tensors = self._create_output_tensors(output_size)

    def _create_output_tensors(self, output_size):
        tensors = []
        for i in range(output_size):
            # TODO(gbxu): create Tensors
            tensors.append(Tensor())
        return tensors

    def get_output_tensor(self, idx):
        if idx == -1:
            return None
        return self._output_tensors[idx]

    def add_in_edge(self, edge, slot_idx=None):
        if slot_idx is not None:
            assert (self.in_edges[slot_idx] is None
                    and self._input_tensors[slot_idx] is None)
            self.in_edges[slot_idx] = edge
            if edge.src_idx != -1:
                output_tensor = edge.src_node.get_output_tensor(edge.src_idx)
                self._input_tensors[slot_idx] = output_tensor
        else:
            self.in_edges.append(edge)
            if edge.src_idx != -1:
                self._input_tensors.append(
                    edge.src_node.get_output_tensor(edge.src_idx))

    def remove_in_edge(self, edge):
        index = self.in_edges.index(edge)
        self.in_edges[index] = None
        self._input_tensors[index] = None

    def add_out_edge(self, edge):
        self.out_edges.append(edge)

    def remove_out_edge(self, edge):
        self.out_edges.remove(edge)

    def infer_shape(self):
        self.op.infer_shape(self)

    def dict(self):
        in_edges = []
        for in_edge in self.in_edges:
            if in_edge is None:
                in_edge_str = ""
            elif in_edge.src_idx == -1:
                in_edge_str = f"^{in_edge.src_node.name}"
            elif in_edge.src_idx == 0:
                in_edge_str = f"{in_edge.src_node.name}"
            else:
                in_edge_str = f"{in_edge.src_node.name}:{in_edge.src_idx}"
            in_edges.append(in_edge_str)
        return dict(name=self.name,
                    op=self.op.name,
                    original_op=self.op.original_name,
                    in_edges=in_edges,
                    output_size=len(self._output_tensors),
                    attrs=dict(self.attrs))

    def json(self):
        '''serialize node
        '''
        return json.dumps(self.dict(),
                          indent=4,
                          cls=serialization.AttrEnconding,
                          sort_keys=True)


class CompositeNode(Node):
    '''CompositeNode contains a few nodes.
    '''
    def __init__(self, nodes):
        # TODO(gbxu): define CompositeNode
        logger().error("We can't support CompositeNode now.")
        raise RuntimeError
