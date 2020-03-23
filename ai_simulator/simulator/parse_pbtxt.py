'''
Author: v-hohua
Version: 0.1
Update date: 12/16/2019

Temporary parser module of AI Simulator project.
This module will read the .pbtxt model file and generate the .json format 
parsed DAG file. Attributes that can directly got from .pbtxt file will be filled
into .json file in this module.

v0.1 supported attributes:
Attribute name      Type            Description
index               Int             ID of node
op                  String          Operation
name                String          Name of node
enabled             Bool            Whether need to execute.
                                    In this module, set all to True.
raw_inputs          List of str     Raw input list in .pbtxt file.
input_ids           List of int     Node ID of dataflow input
dependency_ids      List of int     Node ID of dependency input
successor_ids       List of int     Node ID of successors

Debug attributes: These attributes is used in debugging work.
out_degree          Int             Number of nodes has data flow from it.

Future support attributes:
data_type           String          This node's data type
device              String          Device to run this node
op_attributes       List of kwarg   Special attributes belong to OP      
'''

import tensorflow as tf
from google.protobuf import text_format
import json

class TFNode():
    def __init__(self, index = 0, 
                op = '',
                name = '', 
                device = 'gpu:0', 
                execution_time = 0.0
                ):
        #==============================
        # Attributes of node
        #==============================
        
        # Int. The ID of node.
        self.index = index
        # String. The operation of node.
        self.op = op
        # String. The name of node.
        self.name = name

        # String, the device where the node is assigned.
        self.device = device

        self.execution_time = 0.0

        #==============================
        # Attributes of edge
        #==============================
        # These attributes is initialized by adapter outside this function.

        # List of NodeMeta ref. Node of dataflow inputs
        self.input_ids = []
        # List of int. Read which one of the input node's output. 
        # Current version does not use this attribute
        # In most cases, this attribute is 0. Read the first output of given node.
        # self.input_data_ids = []
        # List of NodeMeta ref. Node of dependency inputs
        self.dependency_ids = []
        # List of NodeMeta ref. Node of successor nodes depends on this node.
        self.successor_ids = []

        #==============================
        # Attributes for debugging and testing
        #==============================
        # List of string. The raw input list in .pbtxt file.
        # The raw input name contains ':1' to indicate which output of a node is 
        # the input.
        # self.raw_inputs = []
        

def load_pbtxt_file(input_filename):
    with open(input_filename) as f:
        txt = f.read()
    graph_def = text_format.Parse(txt, tf.GraphDef())
    return graph_def

def parse_protobuf_graph(graph_def):
    # The list of all nodes in a graph.
    node_list = []
    # Name_list is used to inquire node id by name.
    name_list = []
    # The counter of nodes.
    node_idx = 0

    # Read all nodes, assign their node attributes
    for node in graph_def.node:
        name = str(node.name)
        inputs = list(node.input)
        op = str(node.op)
        
        new_node = TFNode()
        new_node.index = node_idx
        new_node.name = name
        new_node.op = op
        new_node.raw_inputs = inputs
        
        node_idx+=1
        node_list.append(new_node)
        name_list.append(name)

    # Analyze edges and assign edge attributes.
    for node in node_list:
        for input_name in node.raw_inputs:
            input_idx = name_list.index(input_name) if (input_name in name_list) else -1
            # These logic is same as TensorFlow source code
            # https://github.com/tensorflow/tensorflow/blob/r1.13/tensorflow/python/framework/importer.py
            # '^op_name' means control input
            # 'op_name:output_index' means get the id-th output of an op
            if input_name.startswith('^'):
                real_input_name = input_name[1:]
                input_idx = name_list.index(real_input_name) if \
                    (real_input_name in name_list) else -1
            if ':' in input_name:
                components = input_name.split(':')
                if len(components) == 2:
                    real_input_name = components[0]
                    input_idx = name_list.index(real_input_name) if \
                        (real_input_name in name_list) else -1
                else:
                    raise ValueError('Cannot convert %r to a tensor name.' 
                                % (input_name))
            if input_idx >= 0:
                node_list[input_idx].successor_ids.append(node.index)
                if input_name.startswith('^'):
                    node.dependency_ids.append(input_idx)
                else:
                    node.input_ids.append(input_idx)
            else:
                print('[ERROR] Input tensor of a node not found in list!')
                print('Node name: ', node['name'])
                print('Error input: ', input_name)
                print('[HANDLE] Discard this input tensor')
    return node_list
