# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

'''
Implementation of Tensor
'''


class Tensor:
    '''Tensor, created by src node, as output.
    '''
    def __init__(self):
        self._shape = None
        self._element_type = None
        self._partition_strategy = None

    @property
    def shape(self):
        '''for output shape inference
        shape is a list of int, for example: [16, 224, 224, 3]
        '''
        return self._shape

    @property
    def element_type(self):
        '''for type checking between input_tensor and output_tensor.
        '''
        return self._element_type

    @property
    def partition_strategy(self):
        '''partition_strategy for each tensor
        Each tensor has a strategy which is a list of tuple for respective
        input_tensors of this node.
        Each tuple contains partition pivots ranging from 0 to 1.
        Strategy is None before setting parallelisms to its op.
        TODO(gbxu): design for existing parallelisms and extensibility.
        '''
        return self._partition_strategy
