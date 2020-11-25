# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

'''
TODO(gbxu): need more operators
'''
from abc import abstractmethod


class Operator:
    def __init__(self, original_name=None):
        self._original_name = original_name
        self.info = {}

    @property
    def name(self):
        return self.__class__.__name__

    @property
    def original_name(self):
        return self._original_name

    @original_name.setter
    def original_name(self, value):
        assert (self._original_name is None)
        self._original_name = value

    @abstractmethod
    def infer_shape(self, node):
        pass

    @abstractmethod
    def partition(self, node):
        '''partition strategy design
        '''
        pass


class ElementWiseOp(Operator):
    def infer_shape(self, node):
        node._output_tensors[0].element_type = node._input_tensors[
            0].element_type
        node._output_tensors[0].shape = node._input_tensors[0].shape()


class BinaryElementWiseOp(Operator):
    def infer_shape(self, node):
        shape_0 = node._input_tensors[0].shape()
        shape_1 = node._input_tensors[1].shape()
        output_shape = []
        for i in range(len(shape_0)):
            if shape_0[i] != shape_1[i]:
                assert (shape_0[i] == 1 or shape_1[i] == 1)
            output_shape.append(max(shape_0[i], shape_1[i]))
        node._output_tensors[0].element_type = node._input_tensors[
            0].element_type
        node._output_tensors[0].shape = output_shape

    def partition(self, node):
        '''
        strategy examples for 2 devices:
        strategy:
            [
                [(0, 0.5, 1), (0, 1)],
                [(0, 0.5, 1), (0, 1)]
            ]
        '''
        # input_tensors_strategy_0 = node._input_tensors[0].partition_strategy
        # input_tensors_strategy_1 = node._input_tensors[1].partition_strategy
        # for i in range(len(input_tensors_strategy_0)):
        #     assert (input_tensors_strategy_0[i] ==
        #             input_tensors_strategy_1[i])
        # node._output_tensors[0].partition_strategy = []
        # node._output_tensors[0].partition_strategy.append(
        #     input_tensors_strategy_0)
        # TODO(gbxu)
        raise Exception


class ApplyOp(Operator):
    def infer_shape(self, node):
        # TODO(gbxu): check input shapes
        parameter_shape = node._input_tensors[
            self.info["parameter_index"]].shape()
        output_shape = []
        for i in range(len(parameter_shape)):
            output_shape.append(parameter_shape[i])
        node._output_tensors[0].shape = output_shape

    def partition(self, node):
        pass


class GlobalInfoOp(Operator):
    def infer_shape(self, node):
        pass

    def partition(self, node):
        pass


class NoOp(Operator):
    def infer_shape(self, node):
        return

    def partition(self, node):
        return


class AllreduceOp(Operator):
    def infer_shape(self, node):
        return

    def partition(self, node):
        return
