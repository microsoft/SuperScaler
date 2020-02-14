#!/usr/bin/env python3

'''
SuperScalar Graph Visitor
'''

import abc
from collections import namedtuple


class SuperScalarGraphVisitor(abc.ABC):
    def __init__(self, graph):
        self.graph = graph

    @abc.abstractmethod
    def __iter__(self):
        raise NotImplementedError("Need implement the function for " + str(self.graph))


class JsonVisitor(SuperScalarGraphVisitor):
    def __iter__(self):
        for op in self.graph:
            OperationObject = namedtuple("OperationObject", op.keys())
            yield OperationObject(**op)
        

