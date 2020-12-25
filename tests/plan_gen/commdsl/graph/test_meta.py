# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from superscaler.plan_gen.commdsl.graph.meta import TransNodeType
from superscaler.plan_gen.commdsl.graph.meta import CompNodeType
import pytest


def test_trans_node_type():
    assert TransNodeType.SEND.value == 'send'
    assert TransNodeType.RECV.value == 'recv'
    with pytest.raises(AttributeError):
        _ = TransNodeType.BROADCAST


def test_comp_node_type():
    assert CompNodeType.ADD.value == 'add'
    assert CompNodeType.SUB.value == 'sub'
    assert CompNodeType.MUL.value == 'mul'
    assert CompNodeType.DIV.value == 'div'
    assert CompNodeType.COPY.value == 'copy'
    assert CompNodeType.CREATE.value == 'create'
    with pytest.raises(AttributeError):
        _ = CompNodeType.MATMUL
