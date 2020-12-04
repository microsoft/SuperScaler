# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from superscaler.plan_gen.commdsl.graph.segment import DataSegment
from superscaler.plan_gen.commdsl.graph.segment_pool import DataSegmentPool
from superscaler.plan_gen.commdsl.errors import CommDSLRuntimeError


def test_segment_init():
    data_root = DataSegment(4)
    # test root DataSegment
    assert data_root.bnum == 4
    assert data_root.cnum == 1
    assert data_root.root_bnum == 4
    assert data_root.root_blk_begin == 0
    assert data_root.root_blk_end == 4
    assert isinstance(data_root._id, int)
    # test part DataSegment
    data_root.cnum = 4
    data_part = DataSegment(1, data_root, 2)
    assert data_part.name == data_root.name
    assert data_part.bnum == 1
    assert data_part.cnum == 1
    assert data_part.root_bnum == 4
    assert data_part.root_blk_begin == 2
    assert data_part.root_blk_end == 3
    assert data_root._id == data_part._id
    # test fail cases
    try:
        _ = DataSegment([])
        assert False
    except TypeError:
        pass
    try:
        _ = DataSegment(-1)
        assert False
    except CommDSLRuntimeError:
        pass


def test_segment_bnum():
    data_root = DataSegment(4)
    data_root.cnum = 4
    data_part = DataSegment(1, data_root, 2)
    assert data_part._bnum == 1
    assert data_part.cnum == 1
    assert data_part.root_bnum == 4
    assert data_part.root_blk_begin == 2
    assert data_part.root_blk_end == 3
    # test bnum()
    data_part.bnum = 4
    assert data_part._bnum == 4
    assert data_part.cnum == 1
    assert data_part.root_bnum == 16
    assert data_part.root_blk_begin == 8
    assert data_part.root_blk_end == 12
    # test fail case
    try:
        data_part.bnum = 2
        assert False
    except CommDSLRuntimeError:
        pass
    try:
        data_part.bnum = 5
        assert False
    except CommDSLRuntimeError:
        pass


def test_segment_id():
    DataSegmentPool().clear()
    data_seg1 = DataSegment(1)
    assert data_seg1._id == 1
    data_seg2 = DataSegment(2)
    assert data_seg2._id == 2


def test_segment_slice():
    # test for slice, __getitem__
    data_root = DataSegment(1)
    data_root.slice(4)
    assert data_root.bnum == 4
    assert data_root.cnum == 4
    for idx in range(data_root.cnum):
        data_block = data_root[idx]
        assert data_block.bnum == 1
        assert data_block.cnum == 1
        assert data_block.root_bnum == 4
        assert data_block.root_blk_begin == idx
        assert data_block.root_blk_end == idx + 1
    data_4b1 = data_root[1]
    data_4b1.slice(3)
    assert data_root.bnum == 12
    assert data_4b1.bnum == 3
    assert data_4b1.cnum == 3
    assert data_4b1.root_blk_begin == 3
    assert data_4b1.root_blk_end == 6
    assert data_root.cnum == 4
    assert data_root.bnum == 12
    data_root.slice(2)
    assert data_root.cnum == 2
    assert data_root.bnum == 12
    assert data_4b1.bnum == 3
    assert data_4b1.cnum == 3
    assert data_4b1.root_blk_begin == idx
    assert data_4b1.root_blk_end == idx + 3


def test_segment_scale():
    seg = DataSegment(1)
    seg.scale(8)
    assert seg.bnum == 8
    assert seg.cnum == 1
    assert seg.root_bnum == 8
    assert seg.root_blk_begin == 0
    assert seg.root_blk_end == 8
    try:
        seg.scale(1.5)
        assert False
    except TypeError:
        pass
    try:
        seg.scale(-2)
        assert False
    except CommDSLRuntimeError:
        pass


def test_segment_get_item():
    data_root = DataSegment(1)
    data_root.slice(4)
    _ = data_root[1]
    # int IndexError case
    try:
        _ = data_root[4]
        assert False
    except IndexError:
        pass
    # slice case
    data_4b23 = data_root[-2:4]
    assert data_4b23.bnum == 2
    assert data_4b23.cnum == 1
    assert data_4b23.root_blk_begin == 2
    assert data_4b23.root_blk_end == 4
    # slice IndexError case
    # slice in reverse order
    try:
        _ = data_root[3:2]
        assert False
    except IndexError:
        pass
    # slice out of boundary
    try:
        _ = data_root[-5:-1]
        assert False
    except IndexError:
        pass
    # slice has step
    try:
        _ = data_root[2:4:2]
        assert False
    except IndexError:
        pass
    # other types
    try:
        _ = data_root[[1, 2]]
        assert False
    except TypeError:
        pass


def test_segment_overlap():
    # data representation (root-bnum)[root-start:root-end)
    DataSegmentPool().clear()
    data_root1 = DataSegment(1)
    data_root2 = DataSegment(2)
    data_root1.slice(4)            # (4)[0:4)
    data_r1_4b0 = data_root1[0]    # (4)[0:1)
    data_r1_4b1 = data_root1[1]    # (4)[1:2)
    data_r1_4b3 = data_root1[3]    # (4)[3:4)
    # (4)[0:1) & (4)[1:2)
    assert data_r1_4b0.overlap(data_r1_4b1) is False
    # (4)[1:2) & (4)[0:1)
    assert data_r1_4b1.overlap(data_r1_4b0) is False
    # contain: (4)[0:1) & (4)[0:4)
    assert data_r1_4b0.overlap(data_root1) is True
    data_root1.slice(2)            # (4)[0:4)
    data_r1_2b0 = data_root1[0]    # (4)[0:2)
    # contain: (4)[1:2) & (4)[0:2)
    assert data_r1_4b1.overlap(data_r1_2b0) is True
    # (4)[3:4) & (4)[0:2)
    assert data_r1_4b3.overlap(data_r1_2b0) is False
    # different id
    assert data_root2.overlap(data_r1_4b0) is False
    data_root2.slice(3)            # (24)[0:24)
    data_r2_3b2 = data_root2[2]    # (24)[16:24)
    # contain: (24)[16:24) & (24)[0:24)
    data_r2_3b2.overlap(data_root2) is True
    # different id
    data_r2_3b2.overlap(data_r1_4b1) is False


def test_segment_eq():
    data_root = DataSegment(1)
    data_root.slice(3)
    data_r_3b1 = data_root[1]
    data_root.slice(4)
    data_root.slice(3)
    assert data_r_3b1 == data_root[1]
    try:
        data_r_3b1 == 2
        assert False
    except TypeError:
        pass


def test_segment_repr():
    DataSegmentPool().clear()
    data = DataSegment(1)
    data.slice(4)
    info = data.__repr__()
    assert info == 'Data-1:(4)[0:4]'


def test_segment_operator():
    lhs = DataSegment(1)
    rhs = DataSegment(2)
    # __add__
    try:
        _ = lhs + rhs
        assert False
    except CommDSLRuntimeError:
        pass
    # __iadd__
    try:
        lhs += rhs
        assert False
    except CommDSLRuntimeError:
        pass
    # __sub__
    try:
        _ = lhs - rhs
        assert False
    except CommDSLRuntimeError:
        pass
    # __isub__
    try:
        lhs -= rhs
        assert False
    except CommDSLRuntimeError:
        pass
    # __mul__
    try:
        _ = lhs * rhs
        assert False
    except CommDSLRuntimeError:
        pass
    # __imul__
    try:
        lhs *= rhs
        assert False
    except CommDSLRuntimeError:
        pass
    # __idv__
    try:
        _ = lhs / rhs
        assert False
    except CommDSLRuntimeError:
        pass
    # __idiv__
    try:
        lhs /= rhs
        assert False
    except CommDSLRuntimeError:
        pass
