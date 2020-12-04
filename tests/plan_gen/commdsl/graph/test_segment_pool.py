# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from superscaler.plan_gen.commdsl.graph.segment_pool import DataSegmentPool
from superscaler.plan_gen.commdsl.graph.segment import DataSegment


class ErrorSegNoID:

    def __init__(self):
        self.get_id = 0
        pass

    def scale(self):
        pass


class ErrorSegNoScale:

    def __init__(self):
        self.scale = 0
        pass

    def get_id(self):
        pass


def test_segment_pool_gen_id():
    DataSegmentPool().clear()
    assert DataSegmentPool()._id == 0
    assert len(DataSegmentPool()._seg) == 0
    # test gen_id
    sid1 = DataSegmentPool().gen_id()
    assert sid1 == 1
    sid2 = DataSegmentPool().gen_id()
    assert sid2 == 2


def test_segment_pool_register():
    # test register
    DataSegmentPool().clear()
    seg = DataSegment(1)
    _ = DataSegment(4)
    assert len(DataSegmentPool()._seg[1]) == 1
    assert len(DataSegmentPool()._seg[2]) == 1
    assert DataSegmentPool()._seg[1][0] == seg
    seg_no_id = ErrorSegNoID()
    seg_no_scale = ErrorSegNoScale()
    try:
        DataSegmentPool().register(seg_no_id)
        assert False
    except AttributeError:
        pass
    try:
        DataSegmentPool().register(seg_no_scale)
        assert False
    except AttributeError:
        pass


def test_segment_pool_scale():
    # test scale bnum
    data_s1 = DataSegment(1)
    data_s2 = DataSegment(2)
    DataSegmentPool().scale_size(3)
    assert data_s1.bnum == 3
    assert data_s2.bnum == 6
