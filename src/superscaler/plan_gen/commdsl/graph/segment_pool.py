# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

class DataSegmentPool:
    """
    DataSegment Manager (Singleton). This class guarantees that:

    * Data segment block number is consistent and scalable
    * Each data segment id is unique and progressively increases,

    Attributes:
        _id (int): used for unique-id generator.
        _seg (dict{int : list[DataSegment]}):
            logged data segment. indexed by its id.
    """

    class __DataSegmentPool:
        def __init__(self):
            self._id = 0
            self._seg = dict()

    instance = None

    def __init__(self):
        if not DataSegmentPool.instance:
            DataSegmentPool.instance = DataSegmentPool.__DataSegmentPool()

    def __getattr__(self, name):
        return getattr(self.instance, name)

    def register(self, seg):
        """
        Register a data segment into DataSegmentPool.

        Args:
            seg (DataSegment): data segment to be registered

        Returns:
            None
        """
        if not callable(getattr(seg, 'get_id', None)):
            raise AttributeError("seg should have `get_id()`")
        if not callable(getattr(seg, 'scale', None)):
            raise AttributeError("seg should have `scale(factor)`")
        bid = seg.get_id()
        if bid not in self._seg:
            self._seg[bid] = [seg]
        else:
            self._seg[bid].append(seg)

    def scale_size(self, factor):
        """
        Scale the bnum of all data segments by `factor`.

        Args:
            factor (int): scale factor

        Returns:
            None
        """
        for bid in self._seg:
            for seg in self._seg[bid]:
                seg.scale(factor)

    def gen_id(self):
        """
        Generate a unique ID which progressively increases.

        Args:
            None

        Returns:
            ID (int)
        """
        self.instance._id += 1
        return self.instance._id

    def clear(self):
        """
        Clear logged data segment instances

        Args:
            None

        Returns:
            None
        """
        self.instance._id = 0
        self.instance._seg = dict()
