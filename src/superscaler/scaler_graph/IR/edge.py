class Edge:
    '''Edge
    '''
    _ID = 0

    def __init__(self, src_node, src_idx, dest_node, dest_idx):
        assert (src_idx == -1
                or src_node.get_output_tensor(src_idx) is not None)
        self.id = self._ID
        self.__class__._ID += 1
        self.src_node = src_node
        self.src_idx = src_idx  # tensor index
        self.dest_node = dest_node
        self.dest_idx = dest_idx  # slot index

    def is_control_edge(self):
        '''src_idx is -1 for control edges
        '''
        return self.src_idx == -1
