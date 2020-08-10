from .database_loader import DatabaseLoader
from .database_backend import DatabaseBackendLocalFile


class Profiler(object):
    '''
    Virtual class of Profiler with virtual interfaces
    __init__ and get_node_execution_time
    '''
    def __init__(self):
        pass

    def get_node_execution_time(self, node):
        return 0.0


class TFProfiler(Profiler):
    '''
    Profiler class for tensorflow DAG
    '''
    def __init__(self, db_type=DatabaseBackendLocalFile, **kwargs):
        self.database = DatabaseLoader(db_type, **kwargs)

    '''
    Get a node's execution time from DB.
    TODO: if not find in DB, return 0 now. Will add online profiler later.
    '''
    def get_node_execution_time(self, node):
        # TODO Introduce online profiler
        if(not isinstance(node, dict) or 'op' not in node or
           'input_shapes' not in node or 'attr_list' not in node):
            return None
        op = node['op']
        input_shape_list = node['input_shapes']
        attr_list = node['attr_list']
        result = self.database.search_record(op, input_shape_list, attr_list)
        return result
