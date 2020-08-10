from .database_backend import DatabaseBackendException,\
    DatabaseBackendLocalFile

DB_RETVAL_OK = 0
DB_RETVAL_KEY_NOT_FOUND = None


class DatabaseLoader():
    '''
    This class read database loader and accept database search/add requests.
    The database is a key-value structure database.
    The key is made up with op, input shapes, and sorted attributes.
    Elements in the key is separated by '%'
    Key format:
        "op%input%input0_shape%...%input%inputN_shape%attr0_key%attr0_value..."
        op: the string of op.
        input_shape: the string format of input shape list. E.g. '[3,64,64,3]'
        attr_key: the string of attribute name. E.g. 'dtype'
        attr_value: the string of attribute value. E.g. 'DT: TF_FLOAT'
    '''
    '''
    __db_type_support_list: supported database backend classes
    '''
    __db_type_support_list = [DatabaseBackendLocalFile]

    def __init__(self, db_type=DatabaseBackendLocalFile, **kwargs):
        '''
        db_type:    Class of database backend
        kwargs:     Other initialization parameters for database backend
        '''
        if db_type in self.__db_type_support_list:
            self.__db_backend = db_type(**kwargs)
        else:
            raise DatabaseBackendException('Invalid DatabaseBackend type')

    def __gen_input_shape_string(self, input_shape):
        '''
        Transform input shape from list format to string.
        Do not use str() because different environment may cause different
        transform result.
        '''
        final_str = '['
        if len(input_shape) > 0:
            final_str = final_str + str(input_shape[0])
        for i in range(1, len(input_shape)):
            final_str = final_str + ',' + str(input_shape[i])
        final_str = final_str + ']'
        return final_str

    def __gen_universal_key(self, op, input_shape_list, attr_list):
        '''
        Generate the universal key.

        Args:
            input_shape_list: list of all input shape. Each element is a list,
                means the shape.
            attr_list: list of all attributes. Each element is a pair of two
                string, (key, value)
        '''
        '''
        Ignored attribute list. These attributes will not affect op execution
        performance.
        _class: This attribute means the op should be put into the same device
            with another op.
        experimental_debug_info: Debug info.
        E.g. in a graph, a conv2d node 'conv0', its '_class' attr is 'const0'.
        This means this conv2d node should be placed into the same device with
        node 'const0'. This will not affect its execution time.
        '''
        if not isinstance(op, str) or not isinstance(input_shape_list, list)\
           or not isinstance(attr_list, list):
            return None
        IGNORE_ATTR_LIST = ['_class',
                            'experimental_debug_info',
                            '_output_shapes']
        sorted_attr_list = sorted(attr_list, key=lambda key: key[0])
        key = op
        for input_shape in input_shape_list:
            input_shape_str = self.__gen_input_shape_string(input_shape)
            key = key + '%%input%%%s' % input_shape_str
        for attr_name, attr_value in sorted_attr_list:
            if attr_name in IGNORE_ATTR_LIST:
                continue
            attr_name = attr_name.replace(' ', '')
            attr_value = str(attr_value).replace(' ', '')
            key = key + '%%%s%%%s' % (attr_name, attr_value)
        return key

    def search_record(self, op, input_shape_list, attr_list):
        '''
        Search record in database, given key_list, return profiling result in
        dict format. If not found, return DB_RETVAL_KEY_NOT_FOUND

        Args:
            op: string, the op.
            input_shape_list: list of all input shape. Each element is a list,
                means the shape.
            attr_list: list of all attributes. Each element is a pair of two
                string, (key, value)
        '''
        key = self.__gen_universal_key(op, input_shape_list, attr_list)
        value = self.__db_backend.get(key)
        if value is None:
            return DB_RETVAL_KEY_NOT_FOUND
        return value

    def add_record(self, op, input_shape_list, attr_list, result_dict):
        '''
        Add a record into the database. If same key exists, overwrite the old
        record.

        Args:
            op: string, the op.
            input_shape_list: list of all input shape. Each element is a list,
                means the shape.
            attr_list: list of all attributes. Each element is a pair of two
                string, (key, value)
            result_dict: the dict store profiling result.
        '''
        key = self.__gen_universal_key(op, input_shape_list, attr_list)
        value = result_dict
        self.__db_backend.put(key, value)
        return DB_RETVAL_OK

    def remove_record(self, op, input_shape_list, attr_list):
        '''
        Remove a record from the database.

        Args:
            op: string, the op.
            input_shape_list: list of all input shape. Each element is a list,
                means the shape.
            attr_list: list of all attributes. Each element is a pair of two
                string, (key, value)
        '''
        key = self.__gen_universal_key(op, input_shape_list, attr_list)
        self.__db_backend.pop(key)
