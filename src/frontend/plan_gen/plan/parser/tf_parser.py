import tensorflow as tf
from tensorflow.core.framework import node_def_pb2
from google.protobuf import text_format
from .DAG_parser import DAGParser
from .profiler.profiler import TFProfiler
from .profiler.database_backend import DatabaseBackendLocalFile


class ParserError(Exception):
    """Exception raised for errors in the input.

    Attributes:
        expression -- input expression in which the error occurred
        message -- explanation of the error
    """

    def __init__(self, expression):
        self.expression = expression


'''
Plan nodelist def:
    device <string>
    name <string>
    output_shapes <repeated><list>
    input <repeated><string>
    reduction <string>
    T <int>
'''


class TFParser(DAGParser):

    DT_mapping_table = {
            1: 'DT_FLOAT',
            2: 'DT_DOUBLE',
            3: 'DT_INT32',
            4: 'DT_UINT8',
            5: 'DT_INT16',
            6: 'DT_INT8',
            7: 'DT_STRING',
            8: 'DT_COMPLEX64',    # Single-precision complex
            9: 'DT_INT64',
            10: 'DT_BOOL',
            11: 'DT_QINT8',       # Quantized int8
            12: 'DT_QUINT8',      # Quantized uint8
            13: 'DT_QINT32',      # Quantized int32
            14: 'DT_BFLOAT16',    # Float32 truncated to 16 bits.
            15: 'DT_QINT16',      # Quantized int16
            16: 'DT_QUINT16',     # Quantized uint16
            17: 'DT_UINT16',
            18: 'DT_COMPLEX128',  # Double-precision complex
            19: 'DT_HALF',
            20: 'DT_RESOURCE',
            21: 'DT_VARIANT',     # Arbitrary C++ data types
            22: 'DT_UINT32',
            23: 'DT_UINT64'
        }

    def __init__(self, db_type=DatabaseBackendLocalFile, **kwargs):
        super().__init__("TFParser")
        self.__NodeAttrParser = TFNodeAttrParser()

        # Check whether the database is available or not
        try:
            self.__Profiler = TFProfiler(db_type, **kwargs)
        except Exception:
            self.__Profiler = None

    def parse_graphs(self, graph_paths, devices, load_from_memory=False):
        '''
        Parse all nodes from tensorflow DAG.
        Return the node_list that contains all parsed nodes
        graph_paths: path to tensorflow DAG
        load_from_memory: True to load data from memory,
            False to load data from files
        devices: virtual device_id
        '''
        node_list = []
        profiling_data_list = []

        if len(graph_paths) != len(devices):
            raise ValueError("Devices count %d cannot match graph count %d" % (
                                len(devices),
                                len(graph_paths)))

        for graph_path, device_id in zip(graph_paths, devices):
            if load_from_memory is False:
                graph = self.load_protobuf_from_file(graph_path)
            else:
                graph = self.load_protobuf(graph_path)

            profiling_data_sublist = self.get_profiling_data_list(graph,
                                                                  device_id)
            profiling_data_list.extend(profiling_data_sublist)

            for node in graph.node:
                attrs = self.__NodeAttrParser.parse_node(node)
                attrs['op'] = node.op
                attrs['input'] = node.input
                attrs['device'] = device_id
                attrs['name'] = node.name
                node_list.append(attrs)

        # Query the profiler database with the profiling_data info
        # to get the execution_time
        for node, profiling_data in zip(node_list, profiling_data_list):

            if self.__Profiler is not None:
                execution_time =\
                    self.__Profiler.get_node_execution_time(profiling_data)
            else:
                execution_time = None

            # introduce execution_time when it is available on profiler
            if execution_time is not None and 'avg' in execution_time:
                node['execution_time'] = execution_time['avg']

        filtered_node_list = self.__filter_node_list(node_list)
        return filtered_node_list

    def __filter_node_list(self, node_list):
        '''
        Filter all nodes in node_list
        Return the filtered node_list
        '''
        filtered_node_list = []

        for node in node_list:
            filtered_node_list.append(self.__filter_node(node))

        self.__unify_dependency(filtered_node_list)

        return filtered_node_list

    def __filter_node(self, node):
        '''
        Filter all nodes
        Return the filtered node
        index: node index
        node: node parsed from a graph
        '''
        filtered_node = {}
        if 'device' in node:
            filtered_node['device'] = node['device']
        if 'op' in node:
            filtered_node['op'] = self.__unify_op_name(node['op'])
        if 'name' in node:
            filtered_node['name'] = node['name']
        if '_output_shapes' in node:
            filtered_node['output_shapes'] = node['_output_shapes']
        if 'reduction' in node:
            filtered_node['reduction'] = node['reduction']
        if 'tensor_name' in node:
            filtered_node['tensor_name'] = node['tensor_name']
        if 'T' in node and node['T'] in self.DT_mapping_table:
            filtered_node['tensor_type'] = self.DT_mapping_table[node['T']]
        if 'input' in node:
            filtered_node['input'] = node['input']
        if 'execution_time' in node:
            filtered_node['execution_time'] = node['execution_time']

        return filtered_node

    @staticmethod
    def __unify_dependency(node_list):
        '''
        unify dependency as the same as node name
        node_list: the node_list parsed from a graph
        '''
        for node in node_list:
            inputs = []
            for input_str in node['input']:
                # remove ^ character in the begining
                input_str = input_str.replace('^', '')
                # remove :1 in the end
                if ':' in input_str:
                    input_str = input_str[0:input_str.index(':')]
                inputs.append(input_str)
            node['input'] = inputs

    @staticmethod
    def load_protobuf(graph_content):
        '''
        Load protobuf from memory
        Return the protobuf
        graph_content: string description of graph
        '''
        try:
            graph_def = text_format.Parse(graph_content,
                                          tf.compat.v1.GraphDef())
            return graph_def
        except text_format.ParseError as e:
            raise ParserError("Cannot parse description: %s." %
                              (str(e)))
        return graph_def

    @staticmethod
    def load_protobuf_from_file(filename):
        '''
        Load protobuf from file
        Return the protobuf
        filename: path to protobuf
        '''
        graph_def = None
        with open(filename, 'r') as f:
            file_content = f.read()
            try:
                graph_def = text_format.Parse(file_content,
                                              tf.compat.v1.GraphDef())
                return graph_def
            except text_format.ParseError as e:
                raise ParserError("Cannot parse file %s: %s." %
                                  (filename, str(e)))
        return graph_def

    @staticmethod
    def __unify_op_name(op):
        '''
        unify comm op name to 'Allreduce', 'Send' and 'Recv'
        Return unified op
        op: the name of op
        '''
        if 'allreduce' in op.lower():
            return 'Allreduce'
        elif 'send' in op.lower():
            return 'Send'
        elif 'recv' in op.lower():
            return 'Recv'
        else:
            return op

    def get_profiling_data_list(self,
                                graph,
                                device_id,
                                str_format=False):
        '''
        Parse all nodes form tensorflow DAGs.
        Return the profiling_data_list: a list of profiling_node
        profiling node structure:
            node:           original TF node
            device:         device id
            name:           name of node
            op:             op
            output_shapes:  List of each output tensor's shape
            input_shapes:   List of each input tensor's shape
            attr_list:      List of (attr key, attr value) pair
        '''

        profiling_data_list = []

        for node in graph.node:
            # This setting is only for debug
            if str_format is True:
                profiling_node = {"node_def": str(node),
                                  "device": device_id}
            else:
                profiling_node = {"node_def": node,
                                  "device": device_id}
            attrs = self.__NodeAttrParser.parse_node(node)

            # Note: We are currently unable to determine the complete set
            # of valid attributes
            # TODO: Determine the complete set and make code_book for it.
            # profiling_node ignore those attributes with string key,
            # rename the attribute name of _output_shapes as output_shapes
            # and accept all other attribute with original name and key
            profiling_node['attr_list'] = []
            profiling_node['output_shapes'] = []
            for attr_key in attrs:
                if attr_key == '_output_shapes':
                    profiling_node['output_shapes'] =\
                        attrs['_output_shapes']
                elif isinstance(attrs[attr_key], str):
                    continue
                else:
                    attr_value = attrs[attr_key]
                    profiling_node['attr_list'].append((attr_key, attr_value))

            profiling_node['name'] = str(node.name)
            profiling_node['input'], profiling_node['input_index'] =\
                self.__get_input_info(node.input)
            profiling_node['op'] = node.op
            profiling_data_list.append(profiling_node)

        self.__create_input_shapes(profiling_data_list)

        return profiling_data_list

    @staticmethod
    def __get_input_info(node_input):
        """
        In tensorflow proto, Each input is "node:src_output" with "node"
        being a string name and "src_output" indicating which output tensor
        to use from "node". If "src_output" is 0 the ":0" can be omitted.
        Regular inputs may optionally be followed by control inputs that
        have the format "^node".

        we output "node" as input_raw and "src_output" as input_index
        """

        input_raw = []
        input_index = []
        for input_str in node_input:
            if '^' in input_str:
                # For tensorflow, ^ indicates control input
                # In plan_gen, we ignore control input
                continue
            elif ':' in input_str:
                # For tensorflow, :1 indicates which output tensor to use
                # We parse the index directly
                index = int(input_str[input_str.index(':')+1:])
                input_str = input_str[0:input_str.index(':')]
            else:
                # 0 is the default option
                index = 0

            input_raw.append(input_str)
            input_index.append(index)

        return input_raw, input_index

    @staticmethod
    def __create_input_shapes(profiling_data_list):
        """
        With node input and node input_index infos, we can find the
        input shapes by get the output tensor shape of node which has
        the same node name and device with the current node.
        """

        # create a code_book for input_shape
        input_shapes_dict = {}
        for node in profiling_data_list:
            input_shapes_dict[(node['name'], node['device'])] = \
                node['output_shapes']

        # introduce input_shape to node
        for node in profiling_data_list:
            input_shapes = []

            for input_, index_ in zip(node['input'], node['input_index']):
                if (input_, node['device']) in input_shapes_dict and \
                   index_ >= 0 and \
                   index_ < len(input_shapes_dict[(input_, node['device'])]):
                    input_shapes.append(
                        input_shapes_dict[(input_, node['device'])][index_])
                else:
                    raise Exception("Raise Exception on create_input_shapes")

            node['input_shapes'] = input_shapes


'''
Node Message def:
    name <string>
    op <string>
    input <repeated><string>
    device <string>
    attr <map<string, AttrValue>>
    experimental_debug_info <message> // Should be ignored

AttrValue Message def:
    one of value:
        s <bytes> // trans into string
        i <int64> // trans into integer
        f <float>
        b <bool> // trans into 1/0
        type <DataType> // DataType is a enum Message, trans into integer
        shape <TensorShapeProto> // trans into list of int
                                 // expand all shape to 4-d
        tensor <TensorProto>
        list <ListValue> // trans into list of object
        func <NameAttrList> // trans into list of object, ignore
        placeholder <string> // ignore

TensorShapeProto Message def:
    Dim Message def:
        size <int64>
        name <string> // optimal member, ignore
    dim <repeated><Dim>
    unknown_rank <bool> // if true, dimensions are unknown,
                        // trans into [1,1,1,1]

TensorProto Message def:
    dtype <DataType>
    tensor_shape <TensorShapeProto>
    # Ignored tensor content. Only need tensor shape and type
    tensor_content <Bytes>
    half_val <repeated><int32>
    float_val <repeated><float>
    double_val <repeated><double>
    int_val <repeated><int32>
    string_val <repeated><bytes>
    ...


ListValue Message def:
    s <bytes> // string
    i <int64> // integer
    f <float>
    b <bool>
    type <DataType>
    shape <TensorShapeProto>
    func <NameAttrList>

NameAttrList Message def:
    name <string>
    attr <map<string, AttrValue>>
'''


class TFNodeAttrParser():

    def parse_node(self, node, in_str_format=False):
        '''
        Parse all attributes for a node.
        Return the dict that contains all parsed attr key-value
        node: TF node define object
        '''

        if not isinstance(node, node_def_pb2.NodeDef):
            raise ParserError(
                'Node initialization failure for wrong input foramt: %s' %
                str(type(node)))

        attrs = {}
        if not hasattr(node, 'attr'):
            raise ParserError("Faild to find attr from node")
        for attr_name in node.attr:
            attr_value = self.__parse_attr_value(node.attr[attr_name],
                                                 in_str_format=in_str_format)
            attrs[attr_name] = attr_value

        return attrs

    def __parse_attr_value(self, raw_attr_value, in_str_format=False,
                           value_list=False):
        '''
        Parse all attribute value.
        For attr_value Message class, should use HasField() to check the
        existance of an attribute.
        raw_attr_value: original attribute value object
        in_str_format:  If True, parse into string forat for database key.
                        If False, parse into data value for training.
        value_list:  If True, treat raw_attr_value as a list.
                     If False, treat raw_attr_value as a attr.
        '''
        if value_list is True:
            value = []
        else:
            value = None
        fields_name = []

        if not hasattr(raw_attr_value, 'ListFields'):
            raise ParserError("Faild to find ListFields from raw_attr_value")
        for field in raw_attr_value.ListFields():
            name = field[0].name
            fields_name.append(name)

        if 's' in fields_name:
            if value_list is True:
                for itr in raw_attr_value.s:
                    str_value = itr
                    if isinstance(itr, bytes):
                        str_value = itr.decode()
                    value.append(str_value)
            else:
                value = raw_attr_value.s
                if isinstance(value, bytes):
                    value = value.decode()
        elif 'i' in fields_name:
            if value_list is True:
                for itr in raw_attr_value.i:
                    value.append(itr)
            else:
                value = raw_attr_value.i
        elif 'f' in fields_name:
            if value_list is True:
                for itr in raw_attr_value.f:
                    value.append(itr)
            else:
                value = raw_attr_value.f
        elif 'b' in fields_name:
            if value_list is True:
                for itr in raw_attr_value.b:
                    if itr is True:
                        value.append(1)
                    else:
                        value.append(0)
            else:
                if raw_attr_value.b is True:
                    value = 1
                else:
                    value = 0
        elif 'type' in fields_name:
            if value_list is True:
                for itr in raw_attr_value.type:
                    value.append(itr)
            else:
                value = raw_attr_value.type
        elif 'shape' in fields_name:
            if value_list is True:
                for itr in raw_attr_value.shape:
                    value.append(self.__parse_shape_attr(itr))
            else:
                value = self.__parse_shape_attr(raw_attr_value.shape)
        elif 'tensor' in fields_name:
            if value_list is True:
                for itr in raw_attr_value.tensor:
                    value.append(self.__parse_tensor_attr(itr))
            else:
                value = self.__parse_tensor_attr(raw_attr_value.tensor)
        elif 'list' in fields_name:
            if value_list is True:
                for itr in raw_attr_value.list:
                    value.append(self.__parse_attr_value(itr), value_list=True)
            else:
                value = self.__parse_attr_value(raw_attr_value.list,
                                                value_list=True)

        # Ignore func and placeholder feature
        if value_list is True:
            if in_str_format is True and len(value) > 0:
                for i in range(len(value)):
                    value[i] = str(value[i])
        else:
            if in_str_format is True and value is not None:
                value = str(value)

        return value

    def __parse_shape_attr(self, raw_shape_value):
        '''
        Parse the Shape message. Return a list of int.
        raw_shape_value: the shape message object.
        '''
        value = []

        shape_list = raw_shape_value.dim
        for dim in shape_list:
            if not hasattr(dim, 'size'):
                raise ParserError("Faild to find size from dim")
            value.append(dim.size)
        return value

    def __parse_tensor_attr(self, raw_tensor_value):
        '''
        Parse the Tensor massage.
        Return value: [<dtype>, [<int list of shape>]]
        raw_tensor_value: the tensor message object
        '''
        value = []

        for field in raw_tensor_value.ListFields():
            if 'dtype' == field[0].name:
                value.append(raw_tensor_value.dtype)
                break
        if not value:
            return None

        shape = self.__parse_shape_attr(raw_tensor_value.tensor_shape)
        value.append(shape)
        return value
