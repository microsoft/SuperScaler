import tensorflow as tf
from tensorflow.core.framework import node_def_pb2
from google.protobuf import text_format

from adapter.DAG_parser import DAGParser


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
    def __init__(self):
        super().__init__("TFParser")
        self.__NodeAttrParser = TFNodeAttrParser()

    def parse_graphs(self, graph_paths, devices):
        '''
        Parse all nodes from tensorflow DAG.
        Return the node_list that contains all parsed nodes
        graph_paths: path to tensorflow DAG
        devices: virtual device_id
        '''
        node_list = []

        if len(graph_paths) != len(devices):
            raise ValueError("Devices count %s is not same as graph count %s"
                             % (
                                 len(devices),
                                 len(graph_paths)))

        for graph_path, device_id in zip(graph_paths, devices):
            graph = self.__load_protobuf_from_file(graph_path)
            for node in graph.node:
                attrs = self.__NodeAttrParser.parse_node(node)
                attrs['op'] = node.op
                attrs['input'] = node.input
                attrs['device'] = device_id
                attrs['name'] = node.name
                node_list.append(attrs)

        # return self.__plan
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
        if 'T' in node:
            filtered_node['tensor_type'] = node['T']
        if 'input' in node:
            filtered_node['input'] = node['input']

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
    def __load_protobuf_from_file(filename):
        '''
        Load protobuf from file
        Return the protobuf
        filename: path to protobuf
        '''
        graph_def = None
        with open(filename, 'r') as f:
            file_content = f.read()
            try:
                graph_def = text_format.Parse(
                    file_content, tf.compat.v1.GraphDef())
                return graph_def
            except text_format.ParseError as e:
                raise ParserError("Cannot parse file %s: %s."
                                  % (filename, str(e)))
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
        shape <TensorShapeProto> // trans into list of int,
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

class DataType(Enum):
    # Data types that all computation devices are expected to be
    # capable to support.
    # REF type = normal type + 100
    DT_FLOAT = 1
    DT_DOUBLE = 2
    DT_INT32 = 3
    DT_UINT8 = 4
    DT_INT16 = 5
    DT_INT8 = 6
    DT_STRING = 7
    DT_COMPLEX64 = 8  # Single-precision complex
    DT_INT64 = 9
    DT_BOOL = 10
    DT_QINT8 = 11     # Quantized int8
    DT_QUINT8 = 12    # Quantized uint8
    DT_QINT32 = 13    # Quantized int32
    DT_BFLOAT16 = 14  # Float32 truncated to 16 bits.  Only for cast ops.
    DT_QINT16 = 15    # Quantized int16
    DT_QUINT16 = 16   # Quantized uint16
    DT_UINT16 = 17
    DT_COMPLEX128 = 18  # Double-precision complex
    DT_HALF = 19
    DT_RESOURCE = 20
    DT_VARIANT = 21  # Arbitrary C++ data types
    DT_UINT32 = 22
    DT_UINT64 = 23
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
                value = self.__parse_attr_value(
                    raw_attr_value.list, value_list=True)

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
