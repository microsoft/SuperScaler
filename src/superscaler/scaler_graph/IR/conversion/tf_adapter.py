# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import google.protobuf.text_format
import os
import re
from pathlib import Path
from tensorflow.python import types_pb2, tensor_shape
from tensorflow.core.framework import tensor_pb2
from tensorflow.core.framework.op_def_pb2 import OpDef
import tensorflow as tf
from tensorflow.python.util import compat
from tensorflow.python.framework.ops import Operation, Tensor
from tensorflow.python.framework.op_def_library import (
    _IsListValue,
    _MakeBool,
    _MakeFloat,
    _MakeInt,
    _MakeShape,
    _MakeStr,
    _MakeTensor,
    _MakeType,
)
from superscaler.scaler_graph.IR.graph import Graph
from superscaler.scaler_graph.IR.conversion.tensorflow_ops \
    import tf_op_map_to_sc_op, convert_to_tf_node
from superscaler.scaler_graph.IR.util import graph_util
from superscaler.scaler_graph.util.log import logger
__all__ = [
    "import_graph_from_tf_file", "get_tf_runtime_config",
    "export_graph_to_tf_file", "import_tensorflow_model"
]


def get_dtype_proto(node_def, op_def, output_arg):
    def with_number_attr(dtype):
        if len(output_arg.number_attr) != 0:
            for attr in op_def.attr:
                if attr.name == output_arg.number_attr:
                    return [dtype] * node_def.attr[attr.name].i
            raise AssertionError
        else:
            return dtype

    if len(output_arg.type_attr) != 0:
        for attr in op_def.attr:
            if attr.name == output_arg.type_attr:
                return with_number_attr(node_def.attr[attr.name].type)
        raise AssertionError
    elif len(output_arg.type_list_attr) != 0:
        for attr in op_def.attr:
            if attr.name == output_arg.type_list_attr:
                return list(node_def.attr[attr.name].list.type)
        raise AssertionError
    else:
        assert output_arg.type != types_pb2.DT_INVALID
        return with_number_attr(output_arg.type)


def get_dtypes(tf_graph, node_def):
    '''parse tf dtypes.
    '''
    op_def = tf_graph._get_op_def(node_def.op)
    dtypes = [
        get_dtype_proto(node_def, op_def, output_arg)
        for output_arg in op_def.output_arg
    ]
    if len(dtypes) == 1 and isinstance(dtypes[0], list):
        dtypes = dtypes[0]
    return [tf.as_dtype(dtype) for dtype in dtypes]


def from_attr_proto(attr_value):
    '''parse tf node attributions.
    '''
    field_name = attr_value.WhichOneof("value")
    if field_name == "s":
        return attr_value.s
    elif field_name == "i":
        return attr_value.i
    elif field_name == "f":
        return attr_value.f
    elif field_name == "b":
        return attr_value.b
    elif field_name == "type":
        return tf.as_dtype(attr_value.type)
    elif field_name == "shape":
        return tensor_shape.as_shape(attr_value.shape)
    elif field_name == "tensor":
        return attr_value.tensor
    elif field_name == "func":
        return attr_value.func
    elif field_name == "placeholder":
        return attr_value.placeholder
    elif field_name == "list":
        list_value = attr_value.list
        if len(list_value.s) != 0:
            return [value for value in list_value.s]
        elif len(list_value.i) != 0:
            return [value for value in list_value.i]
        elif len(list_value.f) != 0:
            return [value for value in list_value.f]
        elif len(list_value.b) != 0:
            return [value for value in list_value.b]
        elif len(list_value.type) != 0:
            return [tf.as_dtype(value) for value in list_value.type]
        elif len(list_value.shape) != 0:
            return [tensor_shape.as_shape(value) for value in list_value.shape]
        elif len(list_value.tensor) != 0:
            return [value for value in list_value.tensor]
        elif len(list_value.func) != 0:
            return [value for value in list_value.func]
        else:
            return []


def import_graph_from_tf_file(init_path=None, run_path=None):
    '''convert tf graphs to sc graph.
    1. merge tf init graph and run graph into tf_graph_def;
    2. conver tf_graph_def to sc graph
    Return:
        SC graph
    '''
    tf_graph_def = tf.GraphDef()
    if init_path is not None:
        google.protobuf.text_format.Parse(
            Path(init_path).read_text(), tf_graph_def)
    if run_path is not None:
        google.protobuf.text_format.Merge(
            Path(run_path).read_text(), tf_graph_def)
    sc_graph = Graph()
    tf_graph = tf.Graph()

    def add_sc_before_underscore(name):
        '''tf.import_graph_def() can't parse nodes with prefix "_",
        Add "sc" before "_".
        '''
        obj = re.match("^_.*$", name)
        if obj is not None:
            name = "sc" + name
        return name

    name_to_node = {}
    for node in tf_graph_def.node:
        node.name = add_sc_before_underscore(node.name)
        name_to_node[node.name] = node

    def add_sc_node(tf_node: tf.NodeDef):
        if sc_graph.get_node_by_name(node.name) is not None:
            return
        input_node_idxes = []
        for input in tf_node.input:
            if input.startswith("^"):
                input_node_name = input[1:]
                # check control edge name
                input_node_name = add_sc_before_underscore(input_node_name)
                if sc_graph.get_node_by_name(input_node_name) is not None:
                    input_node = sc_graph.get_node_by_name(input_node_name)
                else:
                    add_sc_node(name_to_node[input_node_name])
                    input_node = sc_graph.get_node_by_name(input_node_name)
                index = -1
            else:
                names = input.split(":")
                assert len(names) == 1 or len(names) == 2
                # check data edge name
                names[0] = add_sc_before_underscore(names[0])
                if sc_graph.get_node_by_name(names[0]) is not None:
                    input_node = sc_graph.get_node_by_name(names[0])
                else:
                    add_sc_node(name_to_node[names[0]])
                    input_node = sc_graph.get_node_by_name(names[0])
                if len(names) == 1:
                    index = 0
                else:
                    index = int(names[1])
            input_node_idxes.append((input_node, index))

        attrs = {
            attr_name: from_attr_proto(tf_node.attr[attr_name])
            for attr_name in tf_node.attr
        }
        dtypes = get_dtypes(tf_graph, tf_node)
        sc_node = sc_graph.add_node_and_edge(
            tf_node.name, tf_op_map_to_sc_op(tf_graph._get_op_def(tf_node.op)),
            input_node_idxes, len(dtypes), attrs)
        sc_node.attrs["tf"] = {}
        sc_node.attrs["tf"]["device"] = ""
        sc_node.attrs["tf"]["dtypes"] = dtypes
        if tf_node.HasField("experimental_debug_info"):
            sc_node.attrs["tf"][
                "experimental_debug_info"] = node.experimental_debug_info

    for node in tf_graph_def.node:
        add_sc_node(node)

    for key in ["versions", "library"]:
        if tf_graph_def.HasField(key):
            sc_graph.attrs[key] = getattr(tf_graph_def, key)
    sc_graph.attrs["meta_graph"] = tf.MetaGraphDef()
    sc_graph.attrs["initialized_variables"] = {}
    sc_graph.attrs["lower_name_func"] = (lambda name: name.lower())
    return sc_graph


def get_tf_runtime_config(sc_graph):
    '''find some specific nodes for tf runtime.
    inits: nodes for backtracing all initialization nodes of variables.
    feeds: nodes without input, providing training data.
    fetches: feedback tensors for users, e.g. loss.
    targets: nodes without output, for backtracing all nodes needed to perform.
    '''
    tf_runtime_config = {}
    tf_runtime_config["inits"] = []  # for all assign op
    tf_runtime_config["feeds"] = []  # no need now. it's for training data.
    tf_runtime_config["fetches"] = []  # the successor node of _Retval
    tf_runtime_config["targets"] = []  # send or final
    for sc_node in graph_util.get_output_nodes(sc_graph):
        if sc_node.op.original_name == "_Retval":
            assert (len(sc_node.in_edges) == 1)
            fetch_name = sc_node.in_edges[0].src_node.name + ":%d" % (
                sc_node.in_edges[0].src_idx)
            tf_runtime_config["fetches"].append(fetch_name)
        elif sc_node.name == "init" and sc_node.op.original_name == "NoOp":
            tf_runtime_config["inits"].append(sc_node.name)
        elif sc_node.op.original_name == "NoOp":
            tf_runtime_config["targets"].append(sc_node.name)
    return tf_runtime_config


def sc_attrs_to_tf_attrs_proto(op_def, op_type_name, attrs):
    '''Convert attr values to AttrValue protos
    '''
    attr_protos = {}
    attr_defs = {attr_def.name: attr_def for attr_def in op_def.attr}
    for key, value in attrs.items():
        attr_value = tf.AttrValue()
        if key in attr_defs:
            attr_def = attr_defs[key]
        elif value is None:
            attr_protos[key] = attr_value
            continue
        else:
            attr_def = OpDef.AttrDef()
            if isinstance(value, (str, bytes)):
                attr_def.type = "string"
            elif isinstance(value, float):
                attr_def.type = "float"
            elif isinstance(value, bool):
                attr_def.type = "bool"
            # bool is a subclass of int, so we should check bool
            # before checking int
            elif isinstance(value, int):
                attr_def.type = "int"
            elif isinstance(value, tf.DType):
                attr_def.type = "type"
            elif isinstance(value, tf.TensorShape):
                attr_def.type = "shape"
            elif isinstance(value, tensor_pb2.TensorProto):
                attr_def.type = "tensor"
            elif isinstance(value, tf.NameAttrList):
                attr_def.type = "func"
            elif isinstance(value, list) and len(value) == 0:
                attr_value.list.SetInParent()
                attr_protos[key] = attr_value
                continue
            elif isinstance(value, list) and isinstance(
                    value[0], (str, bytes)):
                attr_def.type = "list(string)"
            elif isinstance(value, list) and isinstance(value[0], bool):
                attr_def.type = "list(bool)"
            # bool is a subclass of int, so we should check bool before
            # checking int
            elif isinstance(value, list) and isinstance(value[0], int):
                attr_def.type = "list(int)"
            elif isinstance(value, list) and isinstance(value[0], float):
                attr_def.type = "list(float)"
            elif isinstance(value, list) and isinstance(value[0], tf.DType):
                attr_def.type = "list(type)"
            elif isinstance(value, list) and isinstance(
                    value[0], tf.TensorShape):
                attr_def.type = "list(shape)"
            elif isinstance(value, list) and isinstance(
                    value[0], tensor_pb2.TensorProto):
                attr_def.type = "list(tensor)"
            else:
                logger().error(f"{value} has unsupported type")
                raise RuntimeError
        if attr_def.HasField("default_value") and value is None:
            attr_value.CopyFrom(attr_def.default_value)
            attr_protos[key] = attr_value
            continue
        if attr_def.type.startswith("list("):
            if not _IsListValue(value):
                logger().error("Expected list for attr " + key)
                raise TypeError
            if attr_def.has_minimum:
                if len(value) < attr_def.minimum:
                    logger().error(
                        "Attr '%s' of '%s' Op passed list of length %d "
                        "less than minimum %d." %
                        (key, op_type_name, len(value), attr_def.minimum))
                    raise ValueError
            attr_value.list.SetInParent()
        if attr_def.type == "string":
            attr_value.s = _MakeStr(value, key)
            if attr_def.HasField("allowed_values"):
                if attr_value.s not in attr_def.allowed_values.list.s:
                    logger().error(
                        "Attr '%s' of '%s' Op passed string '%s' not \
                            in: \"%s\"." % (
                            key,
                            op_type_name,
                            compat.as_text(attr_value.s),
                            '", "'.join(
                                map(compat.as_text,
                                    attr_def.allowed_values.list.s)),
                        ))
                    raise ValueError
        elif attr_def.type == "list(string)":
            attr_value.list.s.extend([_MakeStr(x, key) for x in value])
            if attr_def.HasField("allowed_values"):
                for x in attr_value.list.s:
                    if x not in attr_def.allowed_values.list.s:
                        logger().error(
                            "Attr '%s' of '%s' Op passed string '%s' not \
                                in: \"%s\"." % (
                                key,
                                op_type_name,
                                compat.as_text(x),
                                '", "'.join(
                                    map(compat.as_text,
                                        attr_def.allowed_values.list.s)),
                            ))
                        raise ValueError
        elif attr_def.type == "int":
            attr_value.i = _MakeInt(value, key)
            if attr_def.has_minimum:
                if attr_value.i < attr_def.minimum:
                    logger().error(
                        "Attr '%s' of '%s' Op passed %d less than minimum %d."
                        % (key, op_type_name, attr_value.i, attr_def.minimum))
                    raise ValueError
        elif attr_def.type == "list(int)":
            attr_value.list.i.extend([_MakeInt(x, key) for x in value])
        elif attr_def.type == "float":
            attr_value.f = _MakeFloat(value, key)
        elif attr_def.type == "list(float)":
            attr_value.list.f.extend([_MakeFloat(x, key) for x in value])
        elif attr_def.type == "bool":
            attr_value.b = _MakeBool(value, key)
        elif attr_def.type == "list(bool)":
            attr_value.list.b.extend([_MakeBool(x, key) for x in value])
        elif attr_def.type == "type":
            attr_value.type = _MakeType(value, attr_def)
        elif attr_def.type == "list(type)":
            attr_value.list.type.extend(
                [_MakeType(x, attr_def) for x in value])
        elif attr_def.type == "shape":
            attr_value.shape.CopyFrom(_MakeShape(value, key))
        elif attr_def.type == "list(shape)":
            attr_value.list.shape.extend([_MakeShape(x, key) for x in value])
        elif attr_def.type == "tensor":
            attr_value.tensor.CopyFrom(_MakeTensor(value, key))
        elif attr_def.type == "list(tensor)":
            attr_value.list.tensor.extend([_MakeTensor(x, key) for x in value])
        elif attr_def.type == "func":
            if isinstance(value, tf.NameAttrList):
                attr_value.func.CopyFrom(value)
            elif isinstance(value, compat.bytes_or_text_types):
                attr_value.func.name = value
            else:
                value.add_to_graph(tf.get_default_graph())
                attr_value.func.name = value.name
        else:
            logger().error("Unrecognized Attr type " + attr_def.type)
            raise TypeError

        attr_protos[key] = attr_value
    return attr_protos


def export_graph_to_tf_file(sc_graph, file_path=None):
    '''convert sc graph to tf graph
    TODO(gbxu): the library file path should be configurable.
    '''
    proj_path = os.environ["SUPERSCLAR_PATH"]
    lib_path = proj_path + "/lib/libsuperscaler_pywrap.so"
    if os.path.exists(lib_path):
        tf.load_library(lib_path)
    else:
        logger().error("The library file %s does not exist." % (lib_path))
        raise RuntimeError
    tf_graph = tf.Graph()
    graph_def = tf_graph.as_graph_def(add_shapes=True)
    for key in ["versions", "library"]:
        if key in sc_graph.attrs:
            getattr(graph_def, key).CopyFrom(sc_graph.attrs[key])
    for sc_node in sc_graph.nodes:
        convert_to_tf_node(sc_node)
        attrs = {
            name: value
            for name, value in sc_node.attrs.items() if name != "tf"
        }
        tf_node = graph_def.node.add()
        tf_node.name = sc_node.name
        tf_node.op = sc_node.op.original_name
        tf_node.device = sc_node.attrs["tf"]["device"]
        if "experimental_debug_info" in sc_node.attrs["tf"]:
            tf_node.experimental_debug_info.CopyFrom(
                sc_node.attrs["experimental_debug_info"])
        for name, attr_value in sc_attrs_to_tf_attrs_proto(
                tf_graph._get_op_def(tf_node.op), tf_node.op, attrs).items():
            tf_node.attr[name].CopyFrom(attr_value)

        for in_edge in sc_node.in_edges:
            if in_edge.src_idx == -1:
                in_edge_str = f"^{in_edge.src_node.name}"
            elif in_edge.src_idx == 0:
                in_edge_str = f"{in_edge.src_node.name}"
            else:
                in_edge_str = f"{in_edge.src_node.name}:{in_edge.src_idx}"
            tf_node.input.append(in_edge_str)

    output_graph = tf.Graph()
    with output_graph.as_default():
        tf.import_graph_def(graph_def, name="")
    graph_def = output_graph.as_graph_def(add_shapes=True)
    graph_pbtxt = google.protobuf.text_format.MessageToString(graph_def)
    if file_path is not None:
        file = Path(file_path)
        file.write_text(graph_pbtxt)
    return graph_pbtxt


def import_tensorflow_model(apply_gradient_op, loss, dir_path=None):
    '''import tensorflow graph according to apply_gradient_op, loss
        1. get tensorflow model via apply_gradient_op and loss;
        2. dump graph from tensorflow model.
    Args:
        apply_gradient_op: An Operation that applies the specified gradients,
            get it by call `tf.train.Optimizer.apply_gradients()`.
        loss: A Tensor containing the value to minimize, it's the first input
            argument of `tf.train.Optimizer.compute_gradients()`.
        dir_path: Optional. A temp directory for dumping tf graph.
    Returns:
        A SC Graph.
    '''
    if not (isinstance(apply_gradient_op, Operation)
            and isinstance(loss, Tensor)):
        logger().error("apply_gradient_op or loss is incorrect.")
        raise RuntimeError
    if dir_path is None:
        if "TF_DUMP_GRAPH_PREFIX" not in os.environ.keys():
            logger().error(
                "dir_path can't be None if not setting TF_DUMP_GRAPH_PREFIX.")
            raise RuntimeError
        elif not os.path.isdir(os.environ["TF_DUMP_GRAPH_PREFIX"]):
            logger().error("TF_DUMP_GRAPH_PREFIX is incorrect: %s" %
                           (os.environ["TF_DUMP_GRAPH_PREFIX"]))
            raise RuntimeError
    else:
        if os.path.isdir(dir_path):
            os.environ["TF_DUMP_GRAPH_PREFIX"] = dir_path
            logger().info("dumping dir path: %s" % (dir_path))
        else:
            logger().error("dumping dir path is not correct: %s" % (dir_path))
            raise RuntimeError
    if "TF_CPP_MIN_VLOG_LEVEL" not in os.environ.keys() or int(
            os.environ["TF_CPP_MIN_VLOG_LEVEL"]) < 3:
        logger().error('''The environment variable TF_CPP_MIN_VLOG_LEVEL \
should be set before importing tensorflow, \
e.g.: `TF_DUMP_GRAPH_PREFIX=path_to_empty_directory \
TF_CPP_MIN_VLOG_LEVEL=3 python your_script.py`''')
        raise RuntimeError

    def dump_pbtxts():
        '''we dumps tf graph via TF_DUMP_GRAPH_PREFIX and
        TF_CPP_MIN_VLOG_LEVEL. It's a tf log mechanism and
        setting TF_CPP_MIN_VLOG_LEVEL before importing tensorflow is required.
        '''
        options = tf.RunOptions(output_partition_graphs=True)
        run_metadata = tf.RunMetadata()
        run_config = tf.ConfigProto()
        run_config.gpu_options.allow_growth = True
        run_config.graph_options.place_pruned_graph = True
        with tf.Session(config=run_config) as sess:
            sess.run(tf.global_variables_initializer())
            sess.run([apply_gradient_op, loss],
                     options=options,
                     run_metadata=run_metadata)
        tf.reset_default_graph()

    def get_dumped_pbtxts():
        '''dumping tf graph will generate 2 files:
        init pbtxt for initialization of variables
        run pbtxt for running graph.
        '''
        if len(os.listdir(os.environ["TF_DUMP_GRAPH_PREFIX"])) == 0:
            dump_pbtxts()
        else:
            logger().error("The directory %s contains pbtxt files \
                    before running scaler_graph." %
                           (os.environ["TF_DUMP_GRAPH_PREFIX"]))
            raise RuntimeError
        file_names = os.listdir(os.environ["TF_DUMP_GRAPH_PREFIX"])
        if len(file_names) == 0:
            logger().error(
                '''We cannot get the tensorflow graph from %s. Users should set \
TF_CPP_MIN_VLOG_LEVEL=3 before importing tensorflow, \
e.g.: ` TF_CPP_MIN_VLOG_LEVEL=3 python your_script.py`''' %
                (os.environ["TF_DUMP_GRAPH_PREFIX"]))
            raise RuntimeError
        input_pbtxts = []
        for file_name in file_names:
            obj = re.match(r"^placer_input(_\d+)?\.pbtxt$", file_name)
            if obj is not None:
                input_pbtxts.append(file_name)
        input_pbtxts.sort()
        if len(input_pbtxts) != 2:
            logger().error("Clean up the directory %s first." %
                           (os.environ["TF_DUMP_GRAPH_PREFIX"]))
            raise RuntimeError
        return os.environ["TF_DUMP_GRAPH_PREFIX"] + "/" + input_pbtxts[0], \
            os.environ["TF_DUMP_GRAPH_PREFIX"] + "/" + input_pbtxts[1]

    init_path, run_path = get_dumped_pbtxts()
    sc_graph = import_graph_from_tf_file(init_path, run_path)
    return sc_graph
