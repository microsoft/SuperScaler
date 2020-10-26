import google.protobuf.text_format
import os
import re
from pathlib import Path
from frontend.scaler_graph.IR.graph import Graph
from frontend.scaler_graph.IR import operator
from frontend.scaler_graph.IR.conversion.tensorflow_ops \
    import tf_op_map_to_sc_op
from tensorflow.python import types_pb2, tensor_shape
from tensorflow.core.framework import tensor_pb2
from tensorflow.core.framework.op_def_pb2 import OpDef
import tensorflow as tf
from tensorflow.python.util import compat
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


def get_dtype_proto(node_def, op_def, output_arg):
    def with_number_attr(dtype):
        if len(output_arg.number_attr) != 0:
            for attr in op_def.attr:
                if attr.name == output_arg.number_attr:
                    return [dtype] * node_def.attr[attr.name].i
            raise AssertionError()
        else:
            return dtype

    if len(output_arg.type_attr) != 0:
        for attr in op_def.attr:
            if attr.name == output_arg.type_attr:
                return with_number_attr(node_def.attr[attr.name].type)
        raise AssertionError()
    elif len(output_arg.type_list_attr) != 0:
        for attr in op_def.attr:
            if attr.name == output_arg.type_list_attr:
                return list(node_def.attr[attr.name].list.type)
        raise AssertionError()
    else:
        assert output_arg.type != types_pb2.DT_INVALID
        return with_number_attr(output_arg.type)


def get_dtypes(tf_graph, node_def):
    op_def = tf_graph._get_op_def(node_def.op)
    dtypes = [
        get_dtype_proto(node_def, op_def, output_arg)
        for output_arg in op_def.output_arg
    ]
    if len(dtypes) == 1 and isinstance(dtypes[0], list):
        dtypes = dtypes[0]
    return [tf.as_dtype(dtype) for dtype in dtypes]


def from_attr_proto(attr_value):
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


def import_graph_from_tf_file(pbtxt_path):
    tf_graph_def = tf.GraphDef()
    google.protobuf.text_format.Merge(
        Path(pbtxt_path).read_text(), tf_graph_def)

    sc_graph = Graph()
    tf_graph = tf.Graph()
    name_to_node = {node.name: node for node in tf_graph_def.node}

    def add_sc_node(tf_node: tf.NodeDef):
        if sc_graph.get_node_by_name(node.name) is not None:
            return
        input_node_idxes = []
        for input in tf_node.input:
            if input.startswith("^"):
                input_node_name = input[1:]
                if sc_graph.get_node_by_name(input_node_name) is not None:
                    input_node = sc_graph.get_node_by_name(input_node_name)
                else:
                    add_sc_node(name_to_node[input_node_name])
                    input_node = sc_graph.get_node_by_name(input_node_name)
                index = -1
            else:
                names = input.split(":")
                assert len(names) == 1 or len(names) == 2
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
            tf_node.name,
            tf_op_map_to_sc_op.get(tf_node.op, operator.Operator)(tf_node.op),
            input_node_idxes, len(dtypes), attrs)
        sc_node.attrs["tf"] = {}
        sc_node.attrs["tf"]["device"] = tf_node.device
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


def sc_attrs_to_tf_attrs_proto(op_def, op_type_name, attrs):
    # Convert attr values to AttrValue protos.
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
                raise AssertionError(f"{value} has unsupported type")
        if attr_def.HasField("default_value") and value is None:
            attr_value.CopyFrom(attr_def.default_value)
            attr_protos[key] = attr_value
            continue
        if attr_def.type.startswith("list("):
            if not _IsListValue(value):
                raise TypeError("Expected list for attr " + key)
            if attr_def.has_minimum:
                if len(value) < attr_def.minimum:
                    raise ValueError(
                        "Attr '%s' of '%s' Op passed list of length %d "
                        "less than minimum %d." %
                        (key, op_type_name, len(value), attr_def.minimum))
            attr_value.list.SetInParent()
        if attr_def.type == "string":
            attr_value.s = _MakeStr(value, key)
            if attr_def.HasField("allowed_values"):
                if attr_value.s not in attr_def.allowed_values.list.s:
                    raise ValueError(
                        "Attr '%s' of '%s' Op passed string '%s' not \
                            in: \"%s\"." % (
                            key,
                            op_type_name,
                            compat.as_text(attr_value.s),
                            '", "'.join(
                                map(compat.as_text,
                                    attr_def.allowed_values.list.s)),
                        ))
        elif attr_def.type == "list(string)":
            attr_value.list.s.extend([_MakeStr(x, key) for x in value])
            if attr_def.HasField("allowed_values"):
                for x in attr_value.list.s:
                    if x not in attr_def.allowed_values.list.s:
                        raise ValueError(
                            "Attr '%s' of '%s' Op passed string '%s' not \
                                in: \"%s\"." % (
                                key,
                                op_type_name,
                                compat.as_text(x),
                                '", "'.join(
                                    map(compat.as_text,
                                        attr_def.allowed_values.list.s)),
                            ))
        elif attr_def.type == "int":
            attr_value.i = _MakeInt(value, key)
            if attr_def.has_minimum:
                if attr_value.i < attr_def.minimum:
                    raise ValueError(
                        "Attr '%s' of '%s' Op passed %d less than minimum %d."
                        % (key, op_type_name, attr_value.i, attr_def.minimum))
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
            raise TypeError("Unrecognized Attr type " + attr_def.type)

        attr_protos[key] = attr_value
    return attr_protos


def export_to_graph_def_file(sc_graph, file_path=None):
    tf_graph = tf.Graph()
    graph_def = tf_graph.as_graph_def()
    for key in ["versions", "library"]:
        if key in sc_graph.attrs:
            getattr(graph_def, key).CopyFrom(sc_graph.attrs[key])
    for sc_node in sc_graph.nodes:
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

    graph_pbtxt = google.protobuf.text_format.MessageToString(graph_def)
    if file_path is not None:
        file = Path(file_path)
        file.write_text(graph_pbtxt)
    return graph_pbtxt


def import_tensorflow_model(apply_gradient_op, loss):
    def dump_pbtxts():
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
        if len(os.listdir(os.environ["TF_DUMP_GRAPH_PREFIX"])) == 0:
            dump_pbtxts()
        file_names = os.listdir(os.environ["TF_DUMP_GRAPH_PREFIX"])
        if len(file_names) == 0:
            raise Exception(r'''
                We cannot dump the tensorflow graph under TF_DUMP_GRAPH_PREFIX  
                Users should set TF_DUMP_GRAPH_PREFIX=path_to_empty_directory and TF_CPP_MIN_VLOG_LEVEL=4 before import tensorflow.
                e.g.:
                    TF_DUMP_GRAPH_PREFIX=path_to_empty_directory TF_CPP_MIN_VLOG_LEVEL=4 python your_script.py
                ''')
        id_file = {}
        for file_name in file_names:
            obj = re.match(r"^placer_input(_(\d+))?\.pbtxt$", file_name)
            if obj is not None:
                if obj.group(2) is None:
                    id_file[0] = file_name
                else:
                    id_file[int(obj.group(2))] = file_name
        if len(id_file) != 2:
            raise Exception(
                "Clean up the directory TF_DUMP_GRAPH_PREFIX first.")
        return os.environ["TF_DUMP_GRAPH_PREFIX"] + "/" + id_file[0], \
            os.environ["TF_DUMP_GRAPH_PREFIX"] + "/" + id_file[1]

    init_path, run_path = get_dumped_pbtxts()
    sc_graph_init = import_graph_from_tf_file(init_path)
    sc_graph_run = import_graph_from_tf_file(run_path)
    return sc_graph_init, sc_graph_run
