from frontend.scaler_graph.IR import operator
from frontend.scaler_graph.util.log import logger


def parse_info(sc_op, tf_op_def):
    '''parse op info when importing graphs from other frameworks.
    '''
    if isinstance(sc_op, operator.ApplyOp):
        for index, input_arg in enumerate(tf_op_def.input_arg):
            if input_arg.name == "var":
                sc_op.info["parameter_index"] = index
            elif input_arg.name == "grad" or input_arg.name == "delta":
                sc_op.info["gradient_index"] = index
            elif input_arg.is_ref:
                pass
    else:
        logger("TF_conversion").info(
            "op: %s of tensorflow uses default operator, OpDef structure:\n%s"
            % (tf_op_def.name, tf_op_def))


def tf_op_map_to_sc_op(tf_op_def):
    '''mapping tf op to sc op.
    We need to define some specific ops for superscaler, e.g. conv2d, matmul.
    Trival ops could be mapped to general sc op.
    '''
    mapping = {
        # ElementWiseOp
        "Tanh": operator.ElementWiseOp,
        "Relu": operator.ElementWiseOp,
        "Sigmoid": operator.ElementWiseOp,
        # BinaryElementWiseOp
        "Add": operator.BinaryElementWiseOp,
        "Sub": operator.BinaryElementWiseOp,
        # ApplyOp
        "ApplyGradientDescent": operator.ApplyOp,
        "ResourceApplyGradientDescent": operator.ApplyOp,
        "ApplyProximalGradientDescent": operator.ApplyOp,
        "SparseApplyProximalGradientDescent": operator.ApplyOp,
        "ResourceApplyProximalGradientDescent": operator.ApplyOp,
        "ResourceSparseApplyProximalGradientDescent": operator.ApplyOp,
        "ApplyAdadelta": operator.ApplyOp,
        "SparseApplyAdadelta": operator.ApplyOp,
        "ResourceApplyAdadelta": operator.ApplyOp,
        "ResourceSparseApplyAdadelta": operator.ApplyOp,
        "ApplyAdagrad": operator.ApplyOp,
        "ResourceApplyAdagrad": operator.ApplyOp,
        "ApplyAdagradV2": operator.ApplyOp,
        "ResourceApplyAdagradV2": operator.ApplyOp,
        "ApplyProximalAdagrad": operator.ApplyOp,
        "ResourceApplyProximalAdagrad": operator.ApplyOp,
        "SparseApplyAdagrad": operator.ApplyOp,
        "ResourceSparseApplyAdagrad": operator.ApplyOp,
        "SparseApplyAdagradV2": operator.ApplyOp,
        "ResourceSparseApplyAdagradV2": operator.ApplyOp,
        "ApplyAdagradDA": operator.ApplyOp,
        "SparseApplyAdagradDA": operator.ApplyOp,
        "SparseApplyProximalAdagrad": operator.ApplyOp,
        "ResourceApplyAdagradDA": operator.ApplyOp,
        "ResourceSparseApplyAdagradDA": operator.ApplyOp,
        "ResourceSparseApplyProximalAdagrad": operator.ApplyOp,
        "ApplyFtrl": operator.ApplyOp,
        "SparseApplyFtrl": operator.ApplyOp,
        "ResourceApplyFtrl": operator.ApplyOp,
        "ResourceSparseApplyFtrl": operator.ApplyOp,
        "ApplyFtrlV2": operator.ApplyOp,
        "SparseApplyFtrlV2": operator.ApplyOp,
        "ResourceApplyFtrlV2": operator.ApplyOp,
        "ResourceSparseApplyFtrlV2": operator.ApplyOp,
        "ApplyMomentum": operator.ApplyOp,
        "SparseApplyMomentum": operator.ApplyOp,
        "ResourceApplyMomentum": operator.ApplyOp,
        "ResourceSparseApplyMomentum": operator.ApplyOp,
        "ResourceApplyKerasMomentum": operator.ApplyOp,
        "ResourceSparseApplyKerasMomentum": operator.ApplyOp,
        "ApplyAdam": operator.ApplyOp,
        "ResourceApplyAdam": operator.ApplyOp,
        "ResourceApplyAdamWithAmsgrad": operator.ApplyOp,
        "ApplyAdaMax": operator.ApplyOp,
        "ResourceApplyAdaMax": operator.ApplyOp,
        "ApplyRMSProp": operator.ApplyOp,
        "ApplyCenteredRMSProp": operator.ApplyOp,
        "SparseApplyRMSProp": operator.ApplyOp,
        "SparseApplyCenteredRMSProp": operator.ApplyOp,
        "ResourceApplyRMSProp": operator.ApplyOp,
        "ResourceApplyCenteredRMSProp": operator.ApplyOp,
        "ResourceSparseApplyRMSProp": operator.ApplyOp,
        "ResourceSparseApplyCenteredRMSProp": operator.ApplyOp,
        "ApplyAddSign": operator.ApplyOp,
        "ResourceApplyAddSign": operator.ApplyOp,
        "ApplyPowerSign": operator.ApplyOp,
        "ResourceApplyPowerSign": operator.ApplyOp,
        # NoOp
        "NoOp": operator.NoOp,
    }
    sc_op = mapping.get(tf_op_def.name, operator.Operator)(tf_op_def.name)
    parse_info(sc_op, tf_op_def)
    return sc_op


def convert_to_tf_node(sc_node):
    if sc_node.op.name == "AllreduceOp" and sc_node.op.original_name is None:
        sc_node.attrs["tf"] = {}
        sc_node.attrs["tf"]["device"] = ""
        sc_node.op.original_name = "_SCAllReduce"
        return True
    return False
