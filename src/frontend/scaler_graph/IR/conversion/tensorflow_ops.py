from frontend.scaler_graph.IR import operator
tf_op_map_to_sc_op = {
    # ElementWiseOp
    "Tanh": operator.ElementWiseOp,
    "Relu": operator.ElementWiseOp,
    "Sigmoid": operator.ElementWiseOp,
    # BinaryElementWiseOp
    "Add": operator.BinaryElementWiseOp,
    "Sub": operator.BinaryElementWiseOp,
    # ApplyOp
    "ApplyGradientDescent": operator.ApplyOp,
    # NoOp
    "NoOp": operator.NoOp,
}
