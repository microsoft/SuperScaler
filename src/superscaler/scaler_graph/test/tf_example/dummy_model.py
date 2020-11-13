import tensorflow as tf


def SimpleCNN():
    images = tf.ones(shape=[128, 32, 32, 3], dtype=tf.float32, name="images")
    labels = tf.zeros(shape=[128], dtype=tf.float32, name="labels")
    NUM_CLASSES = 10
    # conv1
    kernel = tf.get_variable(
        name="conv1_weights",
        shape=[5, 5, 3, 64],
        initializer=tf.initializers.ones(dtype=tf.float32),
        dtype=tf.float32)
    conv = tf.nn.conv2d(images,
                        kernel, [1, 1, 1, 1],
                        padding="SAME",
                        name="conv1_matmul")
    biases = tf.get_variable(name="conv1_biases",
                             shape=[64],
                             initializer=tf.constant_initializer(0.0))
    pre_activation = tf.nn.bias_add(conv, biases, name="conv1_add")
    conv1 = tf.nn.relu(pre_activation, name="conv1_relu")
    # pool1
    pool1 = tf.nn.max_pool(conv1,
                           ksize=[1, 3, 3, 1],
                           strides=[1, 2, 2, 1],
                           padding="SAME",
                           name="pool1")
    # norm1
    norm1 = tf.nn.lrn(pool1,
                      4,
                      bias=1.0,
                      alpha=0.001 / 9.0,
                      beta=0.75,
                      name="norm1")

    # conv2
    kernel = tf.get_variable(
        name="conv2_weights",
        shape=[5, 5, 64, 64],
        initializer=tf.initializers.ones(dtype=tf.float32),
        dtype=tf.float32)
    conv = tf.nn.conv2d(norm1,
                        kernel, [1, 1, 1, 1],
                        padding="SAME",
                        name="conv2_matmul")
    biases = tf.get_variable(name="conv2_biases",
                             shape=[64],
                             initializer=tf.constant_initializer(0.1),
                             dtype=tf.float32)
    pre_activation = tf.nn.bias_add(conv, biases, name="conv2_add")
    conv2 = tf.nn.relu(pre_activation, name="conv2_relu")
    # norm2
    norm2 = tf.nn.lrn(conv2,
                      4,
                      bias=1.0,
                      alpha=0.001 / 9.0,
                      beta=0.75,
                      name="norm2")
    # pool2
    pool2 = tf.nn.max_pool(norm2,
                           ksize=[1, 3, 3, 1],
                           strides=[1, 2, 2, 1],
                           padding="SAME",
                           name="pool2")

    # dense3
    reshape = tf.reshape(pool2, [images.get_shape().as_list()[0], -1],
                         name="reshape")
    dim = reshape.get_shape()[1].value
    weights = tf.get_variable(
        name="dense3_weights",
        shape=[dim, 384],
        initializer=tf.initializers.ones(dtype=tf.float32),
        dtype=tf.float32)
    biases = tf.get_variable(name="dense3_biases",
                             shape=[384],
                             initializer=tf.constant_initializer(0.1),
                             dtype=tf.float32)
    dense = tf.matmul(reshape, weights, name="dense3_matmul")
    pre_activation = tf.nn.bias_add(dense, biases, name="dense3_add")
    dense3 = tf.nn.relu(pre_activation, name="dense3_relu")
    # dense4
    weights = tf.get_variable(
        name="dense4_weights",
        shape=[384, 192],
        initializer=tf.initializers.ones(dtype=tf.float32),
        dtype=tf.float32)
    biases = tf.get_variable(name="dense4_biases",
                             shape=[192],
                             initializer=tf.constant_initializer(0.1),
                             dtype=tf.float32)
    dense = tf.matmul(dense3, weights, name="dense4_matmul")
    pre_activation = tf.nn.bias_add(dense, biases, name="dense4_add")
    dense4 = tf.nn.relu(pre_activation, name="dense4_relu")
    # softmax_linear, actually linear layer(WX + b),
    weights = tf.get_variable(
        name="softmax_linear_weights",
        shape=[192, NUM_CLASSES],
        initializer=tf.initializers.ones(dtype=tf.float32),
        dtype=tf.float32)
    biases = tf.get_variable(name="softmax_linear_biases",
                             shape=[NUM_CLASSES],
                             initializer=tf.constant_initializer(0.0),
                             dtype=tf.float32)
    softmax_linear = tf.matmul(dense4, weights, name="softmax_matmul")
    softmax_linear = tf.nn.bias_add(softmax_linear, biases, name="softmax_add")

    labels = tf.cast(labels, tf.int64, name="labels_cast")
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=softmax_linear, name="cross_entropy_per_example")
    loss = tf.reduce_mean(cross_entropy, name="cross_entropy")
    opt = tf.train.GradientDescentOptimizer(0.123456)  # lr
    grads = opt.compute_gradients(loss)
    apply_gradient_op = opt.apply_gradients(grads)
    return apply_gradient_op, loss
