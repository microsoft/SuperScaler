# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import tensorflow as tf
from superscaler import TFRecordDataset

LOCAL_BATCH_SIZE = 128


def model():
    keys_to_features = {
        'image/encoded':
        tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format':
        tf.FixedLenFeature((), tf.string, default_value='png'),
        'image/height':
        tf.FixedLenFeature(
            [],
            tf.int64,
        ),
        'image/width':
        tf.FixedLenFeature([], tf.int64),
        'image/class/label':
        tf.FixedLenFeature([],
                           tf.int64,
                           default_value=tf.zeros([], dtype=tf.int64)),
    }

    def _decode_image(features):
        features = tf.io.parse_single_example(features, keys_to_features)
        data = features['image/encoded']
        png_image = tf.io.decode_png(data, 3, tf.uint8)
        image = tf.cast(png_image, tf.float32)
        image /= 255.0
        image = tf.reshape(image, [32, 32, 3])
        label = features['image/class/label']
        return image, label

    dataset = TFRecordDataset("")
    dataset = dataset.map(_decode_image)
    dataset = dataset.repeat(count=10)
    dataset = dataset.batch(LOCAL_BATCH_SIZE, drop_remainder=True)
    dataset = dataset.cache()
    iterator = dataset.make_initializable_iterator()
    data, labels = iterator.get_next()
    # model
    NUM_CLASSES = 10
    # conv1
    kernel = tf.Variable(tf.random_normal([5, 5, 3, 64]))
    conv = tf.nn.conv2d(data,
                        kernel, [1, 1, 1, 1],
                        padding="SAME",
                        name="conv1_matmul")
    biases = tf.Variable(tf.random_normal([64]))
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
    kernel = tf.Variable(tf.random_normal([5, 5, 64, 64]))
    conv = tf.nn.conv2d(norm1,
                        kernel, [1, 1, 1, 1],
                        padding="SAME",
                        name="conv2_matmul")
    biases = tf.Variable(tf.random_normal([64]))
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
    reshape = tf.reshape(pool2, [data.get_shape().as_list()[0], -1],
                         name="reshape")
    dim = reshape.get_shape()[1].value
    weights = tf.Variable(tf.random_normal([dim, 384]))
    biases = tf.Variable(tf.random_normal([384]))
    dense = tf.matmul(reshape, weights, name="dense3_matmul")
    pre_activation = tf.nn.bias_add(dense, biases, name="dense3_add")
    dense3 = tf.nn.relu(pre_activation, name="dense3_relu")
    # dense4
    weights = tf.Variable(tf.random_normal([384, 192]))
    biases = tf.Variable(tf.random_normal([192]))
    dense = tf.matmul(dense3, weights, name="dense4_matmul")
    pre_activation = tf.nn.bias_add(dense, biases, name="dense4_add")
    dense4 = tf.nn.relu(pre_activation, name="dense4_relu")
    # softmax_linear, actually linear layer(WX + b),
    weights = tf.Variable(tf.random_normal([192, NUM_CLASSES]))
    biases = tf.Variable(tf.random_normal([NUM_CLASSES]))
    softmax_linear = tf.matmul(dense4, weights, name="softmax_matmul")
    softmax_linear = tf.nn.bias_add(softmax_linear, biases, name="softmax_add")
    labels = tf.cast(labels, tf.int64, name="labels_cast")
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=softmax_linear, name="cross_entropy_per_example")
    loss = tf.reduce_mean(cross_entropy, name="cross_entropy")
    opt = tf.train.GradientDescentOptimizer(0.123456)  # lr
    grads = opt.compute_gradients(loss)
    apply_gradient_op = opt.apply_gradients(grads)
    session_run_params = {
        "init_params":
        [iterator.initializer,
         tf.global_variables_initializer()],
        "run_params": [apply_gradient_op, loss]
    }
    return session_run_params


def get_dataset_paths():
    """
    dictionary format:
    {
        "ip": [
            [datasetA_path1,datasetA_path2...],
            [datasetB_path1,datasetB_path2...],
            ...
        ]
        ...
    }
    """
    dataset_paths = {
        "localhost": [
            ["/tmp/sc_test/cifar10/cifar10_train_0.tfrecord"],
            ["/tmp/sc_test/cifar10/cifar10_train_1.tfrecord"],
        ]
    }
    return dataset_paths
