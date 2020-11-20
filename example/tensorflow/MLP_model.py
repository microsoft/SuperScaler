import tensorflow as tf


def MLP():
    # fake data
    data = tf.ones(shape=[32, 784], dtype=tf.float32)
    labels = tf.zeros(shape=[32, 10], dtype=tf.float32)
    # MLP model
    layer_1 = tf.add(
        tf.matmul(data, tf.Variable(tf.random_normal([784, 256]))),
        tf.Variable(tf.random_normal([256])))
    layer_2 = tf.add(
        tf.matmul(layer_1, tf.Variable(tf.random_normal([256, 256]))),
        tf.Variable(tf.random_normal([256])))
    logits = tf.add(
        tf.matmul(layer_2, tf.Variable(tf.random_normal([256, 10]))),
        tf.Variable(tf.random_normal([10])))
    # optimizer
    loss_op = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    optimizer = tf.train.GradientDescentOptimizer(0.001)
    train_op = optimizer.minimize(loss_op)
    return train_op, loss_op
