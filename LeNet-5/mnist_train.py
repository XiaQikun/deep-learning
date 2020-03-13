import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_inference
import os
import numpy as np

# 配置神经网络参数
BATCH_SIZE = 100  # 批处理数据大小
LEARNING_RATE_BASE = 0.001  # 基础学习率
LEARNING_RATE_DECAY = 0.99  # 学习率衰减速度
REGULARIZATION_RATE = 0.0001  # 正则化项
TRAINING_STEPS = 3000  # 训练次数
MOVING_AVERAGE_DECAY = 0.99  # 平均滑动模型衰减参数
# 模型保存的路径和文件名
MODEL_SAVE_PATH = "MNIST_model/"
MODEL_NAME = "mnist_model"


def train(mnist):
    # 定义输入输出placeholder
    x = tf.placeholder(tf.float32, [BATCH_SIZE, mnist_inference.IMAGE_SIZE,
                                    mnist_inference.IMAGE_SIZE, mnist_inference.NUM_CHANNELS], name='x-input')  # 可以直接引用mnist_inference中的超参数
    y_ = tf.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NODE], name='y-input')
    # 定义L2正则化器
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    # 在前向传播时使用L2正则化
    y = mnist_inference.inference(x, regularizer, regularizer)
    global_step = tf.Variable(0, trainable=False)
    # 在可训练参数上定义平均滑动模型
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    # tf.trainable_variables()返回的是图上集合GraphKeys.TRAINABLE_VARIABLES中的元素。这个集合中的元素是所有没有指定trainable=False的参数
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    # 在交叉熵函数的基础上增加权值的L2正则化部分
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    # 设置学习率，其中学习率使用逐渐递减的原则
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        mnist.train.num_examples/BATCH_SIZE, LEARNING_RATE_DECAY,
        staircase=True)
    # 使用梯度下降优化器
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    # with tf.control_dependencies([train_step, variables_averages_op]):
    # train_op = tf.no_op(name='train')
    # 在反向传播的过程中，不仅更新神经网络中的参数还更新每一个参数的滑动平均值
    train_op = tf.group(train_step, variables_averages_op)
    #     # 定义Saver模型保存器
    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        for i in range(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            reshaped_xs = np.reshape(xs, [BATCH_SIZE, mnist_inference.IMAGE_SIZE, mnist_inference.IMAGE_SIZE, mnist_inference.NUM_CHANNELS])
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: reshaped_xs, y_: ys})
            # 每1000轮保存一次模型
            if i % 100 == 0:
                # 输出当前的训练情况，这里只输出了模型在当前训练batch上的损失函数大小
                # 通过损失函数的大小可以大概了解训练的情况，
                # 在验证数据集上的正确率信息会有一个单独的程序来生成
                print("After %d training step(s), loss on training batch is %g." % (step, loss_value))
                # 模型保存
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)


def main(argv=None):
    mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
    train(mnist)


if __name__ == '__main__':
    tf.app.run()
