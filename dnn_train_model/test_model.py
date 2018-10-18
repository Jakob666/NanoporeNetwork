# -*- coding:utf-8 -*-
import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
import numpy as np
from config import *
import dnn_inference
import tensorflow as tf
from sklearn.metrics import roc_auc_score


def get_samples():
    positive_feature = np.random.normal(loc=2, scale=1.5, size=(5, 14))
    positive_label = np.ones(shape=[5, 2], dtype=np.int32)
    positive_label[:, -1] = 0
    positive_samples = np.concatenate((positive_feature, positive_label), axis=1)
    negative_feature = np.random.normal(loc=-3, scale=2, size=(5, 14))
    negative_label = np.zeros(shape=[5, 2], dtype=np.int32)
    negative_label[:, -1] = 1
    negative_samples = np.concatenate((negative_feature, negative_label), axis=1)
    samples = np.append(positive_samples, negative_samples)
    samples = samples.reshape((10, 16))
    np.random.shuffle(samples)
    return samples


def train():
    # 定义输入层输入数据
    x = tf.placeholder(dtype=tf.float32, shape=[None, INPUT_NODE], name="x-input")
    # 定义输入样本的真实标签
    y_real = tf.placeholder(dtype=tf.float32, shape=[None, OUTPUT_NODE], name="y_input")

    # 创建正则化类对象
    # l1_regularizer = tf.contrib.layers.l1_regularizer(REGULARIZATION_RATE)
    l2_regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)

    # 将特征数据输入向前传播过程，得到未经过softmax分类的神经网络输出值
    y_pred = dnn_inference.inference(x, l1_regularizer=None, l2_regularizer=None)

    global_step = tf.Variable(0, trainable=False)

    # 创建滑动平均类对象及该对象的优化方式，该对象对整个模型需要训练的参数进行训练
    variable_averages = tf.train.ExponentialMovingAverage(decay=MOVING_AVERAGE_DECAY, num_updates=global_step)
    variable_averages_op = variable_averages.apply(tf.trainable_variables())

    # 使用互熵损失作为损失函数并计算每个batch的平均互熵损失
    # pred_label = tf.nn.softmax(logits=y_pred)
    pred_label = tf.nn.softmax(y_pred)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(y_real, 1), logits=y_pred)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    # 定义总损失 总损失 = 互熵损失 + 正则化损失（定义在dnn_inference中）
    loss = cross_entropy_mean + tf.add_n(tf.get_collection("losses"))

    # 在学习率方面设置指数递减学习率
    lr = tf.train.exponential_decay(learning_rate=LEARNING_RATE_BASE, global_step=global_step,
                                    decay_steps=SAMPLE_NUM / BATCH_SIZE, decay_rate=LEARNING_RATE_DECAY)

    # 基于上方的设定生成最终的学习过程，同时将train和求取获得平均的过程按如下顺序绑定
    train_step = tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9).minimize(loss=loss, global_step=global_step)
    with tf.control_dependencies([train_step, variable_averages_op]):
        train_op = tf.no_op(name="train")

    # # 计算正确率
    # correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_real, 1))
    # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # 设置用于保存模型的saver
    # saver = tf.train.Saver()

    # 开启训练过程
    with tf.Session() as sess:
        # 初始化所有变量
        tf.global_variables_initializer().run()
        min_loss = 1
        for i in range(1000):
            samples = get_samples()
            train_feature = samples[:, : -2]
            train_label = samples[:, -2:]
            _, loss_val, step, pred_labels = sess.run([train_op, loss, global_step, pred_label], feed_dict={x: train_feature, y_real: train_label})

            # 每100轮训练保存一次模型
            if i % 100 == 0:
                roc = roc_auc_score(train_label, pred_labels)
                print("the training at step %d auc=%.4f" % (i, roc))
            i += 1
    # print("the best training at step %d losses=%.4f" % (training_steps, min_loss))


def main(argv=None):
    train()


if __name__ == "__main__":
    tf.app.run()
