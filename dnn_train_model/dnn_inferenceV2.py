# -*- coding: utf-8 -*-
"""
@author: hbs
@date: 2018-9-11
description:
    神经网络的前向传播过程。相较于 dnn_inference.py加入了batch_normalization的部分。
    在每层输入数据至激励函数之前，将数据进行标准化。
@version: 2.0
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import sys
sys.path.append("/data/nanopore/dL_network/network/")
sys.path.append("/data/nanopore/dL_network/network/dnn_train_model/")
import tensorflow as tf
from config import *


def get_weight(shape, l1_regularizer=None, l2_regularizer=None):
    """
    在训练时可以创建变量，并将模型的正则化损失加入损失集合；在训练完成后可以加载变量。
    :param shape: 权重向量的维度
    :param l1_regularizer: l1正则化类对象
    :param l2_regularizer: l1正则化类对象
    :return:
    """
    # 获得初始化的权重向量
    # weights = tf.get_variable(name="weights", shape=shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
    weights = tf.get_variable(name="weights", shape=shape, initializer=tf.contrib.layers.xavier_initializer())

    # 给权重向量添加正则化优化
    if l1_regularizer is not None and l2_regularizer is not None:
        tf.add_to_collection("losses", l1_regularizer(weights) + l2_regularizer(weights))
    elif l1_regularizer is not None:
        tf.add_to_collection("losses", l1_regularizer(weights))
    elif l2_regularizer is not None:
        tf.add_to_collection("losses", l2_regularizer(weights))
    else:
        tf.add_to_collection("losses", tf.constant(0.0, dtype=tf.float32))
    return weights


def norm_op(wx_plus_b, out_size, ema, training):
    """
    batch_normalization的操作
    :param wx_plus_b: 线型拟合的结果
    :param out_size: 输出数据的维度，与下一层的节点数相等
    :param ema: 滑动平均类对象
    :param training: 布尔值，是否是训练过程
    :return:
    """
    fc_mean, fc_var = tf.nn.moments(wx_plus_b, axes=0)
    scale = tf.Variable(tf.ones([out_size]), trainable=True, name="bn_scale")
    shift = tf.Variable(tf.ones([out_size]), trainable=True, name="bn_shift")
    epsilon = tf.constant(0.001, dtype=tf.float32, name="eps")

    def mean_var_with_update():
        ema_apply_op = ema.apply([fc_mean, fc_var])
        with tf.control_dependencies([ema_apply_op]):
            return tf.identity(fc_mean), tf.identity(fc_var)
    if training:
        mean, var = mean_var_with_update()
        wx_plus_b = tf.nn.batch_normalization(wx_plus_b, mean, var, shift, scale, epsilon)
    else:
        mean, var = ema.average(fc_mean), ema.average(fc_var)
        wx_plus_b = tf.nn.batch_normalization(wx_plus_b, mean, var, shift, scale, epsilon)
    return wx_plus_b


def inference(input_tensor, l1_regularizer, l2_regularizer, ema, reuse=False):
    """
    定义神经网络的向前传播过程
    :param input_tensor: 传入batch_size个样本的特征向量
    :param l1_regularizer: 传入一个l1正则化类对象
    :param l2_regularizer: 传入一个l2正则化类对象
    :param ema: 一个指数滑动平均类变量
    :param reuse: 是否使用训练完成的权重参数，默认为False。只有在训练完成对未知样本进行预测时为True
    :return:
    """
    # 创建输入层到第一个隐藏层的传递
    with tf.variable_scope("layer1", reuse=reuse):#INPUT_NODE
        training = False if reuse else True
        weights = get_weight(shape=[INPUT_NODE, LAYER1_NODE], l1_regularizer=l1_regularizer, l2_regularizer=l2_regularizer)
        bias = tf.get_variable(name="bias1", shape=LAYER1_NODE, initializer=tf.contrib.layers.xavier_initializer())
        wx_plus_b = tf.matmul(input_tensor, weights) + bias
        norm = norm_op(wx_plus_b, LAYER1_NODE, ema, training)
        layer1 = tf.nn.relu(norm)

    # 创建第一个隐藏层到第二个隐藏层的传递
    with tf.variable_scope("layer2", reuse=reuse):
        training = False if reuse else True
        weights = get_weight(shape=[LAYER1_NODE, LAYER2_NODE], l1_regularizer=l1_regularizer, l2_regularizer=l2_regularizer)
        bias = tf.get_variable(name="bias2", shape=LAYER2_NODE, initializer=tf.keras.initializers.lecun_normal())
        wx_plus_b = tf.matmul(layer1, weights) + bias
        norm = norm_op(wx_plus_b, LAYER2_NODE, ema, training)
        layer2 = tf.nn.relu(norm)

    # 创建第二个隐藏层到输出层的传递
    with tf.variable_scope("output", reuse=reuse):
        weights = get_weight(shape=[LAYER2_NODE, OUTPUT_NODE], l1_regularizer=l1_regularizer, l2_regularizer=l2_regularizer)
        bias = tf.get_variable(name="bias3", shape=OUTPUT_NODE, initializer=tf.keras.initializers.lecun_normal())
        output = tf.matmul(layer2, weights) + bias

    return output


if __name__ == "__main__":
    test_tensor = tf.constant([[0.0, 3.0, 6.7, 3.7], [2.2, 3.5, 4.0, 3.2], [2.0, 3.0, 4.0, 3.3], [1.0, 4.0, 8.0, 5.5], [6.0, 5.0, 9.0, 4.4]])
    # test_output = inference(test_tensor, None, None)
    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        print(sess.run(tf.nn.softmax(test_output)))
