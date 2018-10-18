# -*- coding:utf-8 -*-
"""
@author: hbs
@date: 2018-9-8
description:
    神经网络的前向传播过程。
@version: 1.0
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
    :param l2_regularizer: l2正则化类对象
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


def inference(input_tensor, l1_regularizer, l2_regularizer, reuse=False):
    """
    定义神经网络的向前传播过程
    :param input_tensor: 传入batch_size个样本的特征向量
    :param l1_regularizer: 传入一个l1正则化类对象
    :param l2_regularizer: 传入一个l2正则化类对象
    :param reuse
    :return:
    """
    # 创建输入层到第一个隐藏层的传递
    with tf.variable_scope("layer1", reuse=reuse):
        weights = get_weight(shape=[INPUT_NODE, LAYER1_NODE], l1_regularizer=l1_regularizer, l2_regularizer=l2_regularizer)
        bias = tf.get_variable(name="bias1", shape=LAYER1_NODE, initializer=tf.contrib.layers.xavier_initializer())
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights) + bias)

    # 创建第一个隐藏层到第二个隐藏层的传递
    with tf.variable_scope("layer2", reuse=reuse):
        weights = get_weight(shape=[LAYER1_NODE, LAYER2_NODE], l1_regularizer=l1_regularizer, l2_regularizer=l2_regularizer)
        bias = tf.get_variable(name="bias2", shape=LAYER2_NODE, initializer=tf.keras.initializers.lecun_normal())
        layer2 = tf.nn.relu(tf.matmul(layer1, weights) + bias)

    # with tf.variable_scope("layer3", reuse=reuse):
    #     weights = get_weight(shape=[LAYER2_NODE, 10], l1_regularizer=l1_regularizer, l2_regularizer=l2_regularizer)
    #     bias = tf.get_variable(name="bias3", shape=10, initializer=tf.keras.initializers.lecun_normal())
    #     layer3 = tf.nn.relu(tf.matmul(layer2, weights) + bias)

    # 创建第二个隐藏层到输出层的传递
    with tf.variable_scope("output", reuse=reuse):
        weights = get_weight(shape=[LAYER2_NODE, OUTPUT_NODE], l1_regularizer=l1_regularizer, l2_regularizer=l2_regularizer)
        bias = tf.get_variable(name="bias3", shape=OUTPUT_NODE, initializer=tf.keras.initializers.lecun_normal())
        output = tf.matmul(layer2, weights) + bias

    return output

