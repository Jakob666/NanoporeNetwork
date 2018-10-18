# -*- coding:utf-8 -*-
"""
@author: hbs
@date: 2018-9-15
description:
    CNN网络的向前传播过程。参考的是LeNet CNN网络的结构
"""
import os
import sys
import tensorflow as tf
from config import *
import numpy as np
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
sys.path.append("/data/nanopore/dL_network/network/cnn_train_model")


def inference(input_tensor, train, regularizer=None):
    """
    CNN网络的向前传播过程。
    :param input_tensor: 输入层输入的张量，维度 17×4×2
    :param train: boolean型变量，用于区分训练和测试过程
    :param regularizer: 正则化类对象
    :return:
    """
    if regularizer is None:
        tf.add_to_collection("losses", tf.constant(0.0, dtype=tf.float32))
    # 对输入数据先进行一波标准化操作（效果不好）
    # input_tensor = tf.contrib.layers.batch_norm(input_tensor, center=True, scale=True, is_training=train)
    # 第一个卷积层，深度32，窗口大小2×2
    with tf.variable_scope("layer1-conv1"):
        conv1_weights = tf.get_variable(name="weight", shape=[FLITER1_SIZE, FLITER1_SIZE, NANOPORE_CHANNEL, CONV1_DEEP],
                                        initializer=tf.contrib.layers.xavier_initializer())
        conv1_biases = tf.get_variable(name="bias", shape=[CONV1_DEEP], initializer=tf.contrib.layers.xavier_initializer())

        # 使用尺寸为2×2， 水平步长为1，垂直步长为2，不使用全0填充，最终得到 8×3×32 的输出结果
        conv1 = tf.nn.conv2d(input=input_tensor, filter=conv1_weights,
                             strides=[1, CONV1_STRIDE_VEC, CONV1_STRIDE_HOR, 1], padding="VALID")
        relu1 = tf.nn.relu(tf.nn.bias_add(value=conv1, bias=conv1_biases))

    # 第一个池化层，使用max_pooling，过滤器尺寸2×2
    with tf.variable_scope("layer2-pool1"):
        # 过滤器尺寸2×2，垂直步长为2，水平步长为1，使用边缘0填充，最终得到 4×3×32 的输出结果
        pool1 = tf.nn.max_pool(value=relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 1, 1], padding="SAME")

    # 第二个卷积层，过滤器尺寸2×2，深度64，水平步长和垂直步长均为2
    with tf.variable_scope("layer3-conv2"):
        conv2_weights = tf.get_variable(name="weight", shape=[FLITER2_SIZE, FLITER2_SIZE, CONV1_DEEP, CONV2_DEEP],
                                        initializer=tf.contrib.layers.xavier_initializer())
        conv2_biases = tf.get_variable(name="bias", shape=[CONV2_DEEP], initializer=tf.contrib.layers.xavier_initializer())

        # 过滤器尺寸2×2，垂直步长为2，水平步长为1，不使用0填充，最终得到 2×2×64 的输出结果
        conv2 = tf.nn.conv2d(input=pool1, filter=conv2_weights,
                             strides=[1, CONV2_STRIDE_HOR, CONV2_STRIDE_VEC, 1], padding="VALID")
        relu2 = tf.nn.relu(tf.nn.bias_add(value=conv2, bias=conv2_biases))

    # 第二个池化层
    with tf.variable_scope("layer4-pool2"):
        # 过滤器尺寸2×2，垂直步长为1，水平步长为1，不使用0填充，最终得到 1×1×64 的输出结果
        pool2 = tf.nn.max_pool(value=relu2, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding="VALID")

    # 在进入全连接层之前将卷积层输出的张量拉直成线型
    pool_shape = pool2.get_shape().as_list()    # 返回数组[batch_size, length, width, depth]
    node = pool_shape[1] * pool_shape[2] * pool_shape[3]
    # 通过tf.shape()[0]获取传入的样本的数目，pool_shape[0]得到的是None
    reshaped = tf.reshape(tensor=pool2, shape=[tf.shape(input=input_tensor)[0], node])

    # 第一个全连接层，输入的数据是 batch_size×64的维度的张量
    with tf.variable_scope("layer5-fc1"):
        fc1_weights = tf.get_variable(name="weight", shape=[node, FC_SIZE],
                                      initializer=tf.contrib.layers.xavier_initializer())
        fc1_biases = tf.get_variable(name="bias", shape=[FC_SIZE], initializer=tf.keras.initializers.lecun_normal())
        if regularizer is not None:
            tf.add_to_collection("losses", regularizer(fc1_weights))

        fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weights) + fc1_biases)
        # 引入dropout
        if train:
            fc1 = tf.nn.dropout(fc1, keep_prob=KEEP_PROB)

    # 第二个全连接层
    with tf.variable_scope("layer6-fc2"):
        fc2_weights = tf.get_variable(name="weight", shape=[FC_SIZE, OUTPUT_NODE],
                                      initializer=tf.contrib.layers.xavier_initializer())
        fc2_biases = tf.get_variable(name="bias", shape=[OUTPUT_NODE],
                                     initializer=tf.keras.initializers.lecun_normal())
        if regularizer is not None:
            tf.add_to_collection("losses", regularizer(fc2_weights))

        logit = tf.matmul(fc1, fc2_weights) + fc2_biases

    return logit
