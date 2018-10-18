# -*- coding: utf-8 -*-
"""
@author: hbs
@date: 2018-9-15
description:
    cnn网络的训练过程。
"""
import os
import sys
from sklearn.metrics import roc_auc_score
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
sys.path.append("/data/nanopore/dL_network/network/cnn_cellflow_res")
sys.path.append("/data/nanopore/dL_network/network")
from load_data_cnn import CnnNanoporeDataset
import tensorflow as tf
import cnn_inference
from config import *
import numpy as np
import warnings


class CnnNanoporeModel:
    def __init__(self):
        pass

    @staticmethod
    def get_train_set(batch_size, num_epoch):
        """
        生成一个batch的训练数据
        :param batch_size: 指定batch_size的大小
        :param num_epoch: 训练过程的epoch大小
        :return:
        """
        cnd = CnnNanoporeDataset(DATASET, batch_size=batch_size, num_epoch=num_epoch)
        train = cnd.get_train_samples
        return train

    @staticmethod
    def get_test_set(batch_size):
        cnd = CnnNanoporeDataset(DATASET, batch_size=batch_size)
        test = cnd.get_test_samples
        return test

    @staticmethod
    def get_valid_set(batch_size):
        cnd = CnnNanoporeDataset(DATASET, batch_size=batch_size)
        valid = cnd.get_valid_samples
        return valid

    @staticmethod
    def train():
        x = tf.placeholder(dtype=tf.float32, shape=[None, NANOPORE_HEIGHT, NANOPORE_WIDTH, NANOPORE_CHANNEL], name="x")
        y = tf.placeholder(dtype=tf.float32, shape=[None, 2], name="y")

        # 获取训练、验证数据集
        train_set = CnnNanoporeModel.get_train_set(BATCH_SIZE, NUM_EPOCH)
        train_features, train_labels = train_set.get_next()

        valid_set = CnnNanoporeModel.get_valid_set(BATCH_SIZE)
        valid_features, valid_labels = valid_set.get_next()

        # 记录实验步数，创建滑动平均模型
        global_step = tf.Variable(initial_value=0, trainable=False)
        variable_average = tf.train.ExponentialMovingAverage(decay=MOVING_AVERAGE_DECAY, num_updates=global_step)

        # 前向传播过程输出预测结果
        y_pred = cnn_inference.inference(input_tensor=x, train=True, regularizer=None)
        valid_pred = cnn_inference.inference(input_tensor=x, train=False, regularizer=None, reuse=True)
        y_logit = tf.nn.softmax(y_pred, name="prediction")

        # 互熵损失
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(y, 1), logits=y_pred)
        cross_entropy_mean = tf.reduce_mean(cross_entropy)
        loss = cross_entropy_mean + tf.add_n(tf.get_collection("losses"))

        variable_average_op = variable_average.apply(var_list=tf.trainable_variables())

        lr = tf.train.exponential_decay(learning_rate=LEARNING_RATE_BASE, global_step=global_step,
                                        decay_steps=SAMPLE_NUM/BATCH_SIZE, decay_rate=LEARNING_RATE_DECAY)
        train_step = tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9).minimize(loss=loss, global_step=global_step)

        with tf.control_dependencies([train_step, variable_average_op]):
            train_op = tf.no_op(name="train")

        saver = tf.train.Saver()

        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            tf.local_variables_initializer().run()
            sess.run(train_set.initializer)

            max_auc = 0.5
            i = 1
            while i < 25000:
                try:
                    features, labels = sess.run([train_features, train_labels])

                    train_feed = {x: features, y: labels}
                    _, loss_val, step = sess.run([train_op, loss, global_step], feed_dict=train_feed)
                    # 每训练一百次进行一次验证
                    if i % 100 == 0:
                        validset_labels = []
                        validset_results = []
                        sess.run(valid_set.initializer)
                        while True:
                            try:
                                feats, labs = sess.run([valid_features, valid_labels])
                                valid_feed = {x: feats, y: labs}

                                valid_prediction = tf.nn.softmax(sess.run(valid_pred, feed_dict=valid_feed))
                                validset_labels.extend([i[0] for i in labs])
                                validset_results.extend([i[0] for i in valid_prediction.eval()])
                            except tf.errors.OutOfRangeError:
                                score = roc_auc_score(validset_labels, validset_results)
                                if score > max_auc:
                                    max_auc, score = score, max_auc
                                    print(i)
                                    print(max_auc)
                                    saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)
                                del validset_labels, validset_results
                                break
                    i += 1
                except tf.errors.OutOfRangeError:
                    break


def main(argv=None):
    CnnNanoporeModel.train()


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    tf.app.run()


