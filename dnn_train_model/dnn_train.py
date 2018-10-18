# -*- coding: utf-8 -*-
"""
@author: hbs
@date: 2018-9-6
description:
    用于对Nanopore的数据集进行训练，当前脚本使用DNN神经网络。
"""
import os
import warnings
import logging
import logging.handlers
import logging.config
import yaml
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import sys
sys.path.append("/data/nanopore/dL_network/network/")
sys.path.append("/data/nanopore/dL_network/network/dnn_train_model/")
from load_dataset import NanoporeDataset
from config import *
import dnn_inferenceV2
import tensorflow as tf
import numpy as np
from sklearn.metrics import roc_auc_score


class DnnNanoporeModel:
    @staticmethod
    def get_train_data(batch_size, num_epoch):
        """
        生成一个batch的训练数据
        :param batch_size: 指定batch_size的大小
        :param num_epoch: 训练过程的epoch大小
        :return:
        """
        nd = NanoporeDataset(DATASET, batch_size=batch_size, num_epoch=num_epoch)
        train = nd.get_train_samples
        return train

    @staticmethod
    def get_test_data(batch_size):
        nd = NanoporeDataset(DATASET, batch_size=batch_size)
        test = nd.get_test_samples
        return test

    @staticmethod
    def get_valid_data(batch_size):
        nd = NanoporeDataset(DATASET, batch_size=batch_size)
        valid = nd.get_valid_samples
        return valid

    @staticmethod
    def setup_logging():
        with open(LOGGING_CONF, "r") as f:
            config = yaml.load(f)
            logging.config.dictConfig(config)
        logger = logging.getLogger("training_process")
        return logger

    @staticmethod
    def train():
        x = tf.placeholder(dtype=tf.float32, shape=[None, 14], name="x")
        y = tf.placeholder(dtype=tf.float32, shape=[None, 2], name="y")

        # 获取训练数据集
        train_set = DnnNanoporeModel.get_train_data(BATCH_SIZE, NUM_EPOCH)
        train_features, train_labels = train_set.get_next()

        valid_set = DnnNanoporeModel.get_valid_data(BATCH_SIZE)
        valid_features, valid_labels = valid_set.get_next()

        test_set = DnnNanoporeModel.get_test_data(BATCH_SIZE)
        test_features, test_labels = test_set.get_next()

        global_step = tf.Variable(0, trainable=False)

        # 创建滑动平均类对象及该对象的优化方式，该对象对整个模型需要训练的参数进行训练
        variable_averages = tf.train.ExponentialMovingAverage(decay=MOVING_AVERAGE_DECAY, num_updates=global_step)

        # 创建正则化类对象
        l1_regularizer = tf.contrib.layers.l1_regularizer(REGULARIZATION_RATE)
        l2_regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)

        # 将特征数据输入向前传播过程，得到未经过softmax分类的神经网络输出值
        y_pred = dnn_inferenceV2.inference(x, l1_regularizer=None, l2_regularizer=None,
                                           ema=variable_averages)
        y_logit = tf.nn.softmax(y_pred, name="prediction")
        # valid_logit = dnn_inferenceV2.inference(x, None, None, variable_averages, reuse=True)

        # 使用互熵损失作为损失函数并计算每个batch的平均互熵损失
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(y, 1), logits=y_pred)
        cross_entropy_mean = tf.reduce_mean(cross_entropy)

        # 定义总损失 总损失 = 互熵损失 + 正则化损失（定义在dnn_inference中）
        loss = cross_entropy_mean + tf.add_n(tf.get_collection("losses"))

        variable_averages_op = variable_averages.apply(tf.trainable_variables())
        # 在学习率方面设置指数递减学习率（建议在初始设计网络的时候先关闭，以寻找最优学习率的值）
        lr = tf.train.exponential_decay(learning_rate=LEARNING_RATE_BASE, global_step=global_step,
                                        decay_steps=SAMPLE_NUM / BATCH_SIZE, decay_rate=LEARNING_RATE_DECAY)

        # 基于上方的设定生成最终的学习过程，同时将train和求取获得平均的过程按如下顺序绑定
        train_step = tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9).minimize(loss=loss, global_step=global_step)
        with tf.control_dependencies([train_step, variable_averages_op]):
            train_op = tf.no_op(name="train")

        # 设置用于保存模型的saver
        saver = tf.train.Saver()

        # 开启训练过程
        with tf.Session() as sess:
            # 初始化所有变量，local_variables_initializer的使用因为获取训练和测试数据时候需要用到该方法
            tf.global_variables_initializer().run()
            tf.local_variables_initializer().run()
            sess.run(train_set.initializer)     # , test_set.initializer, valid_set.initializer
            # auc值为0.5等同于猜
            max_auc = 0.5
            i = 1
            while True or i < 25000:
                try:
                    # 推荐这样写，防止报错，下面这两行是tensorflow的一个需要注意的坑，稍有不慎会报出Error
                    feats, labs = sess.run([train_features, train_labels])
                    train_feed = {x: feats, y: labs}
                    _, loss_val, step = sess.run([train_op, loss, global_step], feed_dict=train_feed)
                    if i % 100 == 0:
                        validset_labels = []
                        validset_results = []
                        sess.run(valid_set.initializer)
                        while True:
                            try:
                                features, labels = sess.run([valid_features, valid_labels])
                                valid_feed = {x: features, y: labels}

                                valid_prediction = tf.nn.softmax(sess.run(y_pred, feed_dict=valid_feed))
                                validset_labels.extend([i[0] for i in labels])
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

                # 每100轮训练保存一次模型
                # if (i+1) % 10 == 0:
                #     rocs = []
                #     test_set = DnnNanoporeModel.get_test_data(BATCH_SIZE)
                #
                #     while True:
                #         try:
                #             test_samples = test_set.get_next()
                #             test_tensor = test_samples[:, :-2]
                #             test_label = test_samples[:, -2:]
                #             test_y_pred = tf.nn.softmax(dnn_inferenceV2.inference(test_tensor, l1_regularizer=None,
                #                                                                   l2_regularizer=None, reuse=True))
                #             print(sess.run(test_y_pred))
                #             # 计算正确率
                #             # correct_prediction = tf.equal(tf.argmax(test_y_pred, 1), tf.argmax(test_label, 1))
                #             # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                #             # score = roc_auc_score(sess.run(test_label), sess.run(test_y_pred))
                #             # rocs.append(score)
                #         except tf.errors.OutOfRangeError:
                #             break
                #         except ValueError:
                #             pass
                    # auc_mean = np.mean(rocs)
                    # print("step: ", i, " auc: ", auc_mean)
                    #     training_steps = i
                    #     # 保存当前的训练模型，这里设置了global_step参数，可以让每个被保存的模型文件末尾加上训练轮数，
                    #     # 如 model.ckpt-1000 表示训练1000轮得到的模型
                    #     saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)
                    # print("the best losses=%.4f, accuracy=%.4f" % (min_loss, accuracy))


def main(argv=None):
    DnnNanoporeModel.train()


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    tf.app.run()
