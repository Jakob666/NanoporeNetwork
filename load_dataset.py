# -*- coding: utf-8 -*-
"""
@author: hbs
@date: 2018-9-6
description:
    读取甲基化和非甲基化的TFRecord文件，依据不同的文件的数据给相应的记录注明label，如 甲基化的label为1，非甲基化的label为0.
    同时生成训练、测试和验证数据集，比例为60%、20%和20%
"""
import os
import sys
sys.path.append("/data/nanopore/dL_network/network/dnn_train_model/")
from config import *
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
import numpy as np


class LoadDataset:
    """
    用于从TFRecord文件中读取数据
    """
    def __init__(self, tfrecord_file, batch_size, repeat_times=1):
        """
        :param tfrecord_file: TFRecord文件的路径
        :param batch_size: 一次性产生的样本数目
        :param repeat_times: 对dataset执行多少次repeat操作
        """
        self.__target_file = tfrecord_file
        self.__batch_size = batch_size
        self.__repeat = repeat_times

    @staticmethod
    def parse_example(serialized_example):
        """
        对之前的TFRecord文件进行解析。
        :param serialized_example: 从TFRecord文件中读取出的二进制化的example对象
        :return:
        """
        # 对二进制化的样例进行解析
        example = tf.parse_single_example(serialized_example, features={
            # 这里一定要注明维度，否则会报错
            "delta_mean": tf.FixedLenFeature([1, 7], tf.float32),
            "delta_std": tf.FixedLenFeature([1, 7], tf.float32),
            "label": tf.FixedLenFeature([1, 2], tf.float32)
        })
        samples = tf.concat([example["delta_mean"], example["delta_std"]], 1)
        return samples[0], example["label"][0]

    def load_tfrecord(self):
        input_files = self.__target_file
        dataset = tf.data.TFRecordDataset(input_files)
        dataset = dataset.map(self.parse_example)
        dataset = dataset.shuffle(buffer_size=50000)
        if self.__batch_size is not None:
            dataset = dataset.batch(self.__batch_size)
        if self.__repeat != 1:
            dataset = dataset.repeat(self.__repeat)
        iterator = dataset.make_initializable_iterator()

        return iterator


class NanoporeDataset:
    """
    返回存放在TFRecord文件中的不同数据。
    """
    def __init__(self, data_dir, batch_size=None, num_epoch=1):
        """
        :param data_dir: 存放TFRecord数据文件的目录
        :param batch_size: 一个batch的大小
        :param num_epoch: 重复多少次，也对应训练的轮数
        """
        self.__data_dir = data_dir
        self.__batch_size = batch_size
        self.__meth_files, self.__unmeth_files = self.__file_classification()
        self.__epoch = num_epoch

    def __file_classification(self):
        """
        匹配数据目录中的文件，并对其进行分类
        :return:
        """
        meth_files = dict()
        unmeth_files = dict()
        meth_files["train_set"] = tf.train.match_filenames_once(os.path.join(self.__data_dir, "meth_data_train*"))
        meth_files["test_set"] = tf.train.match_filenames_once(os.path.join(self.__data_dir, "meth_data_test*"))
        meth_files["valid_set"] = tf.train.match_filenames_once(os.path.join(self.__data_dir, "meth_data_valid*"))

        unmeth_files["train_set"] = tf.train.match_filenames_once(os.path.join(self.__data_dir, "unmeth_data_train*"))
        unmeth_files["test_set"] = tf.train.match_filenames_once(os.path.join(self.__data_dir, "unmeth_data_train*"))
        unmeth_files["valid_set"] = tf.train.match_filenames_once(os.path.join(self.__data_dir, "unmeth_data_valid*"))
        return meth_files, unmeth_files

    def __get_examples(self, tfrecord_file, repeat=False):
        if repeat:
            d = LoadDataset(tfrecord_file, self.__batch_size, self.__epoch)
        else:
            d = LoadDataset(tfrecord_file, self.__batch_size)
        iterator = d.load_tfrecord()
        return iterator

    # 在别的模块调用该类时，以属性的形式调用方法，返回训练集、测试集和验证集的生成器
    @property
    def get_train_samples(self):
        """
        获取训练数据集，返回一个生成器，每次产生一个batch的数据
        :return:
        """
        train_files = [self.__meth_files["train_set"][0], self.__unmeth_files["train_set"][0]]
        train = self.__get_examples(train_files, repeat=True)
        return train

    @property
    def get_test_samples(self):
        """
        获取测试数据
        :return:
        """
        test_files = [self.__meth_files["test_set"][0], self.__unmeth_files["test_set"][0]]
        test = self.__get_examples(test_files)
        return test

    @property
    def get_valid_samples(self):
        """
        获取验证数据
        :return:
        """
        valid_files = [self.__meth_files["valid_set"][0], self.__unmeth_files["valid_set"][0]]
        valid = self.__get_examples(valid_files)
        return valid


if __name__ == "__main__":
    # 以下代码仅用于测试
    nd = NanoporeDataset("/data/nanopore/dL_network/network/test_dataset/", batch_size=5)
    train = nd.get_test_samples
    features, labels = train.get_next()

    with tf.Session() as sess:
        sess.run((tf.global_variables_initializer(), tf.local_variables_initializer()))
        sess.run(train.initializer)     # 一定不要忘记这句
        # 遍历所有的数据，遍历结束会跑出OutOfRangeError。使用while是因为不确定有多少样例
        while True:
            try:
                print(sess.run([features, labels]))
                print("-----------------")
            except tf.errors.OutOfRangeError:
                break

