# -*- coding:utf-8 -*-
"""
@author: hbs
@date: 2018-9-14
description:
    加载cnn_dataset目录下存储在TFRecord格式文件中的数据。
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import sys
sys.path.append("/data/nanopore/dL_network/network/cnn_train_model/")
import tensorflow as tf
import numpy as np
from load_dataset import LoadDataset, NanoporeDataset


class LoadCnnData:
    def __init__(self, tfrecord_file, batch_size, repeat_times=1):
        """
        :param tfrecord_file: TFRecord文件的路径
        :param batch_size: 一次性产生的样本数目
        :param repeat_times: 对dataset执行多少次repeat操作
        """
        self.__target_file = tfrecord_file
        self.__batch_size = batch_size
        self.__repeat = repeat_times

    # 对父类方法进行复写
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
            "delta_mean": tf.FixedLenFeature([], tf.string),
            "delta_std": tf.FixedLenFeature([], tf.string),
            "label": tf.FixedLenFeature([1, 2], tf.float32)
        })
        # 一定要reshape否则是一维数组
        example["delta_mean"] = tf.reshape(tf.decode_raw(example["delta_mean"], out_type=tf.float64), [4, 17])
        example["delta_std"] = tf.reshape(tf.decode_raw(example["delta_std"], out_type=tf.float64), [4, 17])
        samples = tf.transpose(tf.convert_to_tensor([example["delta_mean"], example["delta_std"]]))
        return samples, example["label"][0]

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


class CnnNanoporeDataset(NanoporeDataset):
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

    # 对父类方法进行复写
    def __get_examples(self, tfrecord_file, repeat=False):
        if repeat:
            d = LoadCnnData(tfrecord_file, self.__batch_size, self.__epoch)
        else:
            d = LoadCnnData(tfrecord_file, self.__batch_size)
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
    nd = CnnNanoporeDataset("/data/nanopore/dL_network/network/test_dataset/", batch_size=5)
    train = nd.get_test_samples
    features, labels = train.get_next()

    with tf.Session() as sess:
        sess.run((tf.global_variables_initializer(), tf.local_variables_initializer()))
        sess.run(train.initializer)  # 一定不要忘记这句
        # 遍历所有的数据，遍历结束会跑出OutOfRangeError。使用while是因为不确定有多少样例
        while True:
            try:
                print(sess.run(features))
                print("-----------------")
                exit()
            except tf.errors.OutOfRangeError:
                break

