# -*- coding:utf-8 -*-
"""
@author: hbs
@date: 2018-9-26
description:
    加载cellflow_dataset目录下存储在TFRecord格式文件中的数据。
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import sys
import tensorflow as tf
import numpy as np
from load_data_cnn import LoadCnnData, CnnNanoporeDataset


class LoadCellflowData(LoadCnnData):
    def __init__(self, tfrecord_file, batch_size, repeat_times=1):
        super(LoadCellflowData, self).__init__(tfrecord_file, batch_size, repeat_times)

    # 对父类方法进行复写
    @staticmethod
    def parse_example(serialized_example):
        """
        对之前的TFRecord文件进行解析。
        :param serialized_example: 从TFRecord文件中读取出的二进制化的example对象
        :return:
        """
        example = tf.parse_single_example(serialized_example, features={
            "chrom": tf.FixedLenFeature(shape=[], dtype=tf.string),
            "position": tf.FixedLenFeature(shape=[], dtype=tf.int64),
            "delta_mean": tf.FixedLenFeature([], tf.string),
            "delta_std": tf.FixedLenFeature([], tf.string)
        })
        example["delta_mean"] = tf.reshape(tf.decode_raw(example["delta_mean"], out_type=tf.float64), [4, 17])
        example["delta_std"] = tf.reshape(tf.decode_raw(example["delta_std"], out_type=tf.float64), [4, 17])
        samples = tf.transpose(tf.convert_to_tensor([example["delta_mean"], example["delta_std"]]))
        # 返回输入到网络中的特征值、染色体和位置3个信息
        return samples, example["chrom"], example["position"]


class CellflowNanoporeDataset:
    def __init__(self, data_dir, batch_size=None, num_epoch=1):
        """
        :param data_dir: 存放TFRecord数据文件的目录
        :param batch_size: 一个batch的大小
        :param num_epoch: 重复多少次，也对应训练的轮数
        """
        self.__data_dir = data_dir
        self.__batch_size = batch_size
        self.__cellflow_file = self.__file()
        self.__epoch = num_epoch

    def __file(self):
        """
        匹配数据目录中的文件，并对其进行分类
        :return:
        """
        cellflow_file = tf.train.match_filenames_once(os.path.join(self.__data_dir, "cellflow*"))
        return cellflow_file

    def __get_examples(self, tfrecord_file, repeat=False):
        if repeat:
            d = LoadCellflowData(tfrecord_file, self.__batch_size, self.__epoch)
        else:
            d = LoadCellflowData(tfrecord_file, self.__batch_size)
        iterator = d.load_tfrecord()
        return iterator

    @property
    def get_data(self):
        data = self.__get_examples(self.__cellflow_file)
        return data


if __name__ == "__main__":
    # 以下代码仅用于测试
    nd = CellflowNanoporeDataset("/data/nanopore/dL_network/network/cellflow_dataset/", batch_size=5)
    data = nd.get_data
    features, chrom, position = data.get_next()

    with tf.Session() as sess:
        sess.run((tf.global_variables_initializer(), tf.local_variables_initializer()))
        sess.run(data.initializer)  # 一定不要忘记这句
        # 遍历所有的数据，遍历结束会跑出OutOfRangeError。使用while是因为不确定有多少样例
        while True:
            try:
                print(sess.run(chrom))
                print("-----------------")
                exit()
            except tf.errors.OutOfRangeError:
                break


