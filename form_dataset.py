# -*- coding:utf-8 -*-
"""
@author: hbs
@date: 2018-9-4
description:
    用于构建训练所需的TFRecord数据集。将之前得到的 甲基化和非甲基化的序列生成相应的数据格式。同时将数据分为训练集、测试集和验证集。
    数据来源：
    甲基化数据和非甲基化数据均存放在端口号28的服务器 /data/nanopore/dL_network/common_seq_in_groups/common_sequence 目录下
"""
import tensorflow as tf
import pandas as pd
import numpy as np
import argparse
import os


class TFrecordFormer:
    @staticmethod
    def int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    @staticmethod
    def bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

    @staticmethod
    def float_feature(value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))

    @staticmethod
    def write_to_tfrecord(filename, example_list):
        """
        将样本的各个特征写入TFRecord文件。
        :param filename: TFRecord文件路径。
        :param example_list: 样本对象的列表，每个样本对象的特征需要用上面的三个方法进行封装
        :return:
        """
        # 创建一个writer对象
        writer = tf.python_io.TFRecordWriter(filename)
        for example in example_list:
            example = tf.train.Example(features=tf.train.Features(feature=example))
            # 二进制方式写入节省内存
            writer.write(example.SerializeToString())
        writer.close()


class SequenceExample:
    def __init__(self, data_dir, meth_tfrecord, unmeth_tfrecord, train=0.7, test=0.2, validation=0.1):
        self.data_dir = data_dir
        self.meth_examples = []
        self.unmeth_examples = []
        self.meth_tfr = meth_tfrecord
        self.unmeth_tfr = unmeth_tfrecord
        self.train_ratio = train
        self.test_ratio = test
        self.valid_ratio = validation

    @staticmethod
    def get_files(sequence_record_dir):
        return (os.path.join(sequence_record_dir, f) for f in os.listdir(sequence_record_dir))

    @staticmethod
    def load_data(file_path):
        data = pd.read_csv(file_path, sep="\t", encoding="utf-8")
        func = lambda s: np.array(list(map(np.float, s.split(","))))
        data[["delta_mean", "delta_std", "length"]] = data[["delta_mean", "delta_std", "length"]].applymap(func)
        return data

    def form_example_list(self):
        files = SequenceExample.get_files(self.data_dir)
        for f in files:
            if os.path.split(f)[-1].startswith("meth"):
                data = SequenceExample.load_data(f)
                self.meth_examples.extend(data.values)
            elif os.path.split(f)[-1].startswith("unmeth"):
                data = SequenceExample.load_data(f)
                self.unmeth_examples.extend(data.values)
        record_number = len(self.meth_examples)
        # 得到训练集、测试集和验证集的样本数目
        train_number = int(np.floor(record_number * self.train_ratio))
        test_number = int(np.floor(record_number * self.test_ratio))
        # valid_number = record_number - train_number - test_number

        # 从数据中获取样本
        shuffled_indx = np.arange(record_number)
        np.random.shuffle(shuffled_indx)
        train_samples_idx = shuffled_indx[: train_number]
        test_samples_idx = shuffled_indx[train_number: train_number + test_number]
        valid_samples_idx = shuffled_indx[train_number + test_number:]

        meth = lambda examples,  sample_idx: (SequenceExample.create_example(examples[idx], True) for idx in sample_idx)
        unmeth = lambda examples, sample_idx: (SequenceExample.create_example(examples[idx], False) for idx in sample_idx)

        meth_train_examples = meth(self.meth_examples, train_samples_idx)
        meth_test_examples = meth(self.meth_examples, test_samples_idx)
        meth_valid_examples = meth(self.meth_examples, valid_samples_idx)

        unmeth_train_examples = unmeth(self.unmeth_examples, train_samples_idx)
        unmeth_test_examples = unmeth(self.unmeth_examples, test_samples_idx)
        unmeth_valid_examples = unmeth(self.unmeth_examples, valid_samples_idx)
        return meth_train_examples, meth_test_examples, meth_valid_examples, unmeth_train_examples, unmeth_test_examples, unmeth_valid_examples

    @staticmethod
    def create_example(example, if_meth):
        """
        example是一个列表，其中的数据依次是 start、end、sequence、δμ、δσ 和 length和 label
        :param example:
        :param if_meth: 是否是甲基化数据
        :return:
        """
        if if_meth:
            label = [1, 0]
        else:
            label = [0, 1]
        example_features = dict()
        # 生成相应类型的特征，注意需要以list的形式传入
        example_features["start"] = TFrecordFormer.int64_feature([example[0]])
        example_features["end"] = TFrecordFormer.int64_feature([example[1]])
        example_features["sequence"] = TFrecordFormer.bytes_feature([bytes(example[2], encoding="utf-8")])
        example_features["delta_mean"] = TFrecordFormer.float_feature(example[3])
        example_features["delta_std"] = TFrecordFormer.float_feature(example[4])
        example_features["length"] = TFrecordFormer.float_feature(example[5])
        example_features["label"] = TFrecordFormer.float_feature(label)
        return example_features

    def write_into_file(self, meth_train, meth_test, meth_valid, unmeth_train, unmeth_test, unmeth_valid):
        TFrecordFormer.write_to_tfrecord(self.meth_tfr + "_train.tfrecords", meth_train)
        TFrecordFormer.write_to_tfrecord(self.meth_tfr + "_test.tfrecords", meth_test)
        TFrecordFormer.write_to_tfrecord(self.meth_tfr + "_valid.tfrecords", meth_valid)
        TFrecordFormer.write_to_tfrecord(self.unmeth_tfr + "_train.tfrecords", unmeth_train)
        TFrecordFormer.write_to_tfrecord(self.unmeth_tfr + "_test.tfrecords", unmeth_test)
        TFrecordFormer.write_to_tfrecord(self.unmeth_tfr + "_valid.tfrecords", unmeth_valid)


if __name__ == "__main__":
    # s = SequenceExample("/data/nanopore/dL_network/common_seq_in_groups/test/",
    #                     "/data/nanopore/dL_network/network/dataset/meth.tfrecords",
    #                     "/data/nanopore/dL_network/network/dataset/unmeth.tfrecords")
    # me, unme = s.form_example_list()
    # s.write_into_file(me, unme)
    parser = argparse.ArgumentParser(prog="form_dataset", description="生成甲基化和非甲基化数据的TFRecord文件")
    parser.add_argument("-s", "--source", action="store", required=True, type=str, help="存放甲基化和非甲基化数据的sequence文件")
    parser.add_argument("-m", "--meth", action="store", required=True, type=str, help="甲基化TFRecord数据集的文件名前缀")
    parser.add_argument("-u", "--unmeth", action="store", required=True, type=str, help="非甲基化TFRecord数据集的文件名前缀")
    args = parser.parse_args()
    source_dir = args.source
    meth_tfr = args.meth
    unmeth_tfr = args.unmeth
    s = SequenceExample(source_dir, meth_tfr, unmeth_tfr)
    m_train, m_test, m_valid, u_train, u_test, u_valid = s.form_example_list()
    s.write_into_file(m_train, m_test, m_valid, u_train, u_test, u_valid)
