# -*- coding:utf-8 -*-
import tensorflow as tf
import os
import pandas as pd
import numpy as np
import argparse
from form_dataset import TFrecordFormer, SequenceExample


class CnnExamples(SequenceExample):
    def __init__(self, data_dir, meth_tfrecord, unmeth_tfrecord, train=0.7, test=0.2, validation=0.1):
        super(CnnExamples, self).__init__(data_dir, meth_tfrecord, unmeth_tfrecord, train, test, validation)

    # 对超类方法进行复写
    @staticmethod
    def load_data(file_path):
        data = pd.read_csv(file_path, sep="\t", encoding="utf-8")
        func = lambda d: np.array([np.array(list(map(np.float, i.split(",")))).tostring() for i in d.split(":")])
        data[["delta_mean", "delta_std"]] = data[["delta_mean", "delta_std"]].applymap(func)
        return data

    # 对超类方法进行复写
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
        example_features["sequence"] = TFrecordFormer.bytes_feature([bytes(example[0], encoding="utf-8")])
        example_features["delta_mean"] = TFrecordFormer.bytes_feature([np.array(example[1]).tostring()])
        example_features["delta_std"] = TFrecordFormer.bytes_feature([np.array(example[2]).tostring()])
        example_features["label"] = TFrecordFormer.float_feature(label)
        return example_features

    def form_example_list(self):
        files = SequenceExample.get_files(self.data_dir)
        for f in files:
            if os.path.split(f)[-1].startswith("meth"):
                data = CnnExamples.load_data(f)
                self.meth_examples.extend(data.values)
            elif os.path.split(f)[-1].startswith("unmeth"):
                data = CnnExamples.load_data(f)
                self.unmeth_examples.extend(data.values)

        record_number = len(self.unmeth_examples)
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

        meth = lambda examples,  sample_idx: (CnnExamples.create_example(examples[idx], True) for idx in sample_idx)
        unmeth = lambda examples, sample_idx: (CnnExamples.create_example(examples[idx], False) for idx in sample_idx)

        meth_train_examples = meth(self.meth_examples, train_samples_idx)
        meth_test_examples = meth(self.meth_examples, test_samples_idx)
        meth_valid_examples = meth(self.meth_examples, valid_samples_idx)

        unmeth_train_examples = unmeth(self.unmeth_examples, train_samples_idx)
        unmeth_test_examples = unmeth(self.unmeth_examples, test_samples_idx)
        unmeth_valid_examples = unmeth(self.unmeth_examples, valid_samples_idx)
        return meth_train_examples, meth_test_examples, meth_valid_examples, unmeth_train_examples, unmeth_test_examples, unmeth_valid_examples


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="form_dataset", description="生成甲基化和非甲基化数据的TFRecord文件")
    parser.add_argument("-s", "--source", action="store", required=True, type=str, help="存放甲基化和非甲基化数据的sequence文件")
    parser.add_argument("-m", "--meth", action="store", required=True, type=str, help="甲基化TFRecord数据集的文件名前缀")
    parser.add_argument("-u", "--unmeth", action="store", required=True, type=str, help="非甲基化TFRecord数据集的文件名前缀")
    args = parser.parse_args()
    source_dir = args.source
    meth_tfr = args.meth
    unmeth_tfr = args.unmeth
    s = CnnExamples(source_dir, meth_tfr, unmeth_tfr)
    m_train, m_test, m_valid, u_train, u_test, u_valid = s.form_example_list()
    s.write_into_file(m_train, m_test, m_valid, u_train, u_test, u_valid)
