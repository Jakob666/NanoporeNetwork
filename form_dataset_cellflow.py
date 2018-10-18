# -*- coding:utf-8 -*-
"""
@author: hbs
@date: 2018-9-26
description:
    生成验证所需要的cellflow细胞系的数据集。用于测试之前构建的CNN模型的精准度。
"""
import tensorflow as tf
import os
import pandas as pd
import numpy as np
import argparse
from form_dataset_cnn import CnnExamples
from form_dataset import TFrecordFormer, SequenceExample
import re


class CellFlowExample(CnnExamples):
    # 对超类的初始化方法进行复写
    def __init__(self, data_dir, output_dir):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.examples = []

    # 对超类方法进行复写
    @staticmethod
    def load_data(file_path, chr_num):
        data = pd.read_csv(file_path, sep="\t", encoding="utf-8")
        func = lambda d: np.array([np.array(list(map(np.float, i.split(",")))).tostring() for i in d.split(":")])
        data[["delta_mean", "delta_std"]] = data[["delta_mean", "delta_std"]].applymap(func)
        data["chr"] = chr_num
        return data

    # 对超类方法进行复写
    @staticmethod
    def create_example(example):
        """
        example是一个列表，其中的数据依次是 start、end、sequence、δμ、δσ 和 length和 label
        :param example:
        :return:
        """
        example_features = dict()
        # 生成相应类型的特征，注意需要以list的形式传入
        example_features["chrom"] = TFrecordFormer.bytes_feature([bytes(example[4], encoding="utf-8")])
        example_features["position"] = TFrecordFormer.int64_feature([int(example[0])])
        example_features["sequence"] = TFrecordFormer.bytes_feature([bytes(example[1], encoding="utf-8")])
        example_features["delta_mean"] = TFrecordFormer.bytes_feature([np.array(example[2]).tostring()])
        example_features["delta_std"] = TFrecordFormer.bytes_feature([np.array(example[3]).tostring()])
        return example_features

    def form_example_list(self):
        files = SequenceExample.get_files(self.data_dir)
        pattern = re.compile("NC_([0-9]+)\..*")
        for f in files:
            if os.path.split(f)[-1].startswith("NC"):
                chr_num = int(re.findall(pattern, f)[0])
                if chr_num >= 23:
                    if chr_num == 23:
                        chr_num = "X"
                    elif chr_num == 24:
                        chr_num = "Y"
                    # 线粒体
                    else:
                        chr_num = "mito"
                chr_num = str(chr_num)
                data = CellFlowExample.load_data(f, chr_num)
                self.examples.extend(data.values)
            else:
                continue

        create = lambda examples_list: (CellFlowExample.create_example(example) for example in examples_list)
        examples = create(self.examples)

        return examples

    def write_into_file(self, data):
        TFrecordFormer.write_to_tfrecord(os.path.join(self.output_dir, "cellflow.tfrecords"), data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="form_dataset", description="生成cellflow数据的TFRecord文件")
    parser.add_argument("-s", "--source", action="store", required=True, type=str, help="存放cellflow的sequence文件的目录")
    parser.add_argument("-o", "--output", action="store", required=True, type=str, help="数据结果存放的目录")
    args = parser.parse_args()
    source_dir = args.source
    out_dir = args.output
    s = CellFlowExample(source_dir, out_dir)
    cellflow_data = s.form_example_list()
    s.write_into_file(cellflow_data)
