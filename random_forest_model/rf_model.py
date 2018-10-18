# -*- coding: utf-8 -*-
"""
@author: hbs
@date: 2018-9-10
description:
    使用随机森林算法对甲基化和非甲基化的数据进行分类。
"""
import os
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score


class NanoporeMethylationClassifier:
    def __init__(self, dataset, train_ratio, test_ratio):
        """
        :param dataset: 数据集存放位置
        :param train_ratio: 训练数据集的占比
        :param test_ratio: 测试数据集的占比
        """
        self.__dataset = dataset
        self.__meth_files = []
        self.__unmeth_files = []
        self.__meth_data = []
        self.__unmeth_data = []
        self.train_ratio = train_ratio
        self.test_ratio = test_ratio

    def __file_classifications(self):
        data_files = os.listdir(self.__dataset)
        classifier = lambda f: self.__meth_files.append(f) if f.startswith("meth") else self.__unmeth_files.append(f)
        for f in data_files:
            classifier(f)
        return None

    def __load_data(self, file_path):
        data = pd.read_csv(file_path, sep="\t", usecols=[3, 4], encoding="utf-8")
        func = lambda s: np.array(list(map(np.float, s.split(","))))
        data[["delta_mean", "delta_std"]] = data[["delta_mean", "delta_std"]].applymap(func)
        return data

    def __get_meth_data(self):
        for f in self.__meth_files:
            f = os.path.join(self.__dataset, f)
            self.__meth_data.extend(self.__load_data(f).values)
        meth_label = np.ones(shape=(len(self.__meth_data), 1))
        self.__meth_data = np.array(self.__meth_data)
        del meth_label
        return None

    def __get_unmeth_data(self):
        for f in self.__unmeth_files:
            f = os.path.join(self.__dataset, f)
            self.__unmeth_data.extend(self.__load_data(f).values)
        unmeth_label = np.zeros(shape=(len(self.__unmeth_data), 1))
        self.__unmeth_data = np.array(self.__unmeth_data)
        del unmeth_label
        return None

    def __form_dataset(self):
        dataset = []
        for rec in self.__meth_data:
            rec = np.append(rec[0], rec[1])
            rec = np.append(rec, [1])
            dataset.append(rec)
        for rec in self.__unmeth_data:
            rec = np.append(rec[0], rec[1])
            rec = np.append(rec, [0])
            dataset.append(rec)
        dataset = np.array(dataset)
        np.random.shuffle(dataset)
        return dataset

    def __rf_classify(self, dataset):
        train_number = int(dataset.shape[0] * self.train_ratio)
        train_samples = dataset[: train_number, :]
        test_samples = dataset[train_number:, :]
        clf = RandomForestClassifier(n_estimators=25, max_depth=15, criterion="gini", oob_score=True)
        train_features = train_samples[:, :-1]
        train_labels = train_samples[:, -1]
        clf.fit(train_features, train_labels)

        test_features = test_samples[:, : -1]
        test_labels = test_samples[:, -1]
        test_pred = clf.predict(test_features)
        auc = roc_auc_score(test_labels, test_pred)
        print(auc)

    def __gbdt_classify(self, dataset):
        train_number = int(dataset.shape[0] * self.train_ratio)
        train_samples = dataset[: train_number, :]
        test_samples = dataset[train_number:, :]
        clf = GradientBoostingClassifier(n_estimators=20, max_depth=15)
        train_features = train_samples[:, :-1]
        train_labels = train_samples[:, -1]
        clf.fit(train_features, train_labels)

        test_features = test_samples[:, : -1]
        test_labels = test_samples[:, -1]
        test_pred = clf.predict(test_features)
        auc = roc_auc_score(test_labels, test_pred)
        print(auc)

    def main(self):
        self.__file_classifications()
        self.__get_meth_data()
        self.__get_unmeth_data()
        dataset = self.__form_dataset()
        # self.__rf_classify(dataset)
        self.__gbdt_classify(dataset)

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    nmc = NanoporeMethylationClassifier("/data/nanopore/dL_network/common_seq_in_groups/common_sequence/", 0.7, 0.3)
    nmc.main()

