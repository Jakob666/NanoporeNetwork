# -*- coding:utf-8 -*-
"""
@author: hbs
@date: 2018-9-27
description:
    依据peak calling的结果，查看cellflow数据的position是否在merip-seq峰的范围内。如果在范围内，则视为甲基化的A，如果不在
    则视为非甲基化的A。再根据之前使用CNN模型得到的logit的结果，判断预测的准确率
"""
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from interval_tree import _Node, IntervalTree
from scipy.stats import fisher_exact


class Eval:
    def __init__(self, peak_calling_file, inference_file, output):
        """
        :param peak_calling_file: peak_calling的文件
        :param inference_file: CNN预测输出结果
        """
        self.peak_file = peak_calling_file
        self.infer_file = inference_file
        self.peak = None
        self.infer = None
        self.interval_tree = IntervalTree()
        self.output = output

    def load_inference_file(self):
        self.infer = pd.read_csv(self.infer_file, sep="\t", encoding="utf-8")
        # self.infer = self.infer.head()
        return None

    def load_peak_file(self):
        self.peak = pd.read_csv(self.peak_file, sep="\t", encoding="utf-8")
        self.peak = self.peak.set_index("chr")
        return None

    def build_interval_tree(self, chr_num):
        """
        :param chr_num: 染色体号
        :return: 返回区间树的根节点
        """
        peak_starts = list(self.peak.ix[chr_num]["start"])
        peak_ends = list(self.peak.ix[chr_num]["end"])
        root = self.interval_tree.interval_tree(list(zip(peak_starts, peak_ends)))
        return root

    def cover_by_peak(self, record, root):
        """
        检验inference的序列的A位点是否是在macs2 peak的范围内
        :param record: inference data中每一条记录，通过apply方法传入
        :param root: 区间树的根节点
        :return:
        """
        position = record["position"]
        res = self.interval_tree.intervals_containing(root, position)
        if res != 0:
            return 1
        return 0

    def in_peak(self):
        """
        是否在peak覆盖范围内
        :return:
        """
        for chr_num, group in self.infer.groupby("chr"):
            try:
                root = self.build_interval_tree(chr_num)
                group["meth"] = group.apply(self.cover_by_peak, axis=1, args=(root, ))
                group.to_csv(self.output, sep="\t", index=False, header=None, mode="a")
            except KeyError:
                continue

    def crosstab(self, cover, y_pred):
        """
        生成2×2列联表的四个值。两个变量分别是 1.预测结果（分为阴性和阳性）  2.是否被macs2的peak覆盖（分为覆盖和不覆盖）
        :param cover: 样本的实际是否被peak覆盖
        :param y_pred: 样本的预测标签值
        :return:
        """
        pos_cover = 0
        pos_uncover = 0
        neg_cover = 0
        neg_uncover = 0
        for i in range(len(cover)):
            if cover[i] == 1 and y_pred[i] == 1:
                pos_cover += 1
            elif cover[i] == 0 and y_pred[i] == 1:
                pos_uncover += 1
            elif cover[i] == 1 and y_pred[i] == 0:
                neg_cover += 1
            else:
                neg_uncover += 1
        return pos_cover, pos_uncover, neg_cover, neg_uncover

    def fisher_test(self, pos_cover, pos_uncover, neg_cover, neg_uncover):
        """
        :param pos_cover: 分类为阳性且被peak覆盖。
        :param pos_uncover: 分类为阳性但未被peak覆盖。
        :param neg_cover: 分类为阴性且被peak覆盖。
        :param neg_uncover: 分类为阴性但未被peak覆盖。
        :return:
        """
        oddsratio, pvalue = fisher_exact([[pos_cover, pos_uncover], [neg_cover, neg_uncover]], alternative="greater")
        return pvalue

    def eval(self):
        self.load_inference_file()
        self.load_peak_file()
        self.in_peak()
        threshold = 0.62
        data = pd.read_csv(self.output, sep="\t", usecols=[0, 1, 3], header=None, encoding="utf-8")
        data.columns = ["chr", "logits", "label"]
        # print(data[data["label"] == 0].describe())
        # print(data[data["label"] == 1].describe())
        y_score = list(data["logits"])
        judge = lambda x, t: 1 if x > t else 0
        # arr = np.ones(shape=data.shape[0])
        y_score = np.array([judge(i, threshold) for i in y_score])
        # 求取预测为阴性阳性、peak是否覆盖的列联表的四个值
        pos_cover, pos_uncover, neg_cover, neg_uncover = self.crosstab(data["label"].values, y_score)
        print(pos_cover, pos_uncover, neg_cover, neg_uncover)
        p_val = self.fisher_test(pos_cover, pos_uncover, neg_cover, neg_uncover)
        print(p_val)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    e = Eval("/data/nanopore/dL_network/network/cnn_cellflow_res/peaks2.tsv",
             "/data/nanopore/dL_network/network/cnn_cellflow_res/cellflow_pred.tsv",
             "/data/nanopore/dL_network/network/cnn_cellflow_res/output.tsv")
    e.eval()

