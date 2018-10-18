# -*- coding:utf-8 -*-
"""
@author: hbs
@date: 2018-9-27
description:
    依据peak calling的结果，查看cellflow数据的position是否在merip-seq峰的范围内。如果在范围内，则视为甲基化的A，如果不在
    则视为非甲基化的A。再根据之前使用CNN模型得到的logit的结果，判断预测的准确率
"""
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score


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
        self.output = output

    def load_inference_file(self):
        self.infer = pd.read_csv(self.infer_file, sep="\t", encoding="utf-8")
        # self.infer = self.infer.head()
        return None

    def load_peak_file(self):
        self.peak = pd.read_csv(self.peak_file, sep="\t", encoding="utf-8")
        self.peak = self.peak.set_index("chr")
        return None

    def cover_by_peak(self, record):
        """
        检验inference的序列的A位点是否是在macs2 peak的范围内
        :param record: inference data中每一条记录，通过apply方法传入
        :return:
        """
        chr_num = record["chr"]
        position = record["position"]
        peak_starts = list(self.peak.ix[chr_num]["start"])
        peak_ends = list(self.peak.ix[chr_num]["end"])
        # 使用一个变种的二分法查找所在区间
        head = 0
        tail = len(peak_starts) - 1
        mid = int((head + tail) / 2)
        while head <= tail:
            try:
                if peak_starts[mid] <= position <= peak_starts[mid+1]:
                    if peak_starts[mid] <= position <= peak_ends[mid]:
                        del peak_starts, peak_ends
                    # return [peak_starts[mid], peak_ends[mid]]
                        return 1
                    else:
                        return 0
                elif position > peak_starts[mid]:
                    head = mid + 1
                    mid = int((head + tail) / 2)
                else:
                    tail = mid
                    mid = int((head + tail) / 2)
            except IndexError:
                return 0
        del peak_starts, peak_ends

        return 0

    def in_peak(self):
        """
        是否在peak覆盖范围内
        :return:
        """
        self.infer["meth"] = self.infer.apply(self.cover_by_peak, axis=1)
        self.infer.to_csv(self.output, sep="\t", index=False)

    def eval(self):
        self.load_inference_file()
        self.load_peak_file()
        self.in_peak()


if __name__ == "__main__":
    e = Eval("/data/nanopore/dL_network/network/cnn_cellflow_res/peaks.tsv",
             "/data/nanopore/dL_network/network/cnn_cellflow_res/cellflow_pred.tsv",
             "/data/nanopore/dL_network/network/cnn_cellflow_res/output.tsv")
    e.eval()


