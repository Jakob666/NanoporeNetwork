# -*- coding:utf-8 -*-
"""
@author: hbs
@date: 2018-9-30
description:
    绘制merip-seq得到的被peak覆盖的位点与未被peak覆盖的位点的特征分布。
"""
import os
import sys
from scipy.stats import mode
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
sys.path.append("/data/nanopore/dL_network/network")
sys.path.append("/data/nanopore/dL_network/network/cnn_train_model")
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from load_dataset_cellflow import CellflowNanoporeDataset
import pandas as pd
from config import *
import numpy as np
import warnings


class DrawFig:
    def __init__(self, inference_file, peak_file, threshold):
        """
        :param inference_file: CNN网络预测结果
        :param peak_file: macs2 peak calling得到的merip-seq结果
        :param threshold: 分类阈值
        """
        self.infer = inference_file
        self.peaks = peak_file
        self.threshold = threshold
        self.infer_data = self.load_infer_data()

    def load_infer_data(self):
        infer_data = pd.read_csv(self.infer, sep="\t", header=None, usecols=[0, 2, 3])
        infer_data.columns = ["chr", "position", "covered"]
        infer_data["chr"] = infer_data["chr"].apply(str)
        infer_data.set_index(["chr", "position"], inplace=True)
        return infer_data

    def load_signal_data(self):
        miu_covered = []
        miu_uncovered = []
        sigma_covered = []
        sigma_uncovered = []

        cnd = CellflowNanoporeDataset(CELLFLOW_DATASET, batch_size=BATCH_SIZE)
        data = cnd.get_data
        features, chrom, position = data.get_next()
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            tf.local_variables_initializer().run()
            sess.run(data.initializer)
            while True:
                try:
                    feats, chr_num, pos = sess.run([features, chrom, position])
                    mcp, mup, scp, sup = self.extract_figure_data(feats, chr_num, pos)
                    miu_covered.extend(mcp)
                    miu_uncovered.extend(mup)
                    sigma_covered.extend(scp)
                    sigma_uncovered.extend(sup)

                except tf.errors.OutOfRangeError:
                    break
        return miu_covered, miu_uncovered, sigma_covered, sigma_uncovered

    def extract_figure_data(self, features, chrs, pos):
        """
        提取用于绘图的数据。
        :param features: 一个batch的样本的特征值集合
        :param chrs: 一个batch的样本所在染色体的集合
        :param pos: 一个batch的样本所在染色体位置的集合
        :return:
        """
        miu_covered_by_peak = []
        miu_uncovered_by_peak = []
        sigma_covered_by_peak = []
        sigma_uncovered_by_peak = []

        for i in range(len(chrs)):
            feat, chr_num, position = features[i], chrs[i], pos[i]
            chr_num = str(chr_num, encoding="utf-8")
            feat = feat.T
            # 取出第7到12个k-mer，因为包含最中心的A
            miu, sigma = [], []
            for j in range(7, 12):
                mius = feat[:, :, j][0].flatten()
                sigmas = feat[:, :, j][1].flatten()
                for k in range(4):
                    if mius[k] != 0:
                        miu.append(mius[k])
                        break
                    if sigmas[k] != 0:
                        sigma.append(sigmas[k])
                        break

            try:
                if self.infer_data.ix[chr_num].ix[position]["covered"] == 0:
                    miu_uncovered_by_peak.append(miu)
                    sigma_uncovered_by_peak.append(sigma)
                elif self.infer_data.ix[chr_num].ix[position]["covered"] == 1:
                    miu_covered_by_peak.append(miu)
                    sigma_covered_by_peak.append(sigma)
            except:
                continue
        return miu_covered_by_peak, miu_uncovered_by_peak, sigma_covered_by_peak, sigma_uncovered_by_peak

    def draw_fig(self, cover_data, uncover_data):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        cover_data = pd.DataFrame(cover_data)

        cover_data = self.calc_probability(cover_data)
        uncover_data = pd.DataFrame(uncover_data)

        uncover_data = self.calc_probability(uncover_data)
        cover_data = pd.DataFrame(cover_data)
        for k, v in cover_data.groupby(by=2, as_index=True):
            v.sort_values(by=0, inplace=True)
            ax.bar(v[0], v[1], v[2], zdir='y', color="navy", alpha=0.75)

        uncover_data = pd.DataFrame(uncover_data)
        for k, v in uncover_data.groupby(by=2, as_index=True):
            v.sort_values(by=0, inplace=True)
            ax.bar(v[0], v[1], v[2], zdir='y', color="orange", alpha=0.7)

        ax.set_xlabel("delta", color="black")
        ax.set_xlim(-15, 15)
        ax.set_ylabel("event number", color="black")
        ax.set_ylim(0, 6)
        ax.set_zlabel("p", color="black")
        ax.legend(loc="best")
        plt.savefig("cnm3.png")

    def calc_probability(self, dataframe):
        """
        计算 delta_mean 或 delta_std 的概率分布
        :param dataframe:
        :return:
        """
        # 序列的总数
        total_record = dataframe.shape[0]
        # func = lambda x: round(x, 1)
        # dataframe = dataframe.applymap(func)
        columns = dataframe.columns
        points = []
        for i in range(len(columns)):
            counter = dataframe[columns[i]].value_counts()
            counter /= total_record
            points += list(zip(list(counter.index), list(counter.values), [i for _ in range(total_record)]))
        return points


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    df = DrawFig("output2.tsv", "cellflow_pred.tsv", 0.62)
    miu_c, miu_u, sig_c, sig_u = df.load_signal_data()
    miu_u = np.random.choice(miu_u, size=3*len(miu_c)).tolist()
    df.draw_fig(miu_c, miu_u)
    # print(set(df.infer_data.index))
    # print(df.infer_data[df.infer_data["chr"]==9 and df.infer_data["position"]==27024138])

