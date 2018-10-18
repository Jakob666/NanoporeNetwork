# -*- coding:utf-8 -*-
"""
@author: hbs
@date: 2018-9-10
description:
    对数据进行标准化
"""
import pandas as pd
import numpy as np
import os


def load_data(file_path):
    data = pd.read_csv(file_path, sep="\t", encoding="utf-8")
    func = lambda s: np.array(list(map(np.float, s.split(","))))
    data[["delta_mean", "delta_std", "length"]] = data[["delta_mean", "delta_std", "length"]].applymap(func)
    return data


def normalize_data(data_dir, meth_output, unmeth_output):
    files = os.listdir(data_dir)
    meth_data = pd.DataFrame(columns=["start", "end", "sequence", "delta_mean", "delta_std", "length"])
    unmeth_data = pd.DataFrame(columns=["start", "end", "sequence", "delta_mean", "delta_std", "length"])
    for f in files:
        filepath = os.path.join(data_dir, f)
        d = load_data(filepath)
        if f.startswith("meth"):
            meth_data = pd.concat([meth_data, d], ignore_index=True)
        else:
            unmeth_data = pd.concat([unmeth_data, d], ignore_index=True)
    meth_data = normalize(meth_data)
    unmeth_data = normalize(unmeth_data)

    func = lambda s: ",".join(list(map(str, s)))
    meth_data[["delta_mean", "delta_std", "length"]] = meth_data[["delta_mean", "delta_std", "length"]].applymap(func)
    unmeth_data[["delta_mean", "delta_std", "length"]] = unmeth_data[["delta_mean", "delta_std", "length"]].applymap(func)
    meth_data.to_csv(meth_output, sep="\t", index=False, encoding="utf-8")
    unmeth_data.to_csv(unmeth_output, sep="\t", index=False, encoding="utf-8")


def normalize(d):
    delta_m = d["delta_mean"].values
    delta_m_mean = np.mean(delta_m)
    delta_m_var = np.var(delta_m)
    for i in range(delta_m.shape[0]):
        delta_m[i] = (delta_m[i] - delta_m_mean) / np.sqrt(delta_m_var + 0.001)

    delta_s = d["delta_std"].values
    delta_s_mean = np.mean(delta_s)
    delta_s_var = np.var(delta_s)
    for i in range(delta_s.shape[0]):
        delta_s[i] = (delta_s[i] - delta_s_mean) / np.sqrt(delta_s_var + 0.001)

    d["delta_mean"] = pd.Series(delta_m)
    d["delta_std"] = pd.Series(delta_s)
    return d


if __name__ == "__main__":
    normalize_data("/data/nanopore/dL_network/common_seq_in_groups/common_sequence/",
                   meth_output="/data/nanopore/dL_network/network/normalized_data/meth_norm.tsv",
                   unmeth_output="/data/nanopore/dL_network/network/normalized_data/unmeth_norm.tsv")

