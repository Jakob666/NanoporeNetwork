# -*- coding:utf-8 -*-
"""
@author: hbs
@date: 2018-9-18
description:
    加载保存的pb模型文件
"""
import os
import sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
sys.path.append("/data/nanopore/dL_network/network")
sys.path.append("/data/nanopore/dL_network/network/cnn_train_model")
import tensorflow as tf
from tensorflow.python.platform import gfile
from config import *


def restore_model_pb(pb_file_path):
    with gfile.FastGFile(pb_file_path, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

        # 如果不进行下面的处理会报出 ValueError
        for node in graph_def.node:
            if node.op == "RefSwitch":
                node.op = "Switch"
                for index in range(len(node.input)):
                    if "moving_" in node.input[index]:
                        node.input[index] = node.input[index] + "/read"
            elif node.op == "AssignSub":
                node.op = "Sub"
                if "use_locking" in node.attr:
                    del node.attr["use_locking"]

        tf.import_graph_def(graph_def, name="")

    return None



