# -*- coding:utf-8 -*-
"""
@author: hbs
@date: 2018-9-14
description:
    将之前持久化的ckpt模型文件转换为仅存储前向过程的 pb模型文件
"""
import os
import sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
sys.path.append("/data/nanopore/dL_network/network/cnn_train_model")
import tensorflow as tf
from tensorflow.python.framework import graph_util
from config import *


def freeze_graph():
    # 检查目录下ckpt文件是否可用
    checkpoint = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
    input_checkpoint = checkpoint.model_checkpoint_path
    ckpt_files = (model for model in os.listdir(MODEL_SAVE_PATH) if os.path.split(model)[-1].endswith(".meta"))
    # 需要导出的运算节点，目前只导出一个，如果导出多个使用逗号分隔即可
    output_node = "prediction"
    for ck in ckpt_files:
        out_filename = ck.split(".")[-2] + ".pb"
        output_graph = os.path.join(PB_DIR, out_filename)
        # clear_device参数一定要设置
        saver = tf.train.import_meta_graph(os.path.join(MODEL_SAVE_PATH, ck), clear_devices=True)
        graph = tf.get_default_graph()
        input_graph_def = graph.as_graph_def()

        with tf.Session() as sess:
            saver.restore(sess, input_checkpoint)
            output_graph_def = graph_util.convert_variables_to_constants(  # 模型持久化，将变量值固定
                sess,
                input_graph_def,
                output_node.split(",")  # 如果有多个输出节点，以逗号隔开
            )
            with tf.gfile.GFile(output_graph, "wb") as f:  # 保存模型
                f.write(output_graph_def.SerializeToString())  # 序列化输出，以二进制形式存储


if __name__ == "__main__":
    freeze_graph()

