# -*- coding:utf-8 -*-
"""
@author: hbs
@date: 2018-9-20
@version: 2.0
description:
    加载训练时保存的auc值较高的模型，对测试数据集进行测试，找到最优的模型.
    修正了上一版本测试程序中的一些错误

@update: 2018-9-29
@version: 2.1
description:
    添加了一个获取分类阈值threshold的步骤，使得 logit > threshold的样本被预测为阳性；反之，预测为阴性。与已有标签进行比较，
    使得假阳性率在 10% 以下。同时，使得预测的accuracy最大。
"""
import os
import sys
from sklearn.metrics import roc_auc_score, accuracy_score
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
sys.path.append("/data/nanopore/dL_network/network/cnn_train_model")
sys.path.append("/data/nanopore/dL_network/network")
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import graph_util
from cnn_inference import inference
from load_data_cnn import CnnNanoporeDataset
from config import *


def evaluate(threshold, output_file):
    """
    :param threshold: 分类阈值。
    :param output_file: 已开启的输出文件对象。
    :return:
    """
    x = tf.placeholder(dtype=tf.float32, shape=[None, NANOPORE_HEIGHT, NANOPORE_WIDTH, NANOPORE_CHANNEL], name="x")

    nd = CnnNanoporeDataset(DATASET, batch_size=BATCH_SIZE)
    test_set = nd.get_test_samples
    test_features, test_labels = test_set.get_next()

    # 测试的前向运算过程
    test_pred = inference(input_tensor=x, train=False, regularizer=None)
    y_logit = tf.nn.softmax(test_pred, name="prediction")

    # 通过变量重命名加载模型，向前传播过程无需求取滑动平均值
    variable_average = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_average.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()
        sess.run(test_set.initializer)
        # get_checkpoint_state方法通过point文件找到目录中最新的模型的文件名
        ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            # global_steps = ckpt.model_checkpoint_path.split("/")[-1].split("-")[-1]
            test_labs = []
            test_results = []
            while True:
                try:
                    feats, labs = sess.run([test_features, test_labels])
                    test_feed = {x: feats}
                    test_prediction = sess.run(y_logit, feed_dict=test_feed)

                    test_labs.extend([i[0] for i in labs])
                    test_results.extend([i[0] for i in test_prediction])
                except tf.errors.OutOfRangeError:
                    for i in range(len(test_results)):
                        test_results[i] = 1.0 if test_results[i] > threshold else 0.0
                    spec = calc_specificity(test_labs, test_results)
                    # if spec < 0.9:
                    #     break

                    # acc = accuracy_score(test_labs, test_results)
                    score = roc_auc_score(test_labs, test_results)
                    del test_labs, test_results
                    result = "\t".join([format(threshold, ".2f"), format(spec, ".4f"), format(score, ".4f")])
                    with open(output_file, "a") as f:
                        f.write(result + "\n")
                    # print(result)
                    break
        else:
            print("No checkpoint file found")
    # 释放计算图与变量，一定写在with tf.Session外，否则会报错
    tf.reset_default_graph()


def calc_specificity(y_true, y_pred):
    """
    计算特异性，即 真阴 / (真阴 + 假阳)
    :param y_true: 真实的标签，其中1是阳性，0是阴性。
    :param y_pred: 通过阈值分类后的标签，其中1是阳性，0是阴性。
    :return:
    """
    total_neg = len(y_true) - sum(y_true)
    true_neg = 0
    for i in range(len(y_true)):
        if y_true[i] == 0 and y_pred[i] == 0:
            true_neg += 1
    spec = true_neg / total_neg

    return spec


if __name__ == "__main__":
    thresholds = list(np.linspace(0.2, 0.8, 60))
    output = "eval_res2.tsv"
    for t in thresholds:
        evaluate(t, output)

