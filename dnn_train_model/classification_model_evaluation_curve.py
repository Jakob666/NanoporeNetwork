# -*-coding:utf-8-*-
"""
@author: hbs
@date: 2018-9-23
description:
    用于绘制模型对甲基化和非甲基化预测的准确率
"""
import os
import sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
sys.path.append("/data/nanopore/dL_network/network/dnn_train_model")
sys.path.append("/data/nanopore/dL_network/network")
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import graph_util
from dnn_inference import inference
from load_dataset import NanoporeDataset
from config import *
from restore_pb_model import restore_model_pb


def ROC_and_AUC():
    nd = NanoporeDataset(DATASET, batch_size=BATCH_SIZE)
    test_set = nd.get_test_samples
    test_features, test_labels = test_set.get_next()
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()
        sess.run(test_set.initializer)
        # 加载pb文件中的图
        restore_model_pb(pb_file_path="/data/nanopore/dL_network/network/dnn_train_model/dnn_tf_model_pb/ckpt-21100.pb")
        x = sess.graph.get_tensor_by_name("x:0")
        # y_logit的形式为 tf.nn.softmax(y_pred, name="prediction")
        y_logit = sess.graph.get_tensor_by_name("prediction:0")
        test_labs = []
        test_results = []
        while True:
            try:
                feats, labs = sess.run([test_features, test_labels])
                test_feed = {x: feats}
                test_prediction = sess.run(y_logit, feed_dict=test_feed)

                test_labs.extend([i for i in labs])
                test_results.extend([i for i in test_prediction])
            except tf.errors.OutOfRangeError:
                test_labs = np.array(test_labs)
                test_results = np.array(test_results)

                # 获取ROC
                fpr, tpr, _ = roc_curve(test_labs[:, 0], test_results[:, 0])
                roc_auc = roc_auc_score(test_labs[:, 0], test_results[:, 0])
                ax.plot(fpr, tpr, label="class: %s, auc=%.4f" % ("dnn network", roc_auc))
                del test_labs, test_results
                break
        else:
            print("No checkpoint file found")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title("ROC")
    ax.legend(loc="best")
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.05)
    ax.grid()
    plt.savefig("/data/nanopore/dL_network/network/dnn_train_model/roc.png")


if __name__ == "__main__":
    ROC_and_AUC()
