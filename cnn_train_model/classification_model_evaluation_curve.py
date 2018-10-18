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
sys.path.append("/data/nanopore/dL_network/network/cnn_train_model")
sys.path.append("/data/nanopore/dL_network/network")
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import graph_util
from cnn_inference import inference
from load_data_cnn import CnnNanoporeDataset
from config import *


def ROC_and_AUC():
    x = tf.placeholder(dtype=tf.float32, shape=[None, NANOPORE_HEIGHT, NANOPORE_WIDTH, NANOPORE_CHANNEL], name="x")
    nd = CnnNanoporeDataset(DATASET, batch_size=BATCH_SIZE)
    test_set = nd.get_test_samples
    test_features, test_labels = test_set.get_next()
    # 测试的前向运算过程
    test_pred = inference(input_tensor=x, train=False, regularizer=None)
    y_logit = tf.nn.softmax(test_pred, name="prediction")

    variable_average = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_average.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()
        sess.run(test_set.initializer)
        # get_checkpoint_state方法通过point文件找到目录中最新的模型的文件名
        ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
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
                    ax.plot(fpr, tpr, label="class: %s, auc=%.4f" % ("cnn network", roc_auc))
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
    plt.savefig("/data/nanopore/dL_network/network/cnn_train_model/roc.png")


if __name__ == "__main__":
    ROC_and_AUC()
