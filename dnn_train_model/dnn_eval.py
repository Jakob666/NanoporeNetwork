# -*- coding: utf-8 -*-
"""
@author: hbs
@date: 2018-9-12
description:
    加载训练时保存的auc值较高的模型，对测试数据集进行测试，找到最优的模型
"""
import os
import sys
from sklearn.metrics import roc_auc_score
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
sys.path.append("/data/nanopore/dL_network/network")
sys.path.append("/data/nanopore/dL_network/network/dnn_train_model")
import tensorflow as tf
from tensorflow.python.platform import gfile
from load_dataset import NanoporeDataset
from config import *
from restore_pb_model import restore_model_pb


class Evaluator:
    @staticmethod
    def testset_eval(pb_file_path):
        nd = NanoporeDataset(DATASET, batch_size=BATCH_SIZE)
        test_set = nd.get_test_samples
        test_features, test_labels = test_set.get_next()
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            tf.local_variables_initializer().run()
            sess.run(test_set.initializer)
            # 加载pb文件中的图
            restore_model_pb(pb_file_path=pb_file_path)
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

                    test_labs.extend([i[0] for i in labs])
                    test_results.extend([i[0] for i in test_prediction])
                except tf.errors.OutOfRangeError:
                    score = roc_auc_score(test_labs, test_results)
                    print("model %s with auc score %.4f" % (os.path.split(pb_file_path)[-1], score))
                    del test_labs, test_results
                    break


if __name__ == "__main__":
    files = (os.path.join(PB_DIR, f) for f in os.listdir(PB_DIR))
    for f in files:
        Evaluator.testset_eval(f)
