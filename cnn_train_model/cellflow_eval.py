# -*- coding:utf-8 -*-
"""
@author: hbs
@date: 2018-9-27
description:
    用cellflow的数据进行网络的评估。只得到最终的softmax分类结果即可。
"""
import os
import sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
sys.path.append("/data/nanopore/dL_network/network")
sys.path.append("/data/nanopore/dL_network/network/cnn_train_model")
import tensorflow as tf
import pandas as pd
from load_dataset_cellflow import CellflowNanoporeDataset
from cnn_inference import inference
from config import *


def evaluate():
    x = tf.placeholder(dtype=tf.float32, shape=[None, NANOPORE_HEIGHT, NANOPORE_WIDTH, NANOPORE_CHANNEL], name="x")

    cnd = CellflowNanoporeDataset(CELLFLOW_DATASET, batch_size=BATCH_SIZE)
    data = cnd.get_data
    features, chrom, position = data.get_next()

    # 测试的前向运算过程
    cellflow_pred = inference(input_tensor=x, train=False, regularizer=None)
    y_logit = tf.nn.softmax(cellflow_pred, name="prediction")

    variable_average = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_average.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()
        sess.run(data.initializer)

        # get_checkpoint_state方法通过point文件找到目录中最新的模型的文件名
        ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            chrom_location = []
            positions = []
            predict_results = []
            while True:
                try:
                    feats, chrs, pos = sess.run([features, chrom, position])
                    eval_feed = {x: feats}
                    pred = sess.run(y_logit, feed_dict=eval_feed)
                    chrom_location.extend(chrs)
                    positions.extend(pos)
                    predict_results.extend([i[0] for i in pred])

                except tf.errors.OutOfRangeError:
                    break
            dec = lambda b: bytes.decode(b)
            chrom_location = list(map(dec, chrom_location))
            cnn_result = pd.DataFrame(data={"chr": chrom_location, "position": positions, "logit": predict_results})
            del chrom_location, positions, predict_results
            cnn_result.to_csv(CELLFLOW_EVAL, sep="\t", header=True, index=False, mode="w", encoding="utf-8")
        else:
            print("No checkpoint file found")

    return None


if __name__ == "__main__":
    evaluate()

