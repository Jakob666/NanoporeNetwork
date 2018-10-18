# -*- coding:utf-8 -*-

# 设置神经网络参数
BATCH_SIZE = 50
NUM_EPOCH = 1000

NANOPORE_HEIGHT = 17
NANOPORE_WIDTH = 4
NANOPORE_CHANNEL = 2

OUTPUT_NODE = 2

# 第一个卷积层的深度和窗口大小
CONV1_DEEP = 32
FLITER1_SIZE = 2
CONV1_STRIDE_HOR = 1
CONV1_STRIDE_VEC = 2

# 第二个卷积层的深度和窗口大小
CONV2_DEEP = 64
FLITER2_SIZE = 2
CONV2_STRIDE_HOR = 2
CONV2_STRIDE_VEC = 1

# 全连接层大小
FC_SIZE = 128
KEEP_PROB = 0.7

# 数据存放位置
DATASET = "/data/nanopore/dL_network/network/cnn_dataset/"
CELLFLOW_DATASET = "/data/nanopore/dL_network/network/cellflow_dataset/"

# 滑动平均模型的参数
MOVING_AVERAGE_DECAY = 0.99

# 学习率的设置
LEARNING_RATE_BASE = 0.01
SAMPLE_NUM = 60000
LEARNING_RATE_DECAY = 0.99

# 模型持久化设置
MODEL_SAVE_PATH = "/data/nanopore/dL_network/network/cnn_train_model/cnn_tf_model/"
MODEL_NAME = "nanopore_classifier.ckpt"
PB_DIR = "/data/nanopore/dL_network/network/cnn_train_model/cnn_tf_model_pb/"

# 用cellflow数据对模型进行评估时文件存储位置
CELLFLOW_EVAL = "/data/nanopore/dL_network/network/cnn_cellflow_res/cellflow_pred.tsv"
