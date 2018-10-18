# -*- coding: utf-8 -*-

# 设置各神经网络层的节点数目
INPUT_NODE = 14
LAYER1_NODE = 50
LAYER2_NODE = 10
OUTPUT_NODE = 2

# 设置神经网络优化参数
BATCH_SIZE = 100
# 衰减的学习率的设置
LEARNING_RATE_BASE = 0.1
LEARNING_RATE_DECAY = 0.99
# 正则化系数
REGULARIZATION_RATE = 0.005
TRAINING_STEPS = 1000
# 滑动平均模型参数
MOVING_AVERAGE_DECAY = 0.99
# 训练过程的epoch
NUM_EPOCH = 1000

# 注明数据集的位置
DATASET = "/data/nanopore/dL_network/network/dnn_dataset/"

# 存放ckpt模型和pb模型的文件目录
MODEL_SAVE_PATH = "/data/nanopore/dL_network/network/dnn_train_model/dnn_tf_model/"
MODEL_NAME = "nanopore_classifier.ckpt"
PB_DIR = "/data/nanopore/dL_network/network/dnn_train_model/dnn_tf_model_pb/"

SAMPLE_NUM = 40000

# 日志设置
LOGGING_CONF = "/data/nanopore/dL_network/network/dnn_train_model/logging.conf.yaml"
