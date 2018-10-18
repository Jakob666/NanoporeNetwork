## 启动开发环境 
> 创建nanopore项目的深度学习网络，运行时需要正在 通过如下命令启动环境
'''python
source activate tensorflow
'''

## 各脚本功能说明
* form_dataset.py
> 用于构建训练所需的TFRecord数据集。将之前得到的 甲基化和非甲基化的序列生成相应的数据格式。同时将数据分为训练集、测试集和验证集。
> 这些数据存放在dataset文件目录中，该数据用于训练DNN神经网络。

* form_dataset_cnn.py
> 继承并修改了form_dataset.py中的类，生成可以用于CNN网络训练的数据集。

* form_dataset_cellflow.py
> 继承了form_dataset_cnn.py中的类，并对其进行修改，生成的文件是cellflow的数据，用于在实际条件下对CNN模型进行评测。

* load_dataset系列文件
> 用于在训练过程中读取之前生成的TFRecord文件中的数据，使用时请导出该模块的NanoporeDataset类，并使用get_meth_data和get_unmeth_data
> 属性方法获取甲基化和非甲基化的训练、验证和测试集。该测试集返回的是TensorFlow的生成器。

## 各目录说明
* dataset系列目录
> 将甲基化和非甲基化数据分为 train、test和valid三部分，各部分分别占总数据的70%、20%和10%。以TFRecord格式存放。

* dnn_train_model
> 存放有用于分类的DNN模型，目前效果不是很好（2018.9.10）

* cnn_train_model
> 存放有用于分类的CNN模型，目前针对该模型使用cellflow的数据进行准确率的判定。

* cnn_cellflow_res
> 存放cnn模型对cellflow数据的预测结果，并与peak calling结果比对，得到模型在实际情况下的准确率。

* random_forest_model
> 存放有用于分类的RF模型（目前效果比dnn神经网络好，不如CNN）


