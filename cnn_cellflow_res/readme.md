# 实际数据对CNN模型评估
## 脚本文件说明
* evalV2.py
> 目前用于测试CNN模型的脚本文件。通过加载之前的模型，对cellflow_dataset的数据进行预测，得到logit的打分。

* interval_tree.py
> 将cellflow上的A位点对应到macs2软件peak calling得到的peak范围内。如果某个A碱基被peak覆盖，则设置标记为1（即认为是潜在甲基化位点），未覆盖则标记为0。
> 比较cellflow上A位点的预测结果与其label值相符程度，得到模型的预测效果。

## 文件说明
* peaks系列
> 是使用macs2得到的peak calling的结果，其中peaks的p-value阈值1e-5，peaks2的p-value的阈值1e-6。

* output系列
> 分别针对上面的peaks和peaks2得到的结果。

### 目前效果十分不好，主要问题：
* 与训练过程区别较大，阳性位点的logits值反而更小。这与之前训练过程恰好相反。
* merip-seq无法精确到单碱基，即使是label为1的cellflow数据也无法肯定就是阳性位点。