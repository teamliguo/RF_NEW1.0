打开解决方案后，要先生成项目RegressionForest，再执行Testcase。


配置文件相关标签说明：

1、BatchRunConfig.xml

- <IS_MODEL_EXIST> </IS_MODEL_EXIST> 控制是否读取序列化模型
为true时自动读取序列化模型，为false时将新建RF模型

- <IS_SERIALIZE_MODEL> </IS_SERIALIZE_MODEL> 设置是否进行模型序列化
当新建RF模型时，通过该标签控制是否将模型序列化出来


2、Config.xml

- <PREDICTION_METHOD> </PREDICTION_METHOD> 设置预测方式
这里有五种，MEAN_PREDICTION_METHOD是原始的用叶子节点均值做预测，LP_PREDICTION_METHOD/MP_PREDICTION_METHOD是分别用欧式距离、MP值做树的权重做预测，VARIANCE_PREDICTION_METHOD方式用叶子节点中特征方差及响应值方差计算权重，INTERNAL_NODE_PREDICTION_METHOD通过测试点落入节点范围情况选择中间节点或者叶子节点预测。

- <OUT_DIMENSION> </OUT_DIMENSION> 设置测试点可以超出节点AABB维度数的阈值
与上面标签中的INTERNAL_NODE_PREDICTION_METHOD一起使用，当测试点超出节点的AABB维度数量达到这个阈值时将停止继续向下走，用当前节点均值预测。


3、TrainingSetConfig.xml

- <IS_DIVIDE_FILE> </IS_DIVIDE_FILE> 是否将测试数据根据预测结果好坏分成好坏两个新测试数据集

- <NEW_FILE_DATA_SIZE> </NEW_FILE_DATA_SIZE> 设置两个新数据集数据量


其他标签同RF组 “ 详细情况说明.pdf ” 中一致。