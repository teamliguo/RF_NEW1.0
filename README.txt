�򿪽��������Ҫ��������ĿRegressionForest����ִ��Testcase��


�����ļ���ر�ǩ˵����

1��BatchRunConfig.xml

- <IS_MODEL_EXIST> </IS_MODEL_EXIST> �����Ƿ��ȡ���л�ģ��
Ϊtrueʱ�Զ���ȡ���л�ģ�ͣ�Ϊfalseʱ���½�RFģ��

- <IS_SERIALIZE_MODEL> </IS_SERIALIZE_MODEL> �����Ƿ����ģ�����л�
���½�RFģ��ʱ��ͨ���ñ�ǩ�����Ƿ�ģ�����л�����


2��Config.xml

- <PREDICTION_METHOD> </PREDICTION_METHOD> ����Ԥ�ⷽʽ
���������֣�MEAN_PREDICTION_METHOD��ԭʼ����Ҷ�ӽڵ��ֵ��Ԥ�⣬LP_PREDICTION_METHOD/MP_PREDICTION_METHOD�Ƿֱ���ŷʽ���롢MPֵ������Ȩ����Ԥ�⣬VARIANCE_PREDICTION_METHOD��ʽ��Ҷ�ӽڵ������������Ӧֵ�������Ȩ�أ�INTERNAL_NODE_PREDICTION_METHODͨ�����Ե�����ڵ㷶Χ���ѡ���м�ڵ����Ҷ�ӽڵ�Ԥ�⡣

- <OUT_DIMENSION> </OUT_DIMENSION> ���ò��Ե���Գ����ڵ�AABBά��������ֵ
�������ǩ�е�INTERNAL_NODE_PREDICTION_METHODһ��ʹ�ã������Ե㳬���ڵ��AABBά�������ﵽ�����ֵʱ��ֹͣ���������ߣ��õ�ǰ�ڵ��ֵԤ�⡣


3��TrainingSetConfig.xml

- <IS_DIVIDE_FILE> </IS_DIVIDE_FILE> �Ƿ񽫲������ݸ���Ԥ�����û��ֳɺû������²������ݼ�

- <NEW_FILE_DATA_SIZE> </NEW_FILE_DATA_SIZE> �������������ݼ�������


������ǩͬRF�� �� ��ϸ���˵��.pdf �� ��һ�¡�