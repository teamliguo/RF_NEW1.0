#include "LeafNodeWeightPredictionMethod.h"
#include <numeric>
#include <math.h>
#include "RegressionForestCommon.h"
#include "common/ProductFactory.h"

using namespace hiveRegressionForest;

hiveOO::CProductFactory<CLeafNodeWeightPredictionMethod> theCreator(KEY_WORDS::LEAF_NODE_WEIGHT_PREDICTION_METHOD);

float CLeafNodeWeightPredictionMethod::predictCertainTestV(const std::vector<float>& vTestFeatureInstance, float vTestResponse, const std::vector<CTree*>& vTreeSet)
{
	_ASSERTE(!vTestFeatureInstance.empty() && !vTreeSet.empty());

	int TreeNumber = vTreeSet.size();
	std::vector<float> NodeWeight(TreeNumber);
	std::vector<float> PredictValueOfTree(TreeNumber, 0.0f);
	std::vector<float> LeafNodeBias(TreeNumber, 0.0f);
	std::vector<const CNode*> LeafNodeSet(TreeNumber);
#pragma omp parallel for
	for (int i = 0; i < TreeNumber; ++i)
	{
		LeafNodeSet[i] = vTreeSet[i]->locateLeafNode(vTestFeatureInstance);
		NodeWeight[i] = LeafNodeSet[i]->getNodeWeight();
		PredictValueOfTree[i] = LeafNodeSet[i]->getNodeMeanV() * NodeWeight[i];
		LeafNodeBias[i] = abs(PredictValueOfTree[i] - vTestResponse);
	}
	float SumOfNodeWeight = std::accumulate(NodeWeight.begin(), NodeWeight.end(), 0.f);
	std::cout << "当前总权重" << SumOfNodeWeight << std::endl;
	__updataNodeWeigth(LeafNodeSet, LeafNodeBias, vTestResponse);
	float PredictValue = std::accumulate(PredictValueOfTree.begin(), PredictValueOfTree.end(), 0.0f);
	return PredictValue / SumOfNodeWeight;
}

//******************************************************************************
//FUNCTION:
void CLeafNodeWeightPredictionMethod::__updataNodeWeigth(std::vector<const CNode*>& vioLeafNodeSet, const std::vector<float>& vLeafNodeBias, float vTestResponse)
{
	_ASSERTE(!vioLeafNodeSet.empty() && !vLeafNodeBias.empty());

	for (auto i = 0; i < vioLeafNodeSet.size(); i++)
	{
		float NewWeight = (1.f + (0.04 - vLeafNodeBias[i] / vTestResponse))* vioLeafNodeSet[i]->getNodeWeight();//此处的4%有待商榷，目前是用相同条件下平均方法测出来的平均偏差率
		const_cast<CNode*>(vioLeafNodeSet[i])->setLeafNodeWeight(NewWeight);
	}
}