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
	float PredictValue = std::accumulate(PredictValueOfTree.begin(), PredictValueOfTree.end(), 0.0f) / SumOfNodeWeight;
	__updataNodeWeigthByChangeThreshold(LeafNodeSet, LeafNodeBias, vTestResponse, PredictValue);

	return PredictValue;
}

//******************************************************************************
//FUNCTION:
void CLeafNodeWeightPredictionMethod::prePredictOOBDataV(const std::vector<std::vector<float>>& vOOBFeatureSet, const std::vector<float>& vOOBResponseSet, const std::vector<CTree*>& vTreeSet)
{
	_ASSERTE(!vOOBFeatureSet.empty() && !vOOBResponseSet.empty() && !vTreeSet.empty());

	int TreeNumber = vTreeSet.size();
	std::vector<float> NodeWeight(TreeNumber);
	std::vector<float> PredictValueOfTree(TreeNumber, 0.0f);
	std::vector<float> LeafNodeBias(TreeNumber, 0.0f);
	std::vector<const CNode*> LeafNodeSet(TreeNumber);
#pragma omp parallel for
	for (int k = 0; k < vOOBFeatureSet.size(); ++k)
	{
		std::cout << "PrePredict " << k << " th OOB Test" << std::endl;
		for (int i = 0; i < TreeNumber; ++i)
		{
			LeafNodeSet[i] = vTreeSet[i]->locateLeafNode(vOOBFeatureSet[k]);
			const_cast<CNode*>(LeafNodeSet[i])->setIsUsed(true);
			NodeWeight[i] = LeafNodeSet[i]->getNodeWeight();
			PredictValueOfTree[i] = LeafNodeSet[i]->getNodeMeanV() * NodeWeight[i];
			LeafNodeBias[i] = abs(PredictValueOfTree[i] - vOOBResponseSet[k]);
		}
		float SumOfNodeWeight = std::accumulate(NodeWeight.begin(), NodeWeight.end(), 0.f);
		float PredictValue = std::accumulate(PredictValueOfTree.begin(), PredictValueOfTree.end(), 0.0f) / SumOfNodeWeight;
		__updataNodeWeigthByChangeThreshold(LeafNodeSet, LeafNodeBias, vOOBResponseSet[k], PredictValue);
	}

	for (int i = 0; i < TreeNumber; ++i)
	{
		vTreeSet[i]->updateUnusedNodeWeight();
	}
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

//******************************************************************************
//FUNCTION:
void CLeafNodeWeightPredictionMethod::__updataNodeWeigthByChangeThreshold(std::vector<const CNode*>& vioLeafNodeSet, const std::vector<float>& vLeafNodeBias, float vTestResponse, float vPredictValue)
{
	_ASSERTE(!vioLeafNodeSet.empty() && !vLeafNodeBias.empty());
	m_TestInstanceCount++;
	m_SumBiasRatio += abs(vPredictValue - vTestResponse) / vTestResponse;
	float Threshold = m_SumBiasRatio / m_TestInstanceCount;
	for (auto i = 0; i < vioLeafNodeSet.size(); i++)
	{
		float NewWeight = (1.f + (Threshold - vLeafNodeBias[i] / vTestResponse))* vioLeafNodeSet[i]->getNodeWeight();
		const_cast<CNode*>(vioLeafNodeSet[i])->setLeafNodeWeight(NewWeight);
	}
}