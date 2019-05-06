#include "LeafNodeWeightPredictionMethod.h"
#include <numeric>
#include <math.h>
#include <algorithm>
#include "RegressionForestCommon.h"
#include "common/ProductFactory.h"

using namespace hiveRegressionForest;

hiveOO::CProductFactory<CLeafNodeWeightPredictionMethod> theCreator(KEY_WORDS::LEAF_NODE_WEIGHT_PREDICTION_METHOD);

//******************************************************************************
//FUNCTION:
float CLeafNodeWeightPredictionMethod::predictCertainTestV(const std::vector<float>& vTestFeatureInstance, float vTestResponse, const std::vector<CTree*>& vTreeSet)
{
	_ASSERTE(!vTestFeatureInstance.empty() && !vTreeSet.empty());

	int TreeNumber = vTreeSet.size();
	std::vector<float> NodeWeight(TreeNumber);
	std::vector<float> PredictValueOfTree(TreeNumber, 0.0f);
	std::vector<float> LeafNodeBiasRatio(TreeNumber, 0.0f);
	std::vector<const CNode*> LeafNodeSet(TreeNumber);
#pragma omp parallel for
	for (int i = 0; i < TreeNumber; ++i)
	{
		LeafNodeSet[i] = vTreeSet[i]->locateLeafNode(vTestFeatureInstance);
		NodeWeight[i] = LeafNodeSet[i]->getNodeWeight();
		PredictValueOfTree[i] = LeafNodeSet[i]->getNodeMeanV() * NodeWeight[i];
		LeafNodeBiasRatio[i] = abs(PredictValueOfTree[i] - vTestResponse) / vTestResponse;
	}
	float SumOfNodeWeight = std::accumulate(NodeWeight.begin(), NodeWeight.end(), 0.f);
	std::cout << "当前总权重" << SumOfNodeWeight << std::endl;
	float PredictValue = std::accumulate(PredictValueOfTree.begin(), PredictValueOfTree.end(), 0.0f) / SumOfNodeWeight;
	__updataNodeWeigthByChangeThreshold(LeafNodeSet, LeafNodeBiasRatio, vTestResponse, PredictValue);
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
	std::vector<float> LeafNodeBiasRatio(TreeNumber, 0.0f);
	std::vector<const CNode*> LeafNodeSet(TreeNumber);
//#pragma omp parallel for
	for (int k = 0; k < vOOBFeatureSet.size(); ++k)
	{
		std::cout << "PrePredict " << k << " th OOB Test" << std::endl;
		for (int i = 0; i < TreeNumber; ++i)
		{
			LeafNodeSet[i] = vTreeSet[i]->locateLeafNode(vOOBFeatureSet[k]);
			const_cast<CNode*>(LeafNodeSet[i])->setIsUsed(true);
			NodeWeight[i] = LeafNodeSet[i]->getNodeWeight();
			PredictValueOfTree[i] = LeafNodeSet[i]->getNodeMeanV() * NodeWeight[i];
			LeafNodeBiasRatio[i] = abs(PredictValueOfTree[i] - vOOBResponseSet[k])/ vOOBResponseSet[k];
			/*if (LeafNodeBiasRatio[i] > 1)
				std::cout << "偏差太大了！" << std::endl;*/
		}
		float SumOfNodeWeight = std::accumulate(NodeWeight.begin(), NodeWeight.end(), 0.f);
		std::cout << "SumOfNodeWeight = " << SumOfNodeWeight << std::endl;
		float PredictValue = std::accumulate(PredictValueOfTree.begin(), PredictValueOfTree.end(), 0.0f) / SumOfNodeWeight;
		__updataNodeWeigthByChangeThreshold(LeafNodeSet, LeafNodeBiasRatio, vOOBResponseSet[k], PredictValue);
	}

	for (int i = 0; i < TreeNumber; ++i)
	{
		vTreeSet[i]->updateUnusedNodeWeight();
	}
	__resetMember();
}

//******************************************************************************
//FUNCTION:
float CLeafNodeWeightPredictionMethod::predictCertainTestBlockV(const std::vector<float>& vTestFeatureInstance, float vTestResponse, const std::vector<CTree*>& vTreeSet)
{
	_ASSERTE(!vTestFeatureInstance.empty() && !vTreeSet.empty());

	int TreeNumber = vTreeSet.size();
	std::vector<float> NodeBlockWeight(TreeNumber);
	std::vector<float> PredictValueOfTree(TreeNumber, 0.0f);
	std::vector<float> LeafNodeBiasRatio(TreeNumber, 0.0f);
	std::vector<const CNode*> LeafNodeSet(TreeNumber);
	std::vector<std::bitset<16>> TestBlockSet(TreeNumber);
//#pragma omp parallel for
	for (int i = 0; i < TreeNumber; ++i)
	{
		LeafNodeSet[i] = vTreeSet[i]->locateLeafNode(vTestFeatureInstance);
		__calBlockId(LeafNodeSet[i], vTestFeatureInstance, TestBlockSet[i]);
		std::unordered_map<std::bitset<16>, float> CurrentNodeBlockWeightSet = LeafNodeSet[i]->getBlockIdWithWeight();
		if(CurrentNodeBlockWeightSet.find(TestBlockSet[i]) != CurrentNodeBlockWeightSet.end())
			NodeBlockWeight[i] = CurrentNodeBlockWeightSet[TestBlockSet[i]];
		else
		{
			NodeBlockWeight[i] = LeafNodeSet[i]->getNodeWeight();
			const_cast<CNode*>(LeafNodeSet[i])->changeBlockIdWithWeight({ TestBlockSet[i], NodeBlockWeight[i] });
		}
		PredictValueOfTree[i] = LeafNodeSet[i]->getNodeMeanV() * NodeBlockWeight[i];
		LeafNodeBiasRatio[i] = abs(PredictValueOfTree[i] - vTestResponse)/ vTestResponse;
	}
	float SumOfNodeWeight = std::accumulate(NodeBlockWeight.begin(), NodeBlockWeight.end(), 0.f);
	std::cout << "当前总权重" << SumOfNodeWeight << std::endl;
	float PredictValue = std::accumulate(PredictValueOfTree.begin(), PredictValueOfTree.end(), 0.0f) / SumOfNodeWeight;
	__updataBlockWeigthByChangeThreshold(LeafNodeSet, LeafNodeBiasRatio, TestBlockSet, vTestResponse, PredictValue);
	return PredictValue;
}

//******************************************************************************
//FUNCTION:
void CLeafNodeWeightPredictionMethod::__updataBlockWeigthByChangeThreshold(std::vector<const CNode*>& vioLeafNodeSet, const std::vector<float>& vLeafNodeBiasRatio, const std::vector<std::bitset<16>>& vBlockIdSet, float vTestResponse, float vPredictValue)
{
	_ASSERTE(!vioLeafNodeSet.empty() && !vLeafNodeBiasRatio.empty() && !vBlockIdSet.empty());
	m_TestInstanceCount++;
	m_SumBiasRatio += abs(vPredictValue - vTestResponse) / vTestResponse;
	float Threshold = m_SumBiasRatio / m_TestInstanceCount;
	for (auto i = 0; i < vioLeafNodeSet.size(); i++)
	{
		const std::unordered_map<std::bitset<16>, float>& NodeBlockMap = vioLeafNodeSet[i]->getBlockIdWithWeight();
		auto iterBlockIdWithWeight = NodeBlockMap.find(vBlockIdSet[i]);
		float NewWeight = (1.f + (Threshold - vLeafNodeBiasRatio[i]))* (iterBlockIdWithWeight->second);
		const_cast<CNode*>(vioLeafNodeSet[i])->changeBlockIdWithWeight({ vBlockIdSet[i], NewWeight });
		float SumBlockWeight = 0.f;
		for (auto iter : NodeBlockMap)
			SumBlockWeight += iter.second;
		float AvreageWeight = SumBlockWeight / NodeBlockMap.size();
		const_cast<CNode*>(vioLeafNodeSet[i])->setLeafNodeWeight(AvreageWeight);
	}
}

//******************************************************************************
//FUNCTION:
void CLeafNodeWeightPredictionMethod::__updataNodeWeigthByConstantThreshold(std::vector<const CNode*>& vioLeafNodeSet, const std::vector<float>& vLeafNodeBiasRatio)
{
	_ASSERTE(!vioLeafNodeSet.empty() && !vLeafNodeBiasRatio.empty());
	for (auto i = 0; i < vioLeafNodeSet.size(); i++)
	{
		float NewWeight = (1.f + (0.04f - vLeafNodeBiasRatio[i]))* vioLeafNodeSet[i]->getNodeWeight();//此处的4%有待商榷，目前是用相同条件下平均方法测出来的平均偏差率
		const_cast<CNode*>(vioLeafNodeSet[i])->setLeafNodeWeight(NewWeight);
	}
}

//******************************************************************************
//FUNCTION:
void CLeafNodeWeightPredictionMethod::__updataNodeWeigthByChangeThreshold(std::vector<const CNode*>& vioLeafNodeSet, const std::vector<float>& vLeafNodeBiasRatio, float vTestResponse, float vPredictValue)
{
	_ASSERTE(!vioLeafNodeSet.empty() && !vLeafNodeBiasRatio.empty());
	m_TestInstanceCount++;
	m_SumBiasRatio += abs(vPredictValue - vTestResponse) / vTestResponse;
	float Threshold = m_SumBiasRatio / m_TestInstanceCount;
	for (auto i = 0; i < vioLeafNodeSet.size(); i++)
	{
		float NewWeight = (1.f + (Threshold - vLeafNodeBiasRatio[i]))* vioLeafNodeSet[i]->getNodeWeight();
		/*if (NewWeight < 0.000001)
			std::cout << "权重太小了！" << std::endl;*/
		const_cast<CNode*>(vioLeafNodeSet[i])->setLeafNodeWeight(NewWeight);
	}
}

void CLeafNodeWeightPredictionMethod::__calBlockId(const CNode * vLeafNode, const std::vector<float>& vTestFeatureInstance, std::bitset<16>& voBlockId)
{
	_ASSERTE(vLeafNode && !vTestFeatureInstance.empty());
	std::pair<std::vector<float>, std::vector<float>> NodeSplitRange;
	NodeSplitRange = vLeafNode->getFeatureSplitRange();
	for (auto i = 0; i < NodeSplitRange.first.size(); i++)
	{
		float MidLocation = (NodeSplitRange.first[i] + NodeSplitRange.second[i]) / 2;
		if (vTestFeatureInstance[i] < MidLocation)
			voBlockId.set(i, 0);
		else
			voBlockId.set(i, 1);
	}
}
