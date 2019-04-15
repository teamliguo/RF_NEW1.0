#include "PathNodeMethod.h"
#include <vector>
#include <stack>
#include <fstream>
#include "RegressionForestConfig.h"
#include "RegressionForestCommon.h"
#include "common/HiveCommonMicro.h"
#include "MpCompute.h"

using namespace hiveRegressionForest;

#define EPSILON 1.0e-6

//****************************************************************************************************
//FUNCTION:
float CPathNodeMethod::calOutNodeBound(const std::vector<float>& vFeature, const std::pair<std::vector<float>, std::vector<float>>& vFeatureRange)
{
	_ASSERTE(vFeature.size() == vFeatureRange.first.size());
	std::vector<float> Lower = vFeatureRange.first;
	std::vector<float> Upper = vFeatureRange.second;
	float sumOutRange = 0;
	for (int i = 0; i < Lower.size(); i++)
	{
		sumOutRange += (vFeature[i] >= Lower[i]) ? 0 : Lower[i] - vFeature[i];
		sumOutRange += (vFeature[i] <= Upper[i]) ? 0 : vFeature[i] - Upper[i];
	}
	return sumOutRange;
}

//****************************************************************************************************
//FUNCTION:
float CPathNodeMethod::calEuclideanDistanceFromNodeBound(const std::vector<float>& vFeature, const std::pair<std::vector<float>, std::vector<float>>& vFeatureRange)
{
	_ASSERTE(vFeature.size() == vFeatureRange.first.size());
	std::vector<float> Lower = vFeatureRange.first;
	std::vector<float> Upper = vFeatureRange.second;
	float EuclideanDistance = 0.f;
	for (int i = 0; i < Lower.size(); i++)
	{
		EuclideanDistance += (vFeature[i] >= Lower[i]) ? 0 : (Lower[i] - vFeature[i])*(Lower[i] - vFeature[i]);
		EuclideanDistance += (vFeature[i] <= Upper[i]) ? 0 : (vFeature[i] - Upper[i])*(vFeature[i] - Upper[i]);
	}
	return sqrt(EuclideanDistance);
}

//****************************************************************************************************
//FUNCTION:
float CPathNodeMethod::traversalPathPrediction(const CTree* vTree, const std::vector<float>& vFeature)
{
	_ASSERTE(vTree);
	std::stack<const CNode*> NodeStack;
	const CNode* TreeRoot = &vTree->getRoot();
	NodeStack.push(TreeRoot);
	float PredictResult = 0.f, WeightSum = 0.f;
	while (!NodeStack.empty())
	{
		const CNode* pCurrentNode = NodeStack.top();
		int SplitFeature = pCurrentNode->getBestSplitFeatureIndex();
		std::pair<std::vector<float>, std::vector<float>> FeatureRange = pCurrentNode->getFeatureRange();

		//针对测试点距每个结点划分维度的远近计算weight
		if (FeatureRange.first[SplitFeature] - vFeature[SplitFeature] > EPSILON)
		{
			float Len = (FeatureRange.second[SplitFeature] - FeatureRange.first[SplitFeature] + EPSILON) / (FeatureRange.first[SplitFeature] - vFeature[SplitFeature]);
			WeightSum += Len * pCurrentNode->getLevel();
			PredictResult += (pCurrentNode->getNodeMeanV()) * (Len*(pCurrentNode->getLevel()));
			if (pCurrentNode->isLeafNode())
				return PredictResult / WeightSum;
		}
		if (vFeature[SplitFeature] - FeatureRange.second[SplitFeature] > EPSILON)
		{
			float Len = (FeatureRange.second[SplitFeature] - FeatureRange.first[SplitFeature] + EPSILON) / (vFeature[SplitFeature] - FeatureRange.second[SplitFeature]);
			WeightSum += Len * pCurrentNode->getLevel();
			PredictResult += (pCurrentNode->getNodeMeanV()) * (Len*(pCurrentNode->getLevel()));
			if (pCurrentNode->isLeafNode())
				return PredictResult / WeightSum;
		}
		if (pCurrentNode->isLeafNode())
			return pCurrentNode->getNodeMeanV();

		NodeStack.pop();
		if (vFeature[pCurrentNode->getBestSplitFeatureIndex()] < pCurrentNode->getBestGap())
			NodeStack.push(&pCurrentNode->getLeftChild());
		else
			NodeStack.push(&pCurrentNode->getRightChild());
	}
}

//****************************************************************************************************
//FUNCTION:
float CPathNodeMethod::traverWithDistanceFromFeatureRange(const CTree* vTree, const std::vector<float>& vFeature)
{
	_ASSERTE(vTree);
	std::stack<const CNode*> NodeStack;
	const CNode* TreeRoot = &vTree->getRoot();
	NodeStack.push(TreeRoot);
	float PredictResult = 0.f, WeightSum = 0.f;
	while (!NodeStack.empty())
	{
		const CNode* pCurrentNode = NodeStack.top();
		int SplitFeature = pCurrentNode->getBestSplitFeatureIndex();
		std::pair<std::vector<float>, std::vector<float>> FeatureRange = pCurrentNode->getFeatureRange();

		float DistanceFromFeatureRange = 1.0f;
		for (int i = 0; i < vFeature.size(); i++)
		{
			if (FeatureRange.first[i] - vFeature[i] > EPSILON)
			{
				DistanceFromFeatureRange *= (FeatureRange.second[i] - FeatureRange.first[i] + EPSILON) / (FeatureRange.first[i] - vFeature[i]);
			}
			if (vFeature[i] - FeatureRange.second[i] > EPSILON)
			{
				DistanceFromFeatureRange *= (FeatureRange.second[i] - FeatureRange.first[i] + EPSILON) / (vFeature[i] - FeatureRange.second[i]);
			}
		}

		if (DistanceFromFeatureRange != 1.0f && fabs(DistanceFromFeatureRange) > 1e-6)
		{
			WeightSum += DistanceFromFeatureRange;
			PredictResult += DistanceFromFeatureRange*pCurrentNode->getNodeMeanV();
			std::cout << "level: " << pCurrentNode->getLevel() << "  NodeMean: " << pCurrentNode->getNodeMeanV() << "  weight: " << DistanceFromFeatureRange << " prediction: " << DistanceFromFeatureRange*pCurrentNode->getNodeMeanV() << std::endl;
		}

		if (pCurrentNode->isLeafNode())
		{
			return PredictResult == 0.f ? pCurrentNode->getNodeMeanV() : PredictResult / WeightSum;
		}

		NodeStack.pop();
		if (vFeature[pCurrentNode->getBestSplitFeatureIndex()] < pCurrentNode->getBestGap())
			NodeStack.push(&pCurrentNode->getLeftChild());
		else
			NodeStack.push(&pCurrentNode->getRightChild());
	}
}

//****************************************************************************************************
//FUNCTION:
float CPathNodeMethod::traversePathWithFeatureCentre(const CTree* vTree, const std::vector<float>& vFeature)
{
	_ASSERTE(vTree);
	std::stack<const CNode*> NodeStack;
	const CNode* TreeRoot = &vTree->getRoot();
	NodeStack.push(TreeRoot);
	float PredictResult = 0.f, WeightSum = 0.f;
	while (!NodeStack.empty())
	{
		const CNode* pCurrentNode = NodeStack.top();
		int SplitFeature = pCurrentNode->getBestSplitFeatureIndex();
		float SplitLocation = pCurrentNode->getBestGap();
		std::pair<std::vector<float>, std::vector<float>> FeatureRange = pCurrentNode->getFeatureRange();

		float DistanceFromFeatures = 0.0f;
		for (int i = 0; i < vFeature.size(); i++)
		{
			DistanceFromFeatures += std::abs(vFeature[i] - (FeatureRange.second[i] - FeatureRange.first[i]) / 2);
		}

		if (DistanceFromFeatures != 0.f)
		{
			WeightSum += 1 / DistanceFromFeatures;
			PredictResult += 1 / DistanceFromFeatures*pCurrentNode->getNodeMeanV();
			//std::cout << "level: " << pCurrentNode->getLevel() << "  NodeMean: " << pCurrentNode->getNodeMeanV() << "  weight: " << 1 / DistanceFromFeatures << std::endl;
		}

		if (pCurrentNode->isLeafNode())
		{
			return PredictResult == 0.f ? pCurrentNode->getNodeMeanV() : PredictResult / WeightSum;
		}

		NodeStack.pop();
		if (vFeature[pCurrentNode->getBestSplitFeatureIndex()] < pCurrentNode->getBestGap())
			NodeStack.push(&pCurrentNode->getLeftChild());
		else
			NodeStack.push(&pCurrentNode->getRightChild());
	}
}


//****************************************************************************************************
//FUNCTION:
float CPathNodeMethod::traverWithDistanceFromFeaturesCentre(const CTree* vTree, const std::vector<float>& vFeature)
{
	_ASSERTE(vTree);
	std::stack<const CNode*> NodeStack;
	const CNode* TreeRoot = &vTree->getRoot();
	NodeStack.push(TreeRoot);
	float PredictResult = 0.f, WeightSum = 0.f;
	while (!NodeStack.empty())
	{
		const CNode* pCurrentNode = NodeStack.top();
		int SplitFeature = pCurrentNode->getBestSplitFeatureIndex();
		float SplitLocation = pCurrentNode->getBestGap();
		std::pair<std::vector<float>, std::vector<float>> FeatureRange = pCurrentNode->getFeatureRange();

		float DistanceFromFeatures = 1.0f;
		for (int i = 0; i < vFeature.size(); i++)
		{
			if (FeatureRange.second[i] == FeatureRange.first[i] && vFeature[i] != FeatureRange.first[i])
				DistanceFromFeatures *= 1.0f / std::abs(vFeature[i]);
			else if (FeatureRange.second[i] == FeatureRange.first[i] && vFeature[i] == FeatureRange.first[i])
				DistanceFromFeatures *= 1.0f;
			else
				DistanceFromFeatures *= (FeatureRange.second[i] - FeatureRange.first[i]) / (std::abs(vFeature[i] - FeatureRange.second[i]) + std::abs(vFeature[i] - FeatureRange.first[i]));
		}
		//DistanceFromFeatures = DistanceFromFeatures - 1.0;
		if (DistanceFromFeatures != 1.0f)
		{
			WeightSum += DistanceFromFeatures;
			PredictResult += DistanceFromFeatures*pCurrentNode->getNodeMeanV();
			//std::cout << "level: " << pCurrentNode->getLevel() << "  NodeMean: " << pCurrentNode->getNodeMeanV() << "  weight: " << DistanceFromFeatures << " prediction: " << DistanceFromFeatures*pCurrentNode->getNodeMeanV() << std::endl;
		}

		if (pCurrentNode->isLeafNode())
		{
			return PredictResult == 0.f ? pCurrentNode->getNodeMeanV() : PredictResult / WeightSum;
		}

		NodeStack.pop();
		if (vFeature[pCurrentNode->getBestSplitFeatureIndex()] < pCurrentNode->getBestGap())
			NodeStack.push(&pCurrentNode->getLeftChild());
		else
			NodeStack.push(&pCurrentNode->getRightChild());
	}
}

//****************************************************************************************************
//FUNCTION:
float CPathNodeMethod::predictWithMonteCarlo(const CNode& vCurLeafNode, const std::vector<float>& vFeature)
{
	CTrainingSet* pTrainingSet = CTrainingSet::getInstance();
	std::vector<int> DataIndex = (const_cast<CNode*>(&vCurLeafNode))->getNodeDataIndexV();
	sort(DataIndex.begin(), DataIndex.end());

	std::vector<float> Weights(DataIndex.size(), 0);

	const CRegressionForestConfig* pRegressionForestConfig = CRegressionForestConfig::getInstance();
	bool OmpParallelSig = pRegressionForestConfig->getAttribute<bool>(KEY_WORDS::OPENMP_PARALLEL_BUILD_TREE);

#pragma omp parallel for if (OmpParallelSig)
	for (int k = 0; k < DataIndex.size(); k++)
	{
		std::vector<float> DataFeatures = pTrainingSet->getFeatureInstanceAt(DataIndex[k]);
		for (int i = 0; i < vFeature.size(); i++)
		{
			Weights[k] += fabs(__computeCDF(vFeature[i], DataFeatures[i]));
		}
	}

	int MinCDFIndex = 0;
	for (int i = 1; i < Weights.size(); i++)
	{
		if (Weights[MinCDFIndex] > Weights[i]) MinCDFIndex = i;
	}

	return pTrainingSet->getResponseValueAt(DataIndex[MinCDFIndex]);
}

//****************************************************************************************************
//FUNCTION:
float CPathNodeMethod::__computeCDF(float vFirst, float vSecond)
{
	return 0.5*(erfc(-vSecond*sqrt(0.5)) - erfc(-vFirst*sqrt(0.5)));
}

//****************************************************************************************************
//FUNCTION:
float CPathNodeMethod::__calInternalNodeMeanValue(const std::vector<int>& vDataSetIndexSet)
{
	CTrainingSet* pTrainingSet = CTrainingSet::getInstance();
	int DataSize = vDataSetIndexSet.size();
	float MeanValue = 0.f;
	for (int i = 0; i < DataSize; ++i)
	{
		MeanValue += pTrainingSet->getResponseValueAt(vDataSetIndexSet[i]);
	}
	return MeanValue / DataSize;
}

//****************************************************************************************************
//FUNCTION:
bool CPathNodeMethod::__isTotalInBoundBox(const std::vector<float>& vTestPoint, const std::vector<float>& vLow, const std::vector<float>& vHeigh)
{
	for (int i = 0; i < vTestPoint.size(); ++i)
	{
		if (vTestPoint[i] < vLow[i] || vTestPoint[i] > vHeigh[i])
			return false;
	}
	return true;
}

//****************************************************************************************************
//FUNCTION:
bool CPathNodeMethod::__isOneMoreOutBoundRange(const std::vector<float>& vTestPoint, const std::vector<float>& vLow, const std::vector<float>& vHeigh, int vOutDimension)
{
	int count = 0;
	for (int i = 0; i < vTestPoint.size(); ++i)
	{
		if (vTestPoint[i] < vLow[i] || vTestPoint[i] > vHeigh[i])
			count++;
		if (count >= vOutDimension)
			return false;
	}
	return true;
}

//****************************************************************************************************
//FUNCTION:
std::vector<int> CPathNodeMethod::__calInternalNodeDataIndex(const CNode* vNode)
{
	if (vNode->isLeafNode())
		return vNode->getNodeDataIndexV();
	const CNode* LeftNode = const_cast<CNode*>(&vNode->getLeftChild());
	const CNode* RightNode = const_cast<CNode*>(&vNode->getRightChild());
	std::vector<int> LeftDataIndex = __calInternalNodeDataIndex(LeftNode);
	std::vector<int> RightNodeDataIndex = __calInternalNodeDataIndex(RightNode);
	LeftDataIndex.insert(LeftDataIndex.end(), RightNodeDataIndex.begin(), RightNodeDataIndex.end());
	return LeftDataIndex;
}

//****************************************************************************************************
//FUNCTION:
float CPathNodeMethod::predictWithMinMPOnWholeDimension(const CTree* vTree, const std::vector<float>& vFeature)
{
	_ASSERTE(vTree);
	std::stack<const CNode*> NodeStack;
	const CNode* TreeRoot = &vTree->getRoot();
	NodeStack.push(TreeRoot);
	CMpCompute* pMpCompute = nullptr;
	CTrainingSet* pTrainingSet = CTrainingSet::getInstance();
	int OutDimension = CTrainingSetConfig::getInstance()->getAttribute<int>(KEY_WORDS::OUT_DIMENSION);
	while (!NodeStack.empty())
	{
		const CNode* pCurrentNode = NodeStack.top();
		std::pair<std::vector<float>, std::vector<float>> FeatureRange = pCurrentNode->getFeatureRange();
		if (OutDimension <= 0) OutDimension = 1;
		if (OutDimension > vFeature.size()) OutDimension = vFeature.size();
		bool IsInBox = __isOneMoreOutBoundRange(vFeature, FeatureRange.first, FeatureRange.second, OutDimension);
		if (IsInBox)
		{
			if (pCurrentNode->isLeafNode())
			{
				std::pair<int, float> MinMPAndIndex = pMpCompute->calMinMPAndIndex(vTree, pCurrentNode->getNodeDataIndexV(), vFeature);
				return pTrainingSet->getResponseValueAt(MinMPAndIndex.first);
			}
			NodeStack.pop();
			if (vFeature[pCurrentNode->getBestSplitFeatureIndex()] < pCurrentNode->getBestGap())
				NodeStack.push(&pCurrentNode->getLeftChild());
			else
				NodeStack.push(&pCurrentNode->getRightChild());
		}
		else
		{
			std::vector<int> DataIndex = pCurrentNode->isLeafNode() ? pCurrentNode->getNodeDataIndexV() : __calInternalNodeDataIndex(pCurrentNode);
			std::pair<int, float> MinMPAndIndex = pMpCompute->calMinMPAndIndex(vTree, DataIndex, vFeature);
			return pTrainingSet->getResponseValueAt(MinMPAndIndex.first);
		}
	}
	_SAFE_DELETE(pMpCompute);
}

//****************************************************************************************************
//FUNCTION:
float CPathNodeMethod::prediceWithInternalNode(const CTree* vTree, const std::vector<float>& vFeature)
{
	_ASSERTE(vTree);
	std::stack<const CNode*> NodeStack;
	const CNode* TreeRoot = &vTree->getRoot();
	NodeStack.push(TreeRoot);
	int OutDimension = CTrainingSetConfig::getInstance()->getAttribute<int>(KEY_WORDS::OUT_DIMENSION);
	while (!NodeStack.empty())
	{
		const CNode* pCurrentNode = NodeStack.top();
		std::pair<std::vector<float>, std::vector<float>> FeatureRange = pCurrentNode->getFeatureRange();
		if (OutDimension <= 0) OutDimension = 1;
		if (OutDimension > vFeature.size()) OutDimension = vFeature.size();
		bool IsInBox = __isOneMoreOutBoundRange(vFeature, FeatureRange.first, FeatureRange.second, OutDimension);
		if (IsInBox)
		{
			if (pCurrentNode->isLeafNode())
			{
				return pCurrentNode->getNodeMeanV();
			}
			NodeStack.pop();
			if (vFeature[pCurrentNode->getBestSplitFeatureIndex()] < pCurrentNode->getBestGap())
				NodeStack.push(&pCurrentNode->getLeftChild());
			else
				NodeStack.push(&pCurrentNode->getRightChild());
		}
		else
		{
			return pCurrentNode->getNodeMeanV();
		}
	}
}
