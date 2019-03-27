#include "WeightedPathNodeMethod.h"
#include <vector>
#include <stack>
#include <fstream>
#include "RegressionForestConfig.h"
#include "RegressionForestCommon.h"

using namespace hiveRegressionForest;

#define EPSILON 1.0e-6

//****************************************************************************************************
//FUNCTION:
float CWeightedPathNodeMethod::calOutNodeBound(const std::vector<float>& vFeature, const std::pair<std::vector<float>, std::vector<float>>& vFeatureRange)
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
float CWeightedPathNodeMethod::calEuclideanDistanceFromNodeBound(const std::vector<float>& vFeature, const std::pair<std::vector<float>, std::vector<float>>& vFeatureRange)
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
float CWeightedPathNodeMethod::traversalPathPrediction(const CTree* vTree, const std::vector<float>& vFeatures)
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
		if (FeatureRange.first[SplitFeature] - vFeatures[SplitFeature] > EPSILON)
		{
			float Len = (FeatureRange.second[SplitFeature] - FeatureRange.first[SplitFeature] + EPSILON) / (FeatureRange.first[SplitFeature] - vFeatures[SplitFeature]);
			WeightSum += Len * pCurrentNode->getLevel();
			PredictResult += (pCurrentNode->getNodeMeanV()) * (Len*(pCurrentNode->getLevel()));
			if (pCurrentNode->isLeafNode())
				return PredictResult / WeightSum;
		}
		if (vFeatures[SplitFeature] - FeatureRange.second[SplitFeature] > EPSILON)
		{
			float Len = (FeatureRange.second[SplitFeature] - FeatureRange.first[SplitFeature] + EPSILON) / (vFeatures[SplitFeature] - FeatureRange.second[SplitFeature]);
			WeightSum += Len * pCurrentNode->getLevel();
			PredictResult += (pCurrentNode->getNodeMeanV()) * (Len*(pCurrentNode->getLevel()));
			if (pCurrentNode->isLeafNode())
				return PredictResult / WeightSum;
		}
		if (pCurrentNode->isLeafNode())
			return pCurrentNode->getNodeMeanV();

		NodeStack.pop();
		if (vFeatures[pCurrentNode->getBestSplitFeatureIndex()] < pCurrentNode->getBestGap())
			NodeStack.push(&pCurrentNode->getLeftChild());
		else
			NodeStack.push(&pCurrentNode->getRightChild());
	}
}

//****************************************************************************************************
//FUNCTION:
float CWeightedPathNodeMethod::traverWithDistanceFromFeatureRange(const CTree* vTree, const std::vector<float>& vFeatures)
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
		for (int i = 0; i < vFeatures.size(); i++)
		{
			if (FeatureRange.first[i] - vFeatures[i] > EPSILON)
			{
				DistanceFromFeatureRange *= (FeatureRange.second[i] - FeatureRange.first[i] + EPSILON) / (FeatureRange.first[i] - vFeatures[i]);
			}
			if (vFeatures[i] - FeatureRange.second[i] > EPSILON)
			{
				DistanceFromFeatureRange *= (FeatureRange.second[i] - FeatureRange.first[i] + EPSILON) / (vFeatures[i] - FeatureRange.second[i]);
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
		if (vFeatures[pCurrentNode->getBestSplitFeatureIndex()] < pCurrentNode->getBestGap())
			NodeStack.push(&pCurrentNode->getLeftChild());
		else
			NodeStack.push(&pCurrentNode->getRightChild());
	}
}

//****************************************************************************************************
//FUNCTION:
float CWeightedPathNodeMethod::traversePathWithFeatureCentre(const CTree* vTree, const std::vector<float>& vFeatures)
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
		for (int i = 0; i < vFeatures.size(); i++)
		{
			DistanceFromFeatures += std::abs(vFeatures[i] - (FeatureRange.second[i] - FeatureRange.first[i]) / 2);
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
		if (vFeatures[pCurrentNode->getBestSplitFeatureIndex()] < pCurrentNode->getBestGap())
			NodeStack.push(&pCurrentNode->getLeftChild());
		else
			NodeStack.push(&pCurrentNode->getRightChild());
	}
}


//****************************************************************************************************
//FUNCTION:
float CWeightedPathNodeMethod::traverWithDistanceFromFeaturesCentre(const CTree* vTree, const std::vector<float>& vFeatures)
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
		for (int i = 0; i < vFeatures.size(); i++)
		{
			if (FeatureRange.second[i] == FeatureRange.first[i] && vFeatures[i] != FeatureRange.first[i])
				DistanceFromFeatures *= 1.0f / std::abs(vFeatures[i]);
			else if (FeatureRange.second[i] == FeatureRange.first[i] && vFeatures[i] == FeatureRange.first[i])
				DistanceFromFeatures *= 1.0f;
			else
				DistanceFromFeatures *= (FeatureRange.second[i] - FeatureRange.first[i]) / (std::abs(vFeatures[i] - FeatureRange.second[i]) + std::abs(vFeatures[i] - FeatureRange.first[i]));
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
		if (vFeatures[pCurrentNode->getBestSplitFeatureIndex()] < pCurrentNode->getBestGap())
			NodeStack.push(&pCurrentNode->getLeftChild());
		else
			NodeStack.push(&pCurrentNode->getRightChild());
	}
}

//****************************************************************************************************
//FUNCTION:
float CWeightedPathNodeMethod::predictWithMonteCarlo(const CNode& vCurLeafNode, const std::vector<float>& vFeatures)
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
		for (int i = 0; i < vFeatures.size(); i++)
		{
			Weights[k] += fabs(__computeCDF(vFeatures[i], DataFeatures[i]));
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
float CWeightedPathNodeMethod::__computeCDF(float vFirst, float vSecond)
{
	return 0.5*(erfc(-vSecond*sqrt(0.5)) - erfc(-vFirst*sqrt(0.5)));
}

//****************************************************************************************************
//FUNCTION:
bool CWeightedPathNodeMethod::__isOneMoreOutBoundRange(const std::vector<float>& vTestPoint, const std::vector<float>& vLow, const std::vector<float>& vHeigh, int vOutDimension)
{
	int count = 0;
	for (int i = 0; i < vTestPoint.size(); ++i)
	{
		if (vTestPoint[i] < vLow[i] || vTestPoint[i] > vHeigh[i])
			count++;
		if(count >= vOutDimension)
			return false;
	}
	return true;
}

//****************************************************************************************************
//FUNCTION:
std::vector<int> CWeightedPathNodeMethod::__calInterNodeDataIndex(const CNode* vNode)
{
	if (vNode->isLeafNode())
		return vNode->getNodeDataIndexV();
	const CNode* LeftNode = const_cast<CNode*>(&vNode->getLeftChild());
	const CNode* RightNode = const_cast<CNode*>(&vNode->getRightChild());
	std::vector<int> LeftDataIndex = __calInterNodeDataIndex(LeftNode);
	std::vector<int> RightNodeDataIndex = __calInterNodeDataIndex(RightNode);
	LeftDataIndex.insert(LeftDataIndex.end(), RightNodeDataIndex.begin(), RightNodeDataIndex.end());
	return LeftDataIndex;
}


//****************************************************************************************************
//FUNCTION:
float CWeightedPathNodeMethod::predictWithMinMPOnWholeDimension(const CTree* vTree, const std::vector<float>& vFeatures, std::vector<int>& voNodeDataIndex)
{
	_ASSERTE(vTree);
	std::stack<const CNode*> NodeStack;
	const CNode* TreeRoot = &vTree->getRoot();
	NodeStack.push(TreeRoot);
	std::ofstream Level;
	Level.open("Level.csv", std::ios::app);
	while (!NodeStack.empty())
	{
		const CNode* pCurrentNode = NodeStack.top();
		std::pair<std::vector<float>, std::vector<float>> FeatureRange = pCurrentNode->getFeatureRange();
		int OutDimension = CRegressionForestConfig::getInstance()->getAttribute<int>(KEY_WORDS::OUT_DIMENSION);
		if (OutDimension <= 0) OutDimension = 1;
		if (OutDimension > vFeatures.size()) OutDimension = vFeatures.size();
		bool IsInBox = __isOneMoreOutBoundRange(vFeatures, FeatureRange.first, FeatureRange.second, OutDimension);
		if (IsInBox)
		{
			if (pCurrentNode->isLeafNode())
			{
				Level << pCurrentNode->getLevel() << "," << 1 << std::endl;
				voNodeDataIndex = pCurrentNode->getNodeDataIndexV();
				CTrainingSet *pTraingSet = CTrainingSet::getInstance();
				std::pair<int, float> MinMPAndIndex = pTraingSet->calMinMPAndIndex(voNodeDataIndex, vFeatures);
				return pTraingSet->getResponseValueAt(MinMPAndIndex.first);
			}
			NodeStack.pop();
			if (vFeatures[pCurrentNode->getBestSplitFeatureIndex()] < pCurrentNode->getBestGap())
				NodeStack.push(&pCurrentNode->getLeftChild());
			else
				NodeStack.push(&pCurrentNode->getRightChild());
		}
		else
		{
			Level << pCurrentNode->getLevel() << std::endl;
			voNodeDataIndex = pCurrentNode->isLeafNode() ? pCurrentNode->getNodeDataIndexV() : __calInterNodeDataIndex(pCurrentNode);
			CTrainingSet *pTraingSet = CTrainingSet::getInstance();
			std::pair<int, float> MinMPAndIndex = pTraingSet->calMinMPAndIndex(voNodeDataIndex, vFeatures);
			return pTraingSet->getResponseValueAt(MinMPAndIndex.first);
		}
	}
	Level.close();
}
