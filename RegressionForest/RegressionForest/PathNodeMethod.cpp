#include "PathNodeMethod.h"
#include <vector>
#include <stack>
#include "RegressionForestCommon.h"

using namespace hiveRegressionForest;

//****************************************************************************************************
//FUNCTION:
float CPathNodeMethod::calOutNodeBound(const std::vector<float>& vFeature, const std::pair<std::vector<float>, std::vector<float>>& vFeatureRange)
{
	_ASSERTE(!vFeature.empty() && !vFeatureRange.first.empty());

	const std::vector<float>& Lower = vFeatureRange.first;
	const std::vector<float>& Upper = vFeatureRange.second;
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
	_ASSERTE(!vFeature.empty() && !vFeatureRange.first.empty());

	const std::vector<float>& Lower = vFeatureRange.first;
	const std::vector<float>& Upper = vFeatureRange.second;
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
	_ASSERTE(!vFeature.empty());

	std::stack<const CNode*> NodeStack;
	const CNode* TreeRoot = &vTree->getRoot();
	NodeStack.push(TreeRoot);
	float PredictResult = 0.f, WeightSum = 0.f;
	while (!NodeStack.empty())
	{
		const CNode* pCurrentNode = NodeStack.top();
		int SplitFeature = pCurrentNode->getBestSplitFeatureIndex();
		const std::pair<std::vector<float>, std::vector<float>>& FeatureRange = pCurrentNode->getFeatureRange();

		if (FeatureRange.first[SplitFeature] - vFeature[SplitFeature] > FLT_EPSILON)
		{
			float Len = (FeatureRange.second[SplitFeature] - FeatureRange.first[SplitFeature] + FLT_EPSILON) / (FeatureRange.first[SplitFeature] - vFeature[SplitFeature]);
			WeightSum += Len * pCurrentNode->getLevel();
			PredictResult += (pCurrentNode->getNodeMeanV()) * (Len*(pCurrentNode->getLevel()));
			if (pCurrentNode->isLeafNode())
				return PredictResult / WeightSum;
		}
		if (vFeature[SplitFeature] - FeatureRange.second[SplitFeature] > FLT_EPSILON)
		{
			float Len = (FeatureRange.second[SplitFeature] - FeatureRange.first[SplitFeature] + FLT_EPSILON) / (vFeature[SplitFeature] - FeatureRange.second[SplitFeature]);
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

	return PredictResult;
}

//****************************************************************************************************
//FUNCTION:
float CPathNodeMethod::traverWithDistanceFromFeatureRange(const CTree* vTree, const std::vector<float>& vFeature)
{
	_ASSERTE(!vFeature.empty());

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
			if (FeatureRange.first[i] - vFeature[i] > FLT_EPSILON)
			{
				DistanceFromFeatureRange *= (FeatureRange.second[i] - FeatureRange.first[i] + FLT_EPSILON) / (FeatureRange.first[i] - vFeature[i]);
			}
			if (vFeature[i] - FeatureRange.second[i] > FLT_EPSILON)
			{
				DistanceFromFeatureRange *= (FeatureRange.second[i] - FeatureRange.first[i] + FLT_EPSILON) / (vFeature[i] - FeatureRange.second[i]);
			}
		}

		if (DistanceFromFeatureRange != 1.0f && fabs(DistanceFromFeatureRange) > 1e-6)
		{
			WeightSum += DistanceFromFeatureRange;
			PredictResult += DistanceFromFeatureRange*pCurrentNode->getNodeMeanV();
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

	return PredictResult;
}

//****************************************************************************************************
//FUNCTION:
float CPathNodeMethod::traversePathWithFeatureCentre(const CTree* vTree, const std::vector<float>& vFeature)
{
	_ASSERTE(!vFeature.empty());

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

	return PredictResult;
}


//****************************************************************************************************
//FUNCTION:
float CPathNodeMethod::traverWithDistanceFromFeaturesCentre(const CTree* vTree, const std::vector<float>& vFeature)
{
	_ASSERTE(!vFeature.empty());

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

		if (DistanceFromFeatures != 1.0f)
		{
			WeightSum += DistanceFromFeatures;
			PredictResult += DistanceFromFeatures*pCurrentNode->getNodeMeanV();
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

	return PredictResult;
}

//****************************************************************************************************
//FUNCTION:
float CPathNodeMethod::predictWithMonteCarlo(const CNode& vCurLeafNode, const std::vector<float>& vFeature)
{
	_ASSERTE(!vFeature.empty());

	CTrainingSet* pTrainingSet = CTrainingSet::getInstance();
	std::vector<int> DataIndex = (const_cast<CNode*>(&vCurLeafNode))->getNodeDataIndex();
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
	_ASSERTE(!vDataSetIndexSet.empty());

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
	_ASSERTE(!vTestPoint.empty() && !vLow.empty() && !vHeigh.empty());

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
	_ASSERTE(!vTestPoint.empty() && !vLow.empty() && !vHeigh.empty());

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
		return vNode->getNodeDataIndex();
	const CNode* LeftNode = const_cast<CNode*>(&vNode->getLeftChild());
	const CNode* RightNode = const_cast<CNode*>(&vNode->getRightChild());
	std::vector<int> LeftDataIndex = __calInternalNodeDataIndex(LeftNode);
	std::vector<int> RightNodeDataIndex = __calInternalNodeDataIndex(RightNode);
	LeftDataIndex.insert(LeftDataIndex.end(), RightNodeDataIndex.begin(), RightNodeDataIndex.end());

	return LeftDataIndex;
}

//****************************************************************************************************
//FUNCTION:
float CPathNodeMethod::prediceWithInternalNode(const CTree* vTree, const std::vector<float>& vFeature)
{
	_ASSERTE(!vFeature.empty());

	std::stack<const CNode*> NodeStack;
	const CNode* TreeRoot = &vTree->getRoot();
	NodeStack.push(TreeRoot);
	int OutDimension = CRegressionForestConfig::getInstance()->getAttribute<int>(KEY_WORDS::OUT_DIMENSION);
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

	return 0.f;
}