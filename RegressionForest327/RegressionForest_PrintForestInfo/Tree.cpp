#include "Tree.h"
#include <cmath>
#include <numeric>
#include <stack>
#include <queue>
#include <boost/format.hpp>
#include <algorithm>
#include <fstream>
#include <unordered_map>
#include "common/HiveCommonMicro.h"
#include "common/ProductFactoryData.h"
#include "math/RandomInterface.h"
#include "TrainingSet.h"
#include "RegressionForestCommon.h"
#include "RegressionForestConfig.h"
#include "BaseInstanceWeightMethod.h"
#include "BaseFeatureWeightMethod.h"
#include "ObjectPool.h"
#include "SingleResponseNode.h"
#include "MultiResponseNode.h"
#include "MpCompute.h"
#include "Utility.h"

using namespace hiveRegressionForest;

#define EPSILON 1.0e-6

CTree::CTree()
{
}

CTree::~CTree()
{
	// NOTES : boost object_pool automatically released
}

//****************************************************************************************************
//FUNCTION:
void CTree::buildTree(IBootstrapSelector* vBootstrapSelector, IFeatureSelector* vFeatureSelector, INodeSpliter* vNodeSpliter, IBaseTerminateCondition* vTerminateCondition, IFeatureWeightGenerator* vFeatureWeightMethod)
{
	//NOTES : 为避免划分过程中存储每个节点的CurrBootstrap，节点划分过程中仅保留一份BootstrapIndex
	//划分后交换Index，将左儿子的都保留在vector左边，右儿子保留在右边，用Range记录左右儿子BootStrap范围
	//Range: first记录第一个元素，second记录最后一个元素的后一位

	std::vector<int> BootstrapIndex;
	__initTreeParameters(vBootstrapSelector, BootstrapIndex);
	_ASSERTE(!m_pRoot);

	m_pRoot = __createNode(1);

	std::pair<std::vector<std::vector<float>>, std::vector<float>> BootstrapDataset;
	std::stack<std::pair<CNode*, std::pair<int, int> >> NodeBootstrapRangeStack;
	NodeBootstrapRangeStack.push({ m_pRoot,{ 0, BootstrapIndex.size() } });

	CTrainingSet::getInstance()->recombineBootstrapDataset(BootstrapIndex, NodeBootstrapRangeStack.top().second, BootstrapDataset);
	m_pRoot->calStatisticsV(BootstrapDataset);
	m_pRoot->setSubEachFeatureSplitRange(m_pRoot->getFeatureRange());
	setBootstrapIndex(BootstrapIndex);
	__sortFeatureResponsePairSet();
	int RangeSplitPos = 0;
	bool IsUpdatingFeaturesWeight = CRegressionForestConfig::getInstance()->getAttribute<bool>(KEY_WORDS::LIVE_UPDATE_FEATURES_WEIGHT);
	std::vector<int> CurrFeatureIndexSubSet;

	while (!NodeBootstrapRangeStack.empty())
	{
		CNode* pCurNode = NodeBootstrapRangeStack.top().first;
		_ASSERTE(pCurNode);

		std::pair<int, int> CurBootstrapRange = NodeBootstrapRangeStack.top().second;
		NodeBootstrapRangeStack.pop();

		CTrainingSet::getInstance()->recombineBootstrapDataset(BootstrapIndex, CurBootstrapRange, BootstrapDataset);//TODO: optimize!!!

		_ASSERTE(vTerminateCondition);
		if (vTerminateCondition->isMeetTerminateConditionV(BootstrapDataset.first, BootstrapDataset.second, __getTerminateConditionExtraParameter(pCurNode)))
		{
			__createLeafNode(pCurNode, BootstrapIndex, CurBootstrapRange, BootstrapDataset);
		}
		else
		{
			CurrFeatureIndexSubSet.clear();

			_selectCandidateFeaturesV(vFeatureSelector, vFeatureWeightMethod, IsUpdatingFeaturesWeight, BootstrapDataset, CurrFeatureIndexSubSet);

			// NOTES : 由于 splitNode 过程如果发现某一边为空，则该节点不再进行划分，即又一种节点划分终止条件, 以
			//         splitNode 函数返回值表示，如false，则不再划分
			if ((vNodeSpliter)->splitNode(pCurNode, CurBootstrapRange, BootstrapIndex, CurrFeatureIndexSubSet, RangeSplitPos))
			{
				pCurNode->setLeftChild(__createNode(pCurNode->getLevel() + 1));
				pCurNode->setRightChild(__createNode(pCurNode->getLevel() + 1));
				NodeBootstrapRangeStack.push(std::make_pair(const_cast<CNode*>(&pCurNode->getLeftChild()), std::pair<int, int>(CurBootstrapRange.first, RangeSplitPos)));
				NodeBootstrapRangeStack.push(std::make_pair(const_cast<CNode*>(&pCurNode->getRightChild()), std::pair<int, int>(RangeSplitPos, CurBootstrapRange.second)));
				CTrainingSet::getInstance()->recombineBootstrapDataset(BootstrapIndex, std::pair<int, int>(CurBootstrapRange.first, RangeSplitPos), BootstrapDataset);
				const_cast<CNode*>(&pCurNode->getLeftChild())->updataFeatureSplitRange(pCurNode->getFeatureSplitRange(), pCurNode->getBestSplitFeatureIndex(), pCurNode->getBestGap(), true);
				const_cast<CNode*>(&pCurNode->getLeftChild())->calStatisticsV(BootstrapDataset);
				CTrainingSet::getInstance()->recombineBootstrapDataset(BootstrapIndex, std::pair<int, int>(RangeSplitPos, CurBootstrapRange.second), BootstrapDataset);
				const_cast<CNode*>(&pCurNode->getRightChild())->calStatisticsV(BootstrapDataset);
				const_cast<CNode*>(&pCurNode->getRightChild())->updataFeatureSplitRange(pCurNode->getFeatureSplitRange(), pCurNode->getBestSplitFeatureIndex(), pCurNode->getBestGap(), false);
			}
			else
			{
				__createLeafNode(pCurNode, BootstrapIndex, CurBootstrapRange, BootstrapDataset);
			}
		}
	}

	__obtainOOBIndex(BootstrapIndex);
}

//****************************************************************************************************
//FUNCTION:
float CTree::predict(const CNode& vCurLeafNode, const std::vector<float>& vFeatures, float& voWeight, unsigned int vResponseIndex /*=0*/) const
{
	_ASSERTE(vCurLeafNode.isLeafNode() && !vFeatures.empty());

	voWeight = vCurLeafNode.calculateNodeWeight(vResponseIndex);

	return vCurLeafNode.predictV(vFeatures, vResponseIndex);
}

//****************************************************************************************************
//FUNCTION:
void CTree::fetchTreeInfo(STreeInfo& voTreeInfo) const
{
	std::vector<const CNode*> AllTreeNodes;
	__dumpAllTreeNodes(AllTreeNodes);

	voTreeInfo.m_FeatureSplitTimes.resize(CTrainingSet::getInstance()->getNumOfFeatures(), 0);
	for (auto TreeNode : AllTreeNodes)
	{
		++voTreeInfo.m_NumOfNodes;
		if (TreeNode->isUnfitted()) ++voTreeInfo.m_NumOfUnfittedLeafNodes;
		
		if (TreeNode->isLeafNode()) ++voTreeInfo.m_NumOfLeafNodes;
		else ++voTreeInfo.m_FeatureSplitTimes[TreeNode->getBestSplitFeatureIndex()];
	}

	const std::vector<int>& OOBIndexSet = this->getOOBIndexSet();
	voTreeInfo.m_InstanceOOBTimes.resize(CTrainingSet::getInstance()->getNumOfInstances(), 0);
	for (auto Itr : OOBIndexSet) voTreeInfo.m_InstanceOOBTimes[Itr]++;
}

//********************************************************************************************************
//FUNCTION:
bool CTree::operator==(const CTree& vTree) const
{
	return m_pRoot->operator==(vTree.getRoot());
}

//****************************************************************************************************
//FUNCTION:
CNode* CTree::__createNode(unsigned int vLevel)
{
	std::string NodeTypeSig;
	if(CRegressionForestConfig::getInstance()->isAttributeExisted(KEY_WORDS::CREATE_NODE_TYPE)) NodeTypeSig = CRegressionForestConfig::getInstance()->getAttribute<std::string>(KEY_WORDS::CREATE_NODE_TYPE);
	else NodeTypeSig = KEY_WORDS::SINGLE_RESPONSE_NODE;
	
	if (NodeTypeSig.empty() || NodeTypeSig == KEY_WORDS::SINGLE_RESPONSE_NODE)
	{
		return CRegressionForestObjectPool<CSingleResponseNode>::getInstance()->allocateNode(vLevel);
	}
	else if (NodeTypeSig == KEY_WORDS::MULTI_RESPONSES_NODE)
	{
		return CRegressionForestObjectPool<CMultiResponseNode>::getInstance()->allocateNode(vLevel);
	}
	else
	{
		_ASSERTE(false);
	}
	
}

//****************************************************************************************************
//FUNCTION:
void CTree::__createLeafNode(CNode* vCurNode, const std::vector<int>& vDataSetIndex, const std::pair<int, int>& vRange, const std::pair<std::vector<std::vector<float>>, std::vector<float>>& vBootstrapDataset)
{
	vCurNode->createAsLeafNodeV(vBootstrapDataset, vDataSetIndex, vRange);
	vCurNode->setNodeSize(vRange.second - vRange.first);
}

//****************************************************************************************************
//FUNCTION:
boost::any CTree::__getTerminateConditionExtraParameter(const CNode* vNode)
{
	return boost::any(vNode->getLevel());
}

//****************************************************************************************************
//FUNCTION:
void CTree::_selectCandidateFeaturesV(IFeatureSelector* vFeatureSelector, IFeatureWeightGenerator* vFeatureWeightMethod, bool vIsUpdatingFeaturesWeight, const std::pair<std::vector<std::vector<float>>, std::vector<float>>& vBootstrapDataset, std::vector<int>& voCandidateFeaturesIndex)
{
	_ASSERTE(!vBootstrapDataset.first.empty());

	static bool IsFeatureWeighted = (CRegressionForestConfig::getInstance()->getAttribute<std::string>(KEY_WORDS::FEATURE_SELECTOR) == (KEY_WORDS::WEIGHTED_FEATURE_SELECTOR));

	static std::vector<float> FeaturesWeightSet;
	if (IsFeatureWeighted)
	{
		__updateFeaturesWeight(vFeatureWeightMethod, vIsUpdatingFeaturesWeight, vBootstrapDataset, FeaturesWeightSet);
	}
	
	vFeatureSelector->generateFeatureIndexSetV(CTrainingSet::getInstance()->getNumOfFeatures(), voCandidateFeaturesIndex, FeaturesWeightSet);
}

//****************************************************************************************************
//FUNCTION:
void CTree::__initTreeParameters(IBootstrapSelector* vBootstrapSelector, std::vector<int>& voBootstrapIndex)
{
	_ASSERTE(vBootstrapSelector);

	std::string BootstrapSelectorSig = CRegressionForestConfig::getInstance()->getAttribute<std::string>(KEY_WORDS::BOOTSTRAP_SELECTOR);
	std::vector<float> InstanceWeightSet;
	
#pragma omp critical
	{
		if (BootstrapSelectorSig == KEY_WORDS::WEIGHTED_BOOTSTRAP_SELECTOR)
		{
			IBaseInstanceWeight* pBaseInstanceWeight = dynamic_cast<IBaseInstanceWeight*>(hiveOO::CProductFactoryData::getInstance()->createProduct(CRegressionForestConfig::getInstance()->getAttribute<std::string>(KEY_WORDS::INSTANCE_WEIGHT_CALCULATE_METHOD)));
			pBaseInstanceWeight->generateInstancesWeightV(CTrainingSet::getInstance()->getNumOfInstances(), InstanceWeightSet);

			_SAFE_DELETE(pBaseInstanceWeight);
		}
	}
	vBootstrapSelector->generateBootstrapIndexSetV(CTrainingSet::getInstance()->getNumOfInstances(), voBootstrapIndex, InstanceWeightSet);
}

//********************************************************************************************************
//FUNCTION:
void CTree::__obtainOOBIndex(std::vector<int>& vBootStrapIndexSet)
{
	_ASSERTE(!vBootStrapIndexSet.empty());

	unsigned int FeatureInstanceSize = vBootStrapIndexSet.size();
	std::sort(vBootStrapIndexSet.begin(), vBootStrapIndexSet.end());

	int Index = 0, i = 0;
	int TrainingSetSize = CTrainingSet::getInstance()->getNumOfInstances();
	for (; i < TrainingSetSize && Index < FeatureInstanceSize; )
	{
		if (i < vBootStrapIndexSet[Index]) m_OOBIndexSet.push_back(i++);
		else if (i == vBootStrapIndexSet[Index]) ++i, ++Index;
		else ++Index;
	}
	for (; i < TrainingSetSize; ++i) m_OOBIndexSet.push_back(i);
}

//****************************************************************************************************
//FUNCTION:
void CTree::__dumpAllTreeNodes(std::vector<const CNode*>& voAllTreeNodes) const
{
	std::queue<const CNode*> q;
	q.push(m_pRoot);

	while (!q.empty())
	{
		voAllTreeNodes.push_back(q.front());
		const CNode* pTempNode = q.front();
		q.pop();
		if (&pTempNode->getLeftChild()) q.push(&pTempNode->getLeftChild());
		if (&pTempNode->getRightChild()) q.push(&pTempNode->getRightChild());
	}
}

//****************************************************************************************************
//FUNCTION:
void CTree::__updateFeaturesWeight(IFeatureWeightGenerator* vFeatureWeightMethod, bool vIsLiveUpdating, const std::pair<std::vector<std::vector<float>>, std::vector<float>>& vBootstrapDataset, std::vector<float>& voFeaturesWeight)
{
	if (!vIsLiveUpdating && vBootstrapDataset.first.size() != CTrainingSet::getInstance()->getNumOfInstances()) return;

	voFeaturesWeight.clear();
	std::vector<std::pair<unsigned int, float>> VariableImportanceSortedList;
	vFeatureWeightMethod->generateFeatureWeight(vBootstrapDataset.first, vBootstrapDataset.second, VariableImportanceSortedList);
	_ASSERTE(!VariableImportanceSortedList.empty());

	voFeaturesWeight.resize(VariableImportanceSortedList.size());
	std::for_each(VariableImportanceSortedList.begin(), VariableImportanceSortedList.end(), [&](const std::pair<unsigned int, float> &vVariableImportance) { voFeaturesWeight[vVariableImportance.first] = vVariableImportance.second; });
	std::cout << voFeaturesWeight.size() << std::endl;
	_ASSERTE(!voFeaturesWeight.empty());
}

//****************************************************************************************************
//FUNCTION:
void CTree::__sortFeatureResponsePairSet()
{
	std::pair<std::vector<std::vector<float>>, std::vector<float>> BootstrapDataset;
	CTrainingSet::getInstance()->recombineBootstrapDataset(m_BootstrapIndex, std::make_pair(0, m_BootstrapIndex.size()), BootstrapDataset);
	std::vector<std::vector<float>>& FeatureSet = BootstrapDataset.first;
	std::vector<float>& ResponseSet = BootstrapDataset.second;
	m_SortedFeatureResponsePairSet.resize(FeatureSet[0].size());
	CMpCompute* pMpCompute = nullptr;
	for (int i = 0; i < FeatureSet[0].size(); ++i)
	{
		pMpCompute->generateSortedFeatureResponsePairSet(FeatureSet, ResponseSet, i, m_SortedFeatureResponsePairSet[i]);
	}
	_SAFE_DELETE(pMpCompute);
}

//****************************************************************************************************
//FUNCTION:
const CNode* CTree::locateLeafNode(const std::vector<float>& vFeatures) const //该数据的15个特征值
{
	_ASSERTE(m_pRoot);

	std::stack<const CNode*> NodeStack; //NOTES : 指针占用内存小
	NodeStack.push(m_pRoot);

	while (!NodeStack.empty())
	{
		const CNode* pCurrNode = NodeStack.top();
		if (pCurrNode->isLeafNode())
		{	
			return pCurrNode;
		}

		NodeStack.pop();
		if (vFeatures[pCurrNode->getBestSplitFeatureIndex()] < pCurrNode->getBestGap())
			NodeStack.push(&pCurrNode->getLeftChild());
		else
			NodeStack.push(&pCurrNode->getRightChild());
	}
}

//******************************************************************************
//FUNCTION:
const CNode* CTree::recordPathNodeInfo(const std::vector<float>& vFeatures, std::vector<SPathNodeInfo>& voPathNodeInfo, std::vector<float>& voOutLeafFeature, std::vector<float>& voOutLeafSplitFeature, int vTreeId) const
{
	_ASSERTE(m_pRoot);
	std::stack<const CNode*> NodeStack; //NOTES : 指针占用内存小
	NodeStack.push(m_pRoot);
	SPathNodeInfo TempPathNode;

	while (!NodeStack.empty())
	{
		const CNode* pCurrNode = NodeStack.top();
		TempPathNode.m_TreeID = vTreeId;
		TempPathNode.m_NodeLevel = pCurrNode->getLevel();
		TempPathNode.m_SplitFeature = pCurrNode->getBestSplitFeatureIndex();
		TempPathNode.m_SplitLocation = pCurrNode->getBestGap();
		TempPathNode.m_FeatureRange = pCurrNode->getFeatureRange();
		TempPathNode.m_FeatureSplitRange = pCurrNode->getFeatureSplitRange();
		voPathNodeInfo.push_back(TempPathNode);
		if (pCurrNode->isLeafNode())
		{
			voOutLeafFeature = pCurrNode->calOutFeatureRange(vFeatures);
			voOutLeafSplitFeature = pCurrNode->calOutFeatureSplitRange(vFeatures);
			return pCurrNode;
		}
		NodeStack.pop();
		if (vFeatures[pCurrNode->getBestSplitFeatureIndex()] < pCurrNode->getBestGap())
			NodeStack.push(&pCurrNode->getLeftChild());
		else
			NodeStack.push(&pCurrNode->getRightChild());
	}
}

//****************************************************************************************************
//FUNCTION:
float CTree::traversalPathPrediction(const std::vector<float>& vFeatures)
{
	_ASSERTE(m_pRoot);
	std::stack<const CNode*> NodeStack;
	NodeStack.push(m_pRoot);
	float PredictResult = 0.f;
	float WeightSum = 0.f;
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
float CTree::calculateOutNodeBound(const std::vector<float>& vFeature, const std::pair<std::vector<float>, std::vector<float>>& vFeatureRange)
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
float CTree::traverWithDistanceFromFeatureRange(const std::vector<float>& vFeatures)
{
	_ASSERTE(m_pRoot);
	std::stack<const CNode*> NodeStack;
	NodeStack.push(m_pRoot);
	float PredictResult = 0.f;
	float WeightSum = 0.f;
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
			//std::cout << "level: " << pCurrentNode->getLevel() << "  NodeMean: " << pCurrentNode->getNodeMeanV() << "  weight: " << DistanceFromFeatureRange << " prediction: " << DistanceFromFeatureRange*pCurrentNode->getNodeMeanV() << std::endl;
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
float CTree::traversePathWithFeatureCentre(const std::vector<float>& vFeatures)
{
	_ASSERTE(m_pRoot);
	std::stack<const CNode*> NodeStack;
	NodeStack.push(m_pRoot);
	float PredictResult = 0.f;
	float WeightSum = 0.f;
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
float CTree::traverWithDistanceFromFeaturesCentre(const std::vector<float>& vFeatures)
{
	_ASSERTE(m_pRoot);
	std::stack<const CNode*> NodeStack;
	NodeStack.push(m_pRoot);
	float PredictResult = 0.f;
	float WeightSum = 0.f;
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
float CTree::travelWithMonteCarlo(const std::vector<float>& vFeatures)
{
	_ASSERTE(m_pRoot);
	std::stack<const CNode*> NodeStack;
	NodeStack.push(m_pRoot);
	float PredictResult = 0.f;
	float WeightSum = 0.f;
	while (!NodeStack.empty())
	{
		const CNode* pCurrentNode = NodeStack.top();
		std::pair<std::vector<float>, std::vector<float>> FeatureRange = pCurrentNode->getFeatureRange();

		float DistanceFromFeatureRange = 0.0f;
		for (int i = 0; i < vFeatures.size(); i++)
		{
			if (FeatureRange.first[i] - vFeatures[i] > EPSILON)
			{
				DistanceFromFeatureRange += computeCDF(vFeatures[i], FeatureRange.first[i]);
			}
			if (vFeatures[i] - FeatureRange.second[i] > EPSILON)
			{
				DistanceFromFeatureRange += computeCDF(FeatureRange.second[i], vFeatures[i]);
			}
		}

		if (fabs(DistanceFromFeatureRange) > 1e-6)
		{
			WeightSum += 1/DistanceFromFeatureRange;
			PredictResult += 1/DistanceFromFeatureRange*pCurrentNode->getNodeMeanV();
			//std::cout << "level: " << pCurrentNode->getLevel() << "  NodeMean: " << pCurrentNode->getNodeMeanV() << "  weight: " << 1 / DistanceFromFeatureRange << " prediction: " << 1 / DistanceFromFeatureRange*pCurrentNode->getNodeMeanV() << std::endl;
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
float CTree::computeCDF(float vFirst, float vSecond)
{
	return 0.5*(erfc(-vSecond*sqrt(0.5)) - erfc(-vFirst*sqrt(0.5)));
}

//****************************************************************************************************
//FUNCTION:
void CTree::printYRangeWithLeafXRange(const std::vector<float>& vFeatures, const std::string& vFilePath, const CNode* vNode) const
{
	std::ofstream PrintFile(vFilePath, std::ios::app);
	if (PrintFile.is_open())
	{
		const std::pair<std::vector<float>, std::vector<float>>& FeatureRange = vNode->getFeatureRange();
		const std::pair<std::vector<float>, std::vector<float>>& SplitFeatureRange = vNode->getFeatureSplitRange();
		std::pair<std::vector<float>, std::vector<float>> OutRange;
		std::vector<float> ResponseVar, SplitRangeVar, OutSplitRangeVar;
		std::vector<std::pair<float, float>> ResponseRange, SplitResponseRange, OutResponseRange;
		int FeatureNum = vFeatures.size();
		for (int i = 0; i < FeatureNum; i++)
		{
			if (vFeatures[i] < FeatureRange.first[i])
			{
				OutRange.first.push_back(vFeatures[i]);
				OutRange.second.push_back(FeatureRange.second[i]);
			}
			else if (vFeatures[i] > FeatureRange.second[i])
			{
				OutRange.first.push_back(FeatureRange.first[i]);
				OutRange.second.push_back(vFeatures[i]);
			}
			else
			{
				OutRange.first.push_back(vFeatures[i]);
				OutRange.second.push_back(vFeatures[i]);
			}
		}
		std::vector<int> OutRangeNum = calFeatureRangeResponse(OutRange, OutResponseRange, OutSplitRangeVar);
		std::vector<int> FeatureRangeNum = calFeatureRangeResponse(FeatureRange, ResponseRange, ResponseVar);
		std::vector<int> SplitFeatureRangeNum = calFeatureRangeResponse(SplitFeatureRange, SplitResponseRange, SplitRangeVar);
		PrintFile << "Y-Feature-Range" << ",";
		for (int i = 0; i < FeatureNum; i++)
			PrintFile << ResponseRange[i].first << "~" << ResponseRange[i].second << ",";
		PrintFile << std::endl;
		PrintFile << "Y-Feature-Var" << ",";
		for (int i = 0; i < FeatureNum; i++)
			PrintFile << ResponseVar[i] << ",";
		PrintFile << std::endl;
		PrintFile << "Y-Split-Range" << ",";
		for (int i = 0; i < FeatureNum; i++)
			PrintFile << SplitResponseRange[i].first << "~" << SplitResponseRange[i].second << ",";
		PrintFile << std::endl;
		PrintFile << "Y-Split-Var" << ",";
		for (int i = 0; i < FeatureNum; i++)
			PrintFile << SplitRangeVar[i] << ",";
		PrintFile << std::endl;
		PrintFile << "Y-Out-Range" << ",";
		for (int i = 0; i < FeatureNum; i++)
			PrintFile << OutResponseRange[i].first << "~" << OutResponseRange[i].second << ",";
		PrintFile << std::endl;
		PrintFile << "Y-Out-Var" << ",";
		for (int i = 0; i < FeatureNum; i++)
			PrintFile << OutSplitRangeVar[i] << ",";
		PrintFile << std::endl;
		/*PrintFile << "Y-Num-Feature-Range:" << ",";
		for (int i = 0; i < FeatureNum; i++)
			PrintFile << FeatureRangeNum[i] << ",";
		PrintFile << std::endl;*/
		/*PrintFile << "Y-Num-Split-Range:" << ",";
		for (int i = 0; i < FeatureNum; i++)
			PrintFile << SplitFeatureRangeNum[i] << ",";
		PrintFile << std::endl;
		PrintFile << "Y-Num-Out-Range:" << ",";
		for (int i = 0; i < FeatureNum; i++)
			PrintFile << OutRangeNum[i] << ",";
		PrintFile << std::endl;*/
	}
}

//****************************************************************************************************
//FUNCTION:
void CTree::printResponseInfoInAABB(const std::vector<float>& vFeatures, const std::string & vFilePath, const CNode * vNode) const
{
	std::ofstream PrintFile(vFilePath, std::ios::app);
	if (PrintFile.is_open())
	{
		std::vector<int> TreeBootstrapIndex = getBootstrapIndex();
		std::pair<std::vector<std::vector<float>>, std::vector<float>> TreeBootstrapDataset;
		CTrainingSet::getInstance()->recombineBootstrapDataset(TreeBootstrapIndex, std::make_pair(0, TreeBootstrapIndex.size()), TreeBootstrapDataset);
		std::pair<std::vector<float>, std::vector<float>> FeatureRange;
		const std::vector<std::vector<float>>& Dataset = vNode->getBootstrapDataset().first;
		for (int i = 0; i < vFeatures.size(); i++)
		{
			FeatureRange.first.push_back(std::min(vFeatures[i], Dataset[0][i]));
			FeatureRange.second.push_back(std::max(vFeatures[i], Dataset[0][i]));
		}
		std::vector<int>& ResponseIndex = calDupIndex(calFeatureRangeIndex(FeatureRange, TreeBootstrapDataset.first));
		if (ResponseIndex.size() == 0)
			PrintFile << "NO DUP DATA" << std::endl;
		else
		{
			std::vector<float> ResponseSet;
			for (int i = 0; i < ResponseIndex.size(); i++)
			{
				ResponseSet.push_back(TreeBootstrapDataset.second[ResponseIndex[i]]);
				for (int k = 0; k < vFeatures.size(); k++)
					std::cout << TreeBootstrapDataset.first[ResponseIndex[i]][k] << ", ";
				std::cout << std::endl;
			}
			float ResponseVar = calSumSquareError(ResponseSet) / ResponseSet.size();
			std::cout << ResponseVar << std::endl;
			PrintFile << "Num:" << "," << ResponseSet.size() << "," << "Var:" << "," << ResponseVar << std::endl;
		}
	}
}

//******************************************************************************
//FUNCTION:
std::vector<int> CTree::calFeatureRangeResponse(const std::pair<std::vector<float>, std::vector<float>>& vLeafNodeFeatureRange, std::vector<std::pair<float, float>>& voResponseRange, std::vector<float>& voResponseVariance) const
{
	std::vector<float> InterResponse;
	std::vector<int> InterResponseNum;

	for (int i = 0; i < vLeafNodeFeatureRange.first.size(); i++)
	{
		float MinX = vLeafNodeFeatureRange.first[i], MaxX = vLeafNodeFeatureRange.second[i];
		InterResponse = calSecondParRange(MinX, MaxX, m_SortedFeatureResponsePairSet[i]);
		float MinResponse = *std::min_element(InterResponse.begin(), InterResponse.end());
		float MaxResponse = *std::max_element(InterResponse.begin(), InterResponse.end());
		voResponseRange.push_back(std::make_pair(MinResponse, MaxResponse));
		float ResponseVar = calSumSquareError(InterResponse);
		voResponseVariance.push_back(ResponseVar / InterResponse.size());
		InterResponseNum.push_back(InterResponse.size());

		InterResponse.clear();
	}
	return InterResponseNum;
}

//****************************************************************************************************
//FUNCTION:
std::vector<std::vector<int>> CTree::calFeatureRangeIndex(const std::pair<std::vector<float>, std::vector<float>>& vLeafNodeFeatureRange, const std::vector<std::vector<float>>& vFeatureSet) const
{
	int FeatureNum = vLeafNodeFeatureRange.first.size();
	std::vector<std::pair<float, int>> TempFeatureIndex;
	std::vector<std::vector<int>> ResponseIndex(FeatureNum);

	for (int i = 0; i < FeatureNum; i++)
	{
		for (int k = 0; k < vFeatureSet.size(); k++)
		{
			TempFeatureIndex.push_back(std::make_pair(vFeatureSet[k][i], k));
		}
		sort(TempFeatureIndex.begin(), TempFeatureIndex.end(), [](std::pair<float, int> &vFirst, std::pair<float, int>& vSecond) {return vFirst.first < vSecond.first; });
		float MinX = vLeafNodeFeatureRange.first[i], MaxX = vLeafNodeFeatureRange.second[i];
		ResponseIndex[i] = calSecondParRange(MinX, MaxX, TempFeatureIndex);
		TempFeatureIndex.clear();
	}
	return ResponseIndex;
}

//****************************************************************************************************
//FUNCTION:
std::vector<int> CTree::calDupIndex(const std::vector<std::vector<int>>& vResponseIndex) const
{
	std::vector<int> Index;
	std::unordered_map<int, int> DupIndex;
	for (int i = 0; i < vResponseIndex[0].size(); i++)
		DupIndex[vResponseIndex[0][i]] = 1;
	for (int i = 1; i < vResponseIndex.size(); i++)
	{
		for (int k = 0; k < vResponseIndex[i].size(); k++)
		{
			if (DupIndex.find(vResponseIndex[i][k]) != DupIndex.end())
				DupIndex[vResponseIndex[i][k]]++;
		}
	}
	for (auto it = DupIndex.begin(); it != DupIndex.end(); it++)
		if ((*it).second == vResponseIndex.size())
			Index.push_back((*it).first);
	return Index;
}
