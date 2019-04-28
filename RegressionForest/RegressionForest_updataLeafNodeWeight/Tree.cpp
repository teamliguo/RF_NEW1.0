#include "Tree.h"
#include <stack>
#include <algorithm>
#include <fstream>
#include "common/HiveCommonMicro.h"
#include "common/ProductFactoryData.h"
#include "BaseInstanceWeightMethod.h"
#include "BaseFeatureWeightMethod.h"
#include "ObjectPool.h"
#include "SingleResponseNode.h"
#include "MultiResponseNode.h"

using namespace hiveRegressionForest;

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
				CNode* pLeftNode = __createNode(pCurNode->getLevel() + 1);
				CNode* pRightNode = __createNode(pCurNode->getLevel() + 1);				
				NodeBootstrapRangeStack.push(std::make_pair(pLeftNode, std::pair<int, int>(CurBootstrapRange.first, RangeSplitPos)));
				NodeBootstrapRangeStack.push(std::make_pair(pRightNode, std::pair<int, int>(RangeSplitPos, CurBootstrapRange.second)));
				CTrainingSet::getInstance()->recombineBootstrapDataset(BootstrapIndex, std::pair<int, int>(CurBootstrapRange.first, RangeSplitPos), BootstrapDataset);
				pLeftNode->calStatisticsV(BootstrapDataset);
				CTrainingSet::getInstance()->recombineBootstrapDataset(BootstrapIndex, std::pair<int, int>(RangeSplitPos, CurBootstrapRange.second), BootstrapDataset);
				pRightNode->calStatisticsV(BootstrapDataset);
				pCurNode->setLeftChild(pLeftNode);
				pCurNode->setRightChild(pRightNode);
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
void CTree::__dumpAllLeafNodes(std::vector<const CNode*>& voAllLeafNodes) const
{
	std::queue<const CNode*> q;
	q.push(m_pRoot);

	while (!q.empty())
	{
		const CNode* pTempNode = q.front();
		if (pTempNode->isLeafNode())
			voAllLeafNodes.push_back(q.front());
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
	for (int i = 0; i < FeatureSet[0].size(); ++i)
	{
		__generateSortedFeatureResponsePairSet(FeatureSet, ResponseSet, i, m_SortedFeatureResponsePairSet[i]);
	}
}

//****************************************************************************************************
//FUNCTION:
void CTree::__generateSortedFeatureResponsePairSet(const std::vector<std::vector<float>>& vFeatureSet, const std::vector<float>& vResponseSet, unsigned int vFeatureIndex, std::vector<std::pair<float, float>>& voSortedFeatureResponseSet)
{
	_ASSERTE(!vFeatureSet.empty() && !vResponseSet.empty());

	voSortedFeatureResponseSet.resize(vFeatureSet.size());

	const CTrainingSet *pTrainingSet = CTrainingSet::getInstance();
	for (int i = 0; i < vFeatureSet.size(); ++i)
	{
		voSortedFeatureResponseSet[i] = std::make_pair(vFeatureSet[i][vFeatureIndex], vResponseSet[i]);
	}

	std::sort(voSortedFeatureResponseSet.begin(), voSortedFeatureResponseSet.end(), [](const std::pair<float, float>& P1, const std::pair<float, float>& P2) {return P1.first < P2.first; });
}

//****************************************************************************************************
//FUNCTION:
const CNode* CTree::locateLeafNode(const std::vector<float>& vFeatures) const
{
	_ASSERTE(m_pRoot);

	std::stack<const CNode*> NodeStack;
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

//****************************************************************************************************
//FUNCTION:
void CTree::updateUnusedNodeWeight()
{
	std::vector<const CNode*> LeafNodeSet;
	__dumpAllLeafNodes(LeafNodeSet);
	float SumUsedNodeWeight = 0.f;
	int UsedNodeCount = 0;
	for (int i = 0; i < LeafNodeSet.size(); ++i)
	{
		if (LeafNodeSet[i]->isUsed())
		{
			SumUsedNodeWeight += LeafNodeSet[i]->getNodeWeight();
			UsedNodeCount++;
		}
	}
	_ASSERTE(UsedNodeCount > 0);
	float MeamWeight = SumUsedNodeWeight / UsedNodeCount;
	for (int i = 0; i < LeafNodeSet.size(); ++i)
	{
		if (!LeafNodeSet[i]->isUsed())
		{
			const_cast<CNode*>(LeafNodeSet[i])->setLeafNodeWeight(MeamWeight);
		}
	}
}
