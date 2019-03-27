#include "RegressionForest.h"
#include <omp.h>
#include <numeric>
#include <fstream>
#include <algorithm>
#include "common/CommonInterface.h"
#include "common/ConfigParser.h"
#include "common/HiveCommonMicro.h"
#include "common/ProductFactoryData.h"
#include "common/productfactory.h"
#include "Tree.h"
#include "TrainingSet.h"
#include "BaseInstanceWeightMethod.h"
#include "RegressionForestConfig.h"
#include "RegressionForestCommon.h"

using namespace hiveRegressionForest;

CRegressionForest::CRegressionForest()
{
}

CRegressionForest::~CRegressionForest()
{
	__clearForest();
}

//****************************************************************************************************
//FUNCTION:
void CRegressionForest::buildForest(const std::string& vConfigFile)
{
	_ASSERTE(!vConfigFile.empty());

	const CRegressionForestConfig* pRegressionForestConfig = CRegressionForestConfig::getInstance();
	if (!CRegressionForestConfig::isConfigParsed())
	{
		bool IsConfigParsed = hiveConfig::hiveParseConfig(vConfigFile, hiveConfig::EConfigType::XML, CRegressionForestConfig::getInstance());
		_ASSERTE(IsConfigParsed);
	}

	clock_t Begin = clock();

	__initForest();

	IBootstrapSelector* pBootstrapSelector = nullptr;
	IFeatureSelector* pFeatureSelector = nullptr;
	INodeSpliter* pNodeSpliter = nullptr;
	IBaseTerminateCondition* pTerminateCondition = nullptr;
	IFeatureWeightGenerator* pFeatureWeightMethod = nullptr;
	__initForestParameters(pBootstrapSelector, pFeatureSelector, pNodeSpliter, pTerminateCondition, pFeatureWeightMethod);

	bool OmpParallelSig = pRegressionForestConfig->getAttribute<bool>(KEY_WORDS::OPENMP_PARALLEL_BUILD_TREE);

#pragma omp parallel for if (OmpParallelSig)
	for (auto i = 0; i < m_Trees.size(); ++i)
	{
		m_Trees[i]->buildTree(pBootstrapSelector, pFeatureSelector, pNodeSpliter, pTerminateCondition, pFeatureWeightMethod);
		std::cout << "Successfully built the " << i << " th tree." << std::endl;
	}
	clock_t End = clock();
		
	_SAFE_DELETE(pBootstrapSelector);
	_SAFE_DELETE(pFeatureSelector);
	_SAFE_DELETE(pNodeSpliter);
	_SAFE_DELETE(pTerminateCondition);
	_SAFE_DELETE(pFeatureWeightMethod);

	hiveCommon::hiveOutputEvent("Successfully built regression forests in " + std::to_string(End - Begin) + " milliseconds.");
	_LOG_("Successfully built regression forests in " + std::to_string(End - Begin) + " milliseconds.");
}

//****************************************************************************************************
//FUNCTION:
void CRegressionForest::outputForestInfo(const std::string& vOutputFileName) const
{
	int NumOfNodes = 0, NumOfLeafNodes = 0, NumOfUnfittedLeafNode = 0;
	std::vector<int> FeatureSplitTimes(CTrainingSet::getInstance()->getNumOfFeatures(), 0);
	std::vector<int> InstancesOOBTimes(CTrainingSet::getInstance()->getNumOfInstances(), 0);
	std::vector<int> OOBSize(m_Trees.size(), 0);
		
	for (auto i = 0; i < m_Trees.size(); ++i)
	{
		STreeInfo TreeInfo;
		m_Trees[i]->fetchTreeInfo(TreeInfo);

		NumOfNodes				+= TreeInfo.m_NumOfNodes;
		NumOfLeafNodes			+= TreeInfo.m_NumOfLeafNodes;
		NumOfUnfittedLeafNode	+= TreeInfo.m_NumOfUnfittedLeafNodes;

		for (int m = 0; m < TreeInfo.m_FeatureSplitTimes.size(); ++m)
			FeatureSplitTimes[m] += TreeInfo.m_FeatureSplitTimes[m];

		for (int k = 0; k < TreeInfo.m_InstanceOOBTimes.size(); ++k)
			InstancesOOBTimes[k] += TreeInfo.m_InstanceOOBTimes[k];

		OOBSize[i] = m_Trees[i]->getOOBIndexSet().size();
	}

	std::fstream ForestInfo(vOutputFileName, std::ios::out);
	if (ForestInfo.is_open())
	{
		ForestInfo << "Number of Nodes : " << "," << NumOfNodes << "\n";
		ForestInfo << "Number of Leaf Nodes : " << "," << NumOfLeafNodes << "\n";
		ForestInfo << "Number of Unfitted Leaf Nodes : " << "," << NumOfUnfittedLeafNode << "\n";

		ForestInfo << "Feature Split Times:\n";
		for (int i = 0; i < FeatureSplitTimes.size(); ++i)
		{
			ForestInfo << FeatureSplitTimes[i] << ",";
			if (i % 100 == 99  && i != 0) ForestInfo << "\n";			
		}
		ForestInfo << "\n";

		ForestInfo << "OOB Size in Each Tree : \n";
		for (auto Itr : OOBSize) ForestInfo << Itr << ",";
		ForestInfo << "\n";

		ForestInfo << "Instance OOB Times : \n";
		for (int k = 0; k < InstancesOOBTimes.size(); ++k)
		{
			ForestInfo << InstancesOOBTimes[k] << ",";
			if (k % 100 == 99 && k != 0) ForestInfo << "\n";			
		}
		ForestInfo << "\n";
	}
	else
	{
		std::cout << "Forest Information File failed to open..." << std::endl;
	}
	ForestInfo.close();
}

//****************************************************************************************************
//FUNCTION:
float CRegressionForest::predict(const std::vector<float>& vFeatures, unsigned int vNumOfUsingTrees, float& voLPPredictSet, float& voMPPredictSet, float& voMPDissimilarity, bool vIsWeightedPrediction) const
{ 
	return __predictCertainResponse(vFeatures, vNumOfUsingTrees, vIsWeightedPrediction, voLPPredictSet, voMPPredictSet, voMPDissimilarity);
}

//****************************************************************************************************
//FUNCTION:
void CRegressionForest::predict(const std::vector<float>& vFeatures, unsigned int vNumOfUsingTrees, bool vIsWeightedPrediction, unsigned int vNumResponse, std::vector<float>& voPredictValue) const
{
	_ASSERTE(!vFeatures.empty());

	voPredictValue.resize(vNumResponse); 
	/*for (int i = 0; i < vNumResponse; ++i)
		voPredictValue[i] = __predictCertainResponse(vFeatures, vNumOfUsingTrees, vIsWeightedPrediction, 0, 0, i);*/
}

//****************************************************************************************************
//FUNCTION:
float CRegressionForest::__predictCertainResponse(const std::vector<float>& vFeatures, unsigned int vNumOfUsingTrees, bool vIsWeightedPrediction, float& voLPPredictSet, float& voMPPredictSet, float& voMPDissimilarity, unsigned int vResponseIndex) const
{
	_ASSERTE(!vFeatures.empty() && vNumOfUsingTrees > 0);

	float PredictValue = 0.0f;
	std::vector<float> PredictValueOfTree(vNumOfUsingTrees, 0.0f);
	std::vector<float> NodeWeight(vNumOfUsingTrees, 0.0f);
	std::vector<float> MPDistance(vNumOfUsingTrees, 0.0f);
	std::vector<float> EuclideanDistance(vNumOfUsingTrees, 0.0f);

	CTrainingSet* pTrainingSet = CTrainingSet::getInstance();

	static std::vector<const CNode*> LeafNodeSet;
	LeafNodeSet.resize(this->getNumOfTrees());
#pragma omp parallel for
	for (int i = 0; i < vNumOfUsingTrees; ++i)
	{
		//std::cout << "tree " << i << ": " << std::endl;
		if (vResponseIndex == 0) LeafNodeSet[i] = m_Trees[i]->locateLeafNode(vFeatures);
		PredictValueOfTree[i] = m_Trees[i]->predict(*LeafNodeSet[i], vFeatures, NodeWeight[i], vResponseIndex);
		float Dissimilarity = pTrainingSet->calMPDissimilarityGlobal(LeafNodeSet[i]->getNodeDataIndexV(), vFeatures, PredictValueOfTree[i]);
		MPDistance[i] = 1.0 / (Dissimilarity + 1e-6);
		EuclideanDistance[i] = 1.0 / (pTrainingSet->calEuclideanDistance(LeafNodeSet[i]->getNodeDataIndexV(), vFeatures) + 1e-6);
		voMPDissimilarity += Dissimilarity + 1e-6;
	}
	voMPDissimilarity /= vNumOfUsingTrees;
	//std::cout << std::endl;.

	_ASSERTE(PredictValueOfTree.size() == NodeWeight.size());

	if (vIsWeightedPrediction)
	{
		for (int k = 0; k < vNumOfUsingTrees; ++k)
			PredictValue += PredictValueOfTree[k] * NodeWeight[k];
		float SumWeight = std::accumulate(NodeWeight.begin(), NodeWeight.end(), 0.0f);
		_ASSERTE(SumWeight > 0);

		return PredictValue / SumWeight;
	}
	else
	{
		for (int k = 0; k < vNumOfUsingTrees; ++k)
		{
			voLPPredictSet += PredictValueOfTree[k] * EuclideanDistance[k];
			voMPPredictSet += PredictValueOfTree[k] * MPDistance[k];
		}
		float SumLPWeight = std::accumulate(EuclideanDistance.begin(), EuclideanDistance.end(), 0.0f);
		float SumMPWeight = std::accumulate(MPDistance.begin(), MPDistance.end(), 0.0f);
		voLPPredictSet = voLPPredictSet / SumLPWeight;
		voMPPredictSet = voMPPredictSet / SumMPWeight;

		PredictValue = std::accumulate(PredictValueOfTree.begin(), PredictValueOfTree.end(), 0.0f);
		return PredictValue / vNumOfUsingTrees;
	}
}

//********************************************************************************************************
//FUNCTION:
bool CRegressionForest::operator==(const CRegressionForest& vRegressionForest) const
{
	// NOTES : 这里没有比较 OOB Error，两个原因：
	//         1、由于如果森林模型中树都一致，那么 OOB Error也会一致

	if (m_Trees.size() != vRegressionForest.getNumOfTrees()) return false;

	for (auto Index = 0; Index < m_Trees.size(); ++Index)
	{
		if (!(m_Trees[Index]->operator==(*(vRegressionForest.getTreeAt(Index))))) return false;
	}

	return true;
}

//****************************************************************************************************
//FUNCTION:
void CRegressionForest::__initForest()
{
	int NumOfTrees = CRegressionForestConfig::getInstance()->getAttribute<int>(KEY_WORDS::NUMBER_OF_TREE);
	m_Trees.resize(NumOfTrees);

	for (unsigned int i = 0; i < m_Trees.size(); ++i)
	{
		// NOTES : 判断建树的方式（2-stage等）
		if (!CRegressionForestConfig::getInstance()->isAttributeExisted(KEY_WORDS::BUILD_TREE_TYPE))
			m_Trees[i] = new CTree();
		else
			m_Trees[i] = dynamic_cast<CTree*>(hiveOO::CProductFactoryData::getInstance()->createProduct(CRegressionForestConfig::getInstance()->getAttribute<std::string>(KEY_WORDS::BUILD_TREE_TYPE)));
	}
}

//****************************************************************************************************
//FUNCTION:
void CRegressionForest::__initForestParameters(IBootstrapSelector*& voBootstrapSelector, IFeatureSelector*& voFeatureSelector, INodeSpliter*& voNodeSpliter, IBaseTerminateCondition*& voTerminateCondition, IFeatureWeightGenerator*& voFeatureWeightMethod)
{
	const CRegressionForestConfig *pRegressionForestConfig = CRegressionForestConfig::getInstance();
	
	std::string BootstrapSelectorSig = pRegressionForestConfig->getAttribute<std::string>(KEY_WORDS::BOOTSTRAP_SELECTOR);
	voBootstrapSelector = dynamic_cast<IBootstrapSelector*>(hiveOO::CProductFactoryData::getInstance()->createProduct(BootstrapSelectorSig));
	
	std::string FeatureSelectorSig = pRegressionForestConfig->isAttributeExisted(KEY_WORDS::FEATURE_SELECTOR) ? pRegressionForestConfig->getAttribute<std::string>(KEY_WORDS::FEATURE_SELECTOR) : KEY_WORDS::UNIFORM_FEATURE_SELECTOR;
	voFeatureSelector = dynamic_cast<IFeatureSelector*>(hiveOO::CProductFactoryData::getInstance()->createProduct(FeatureSelectorSig));
	
	std::string SpliterSig = pRegressionForestConfig->getAttribute<std::string>(KEY_WORDS::NODE_SPLIT_METHOD);
	voNodeSpliter = dynamic_cast<INodeSpliter*>(hiveOO::CProductFactoryData::getInstance()->createProduct(SpliterSig));
																										
	std::string TerminateConditionSig = pRegressionForestConfig->getAttribute<std::string>(KEY_WORDS::LEAF_NODE_CONDITION);
	voTerminateCondition = dynamic_cast<IBaseTerminateCondition*>(hiveOO::CProductFactoryData::getInstance()->createProduct(TerminateConditionSig));

	std::string FeatureWeightSig = pRegressionForestConfig->getAttribute<std::string>(KEY_WORDS::FEATURE_WEIGHT_CALCULATE_METHOD);
	voFeatureWeightMethod = dynamic_cast<IFeatureWeightGenerator*>(hiveOO::CProductFactoryData::getInstance()->createProduct(FeatureWeightSig));
}

//****************************************************************************************************
//FUNCTION:
void CRegressionForest::__clearForest()
{
	for (unsigned int i = 0; i < m_Trees.size(); ++i)
		if (m_Trees[i]) _SAFE_DELETE(m_Trees[i]);
}

void CRegressionForest::outputOOBInfo(const std::string& vOutputFileName) const
{
	std::fstream OutPutFile(vOutputFileName, std::ios::out);
	if (OutPutFile.is_open())
	{
		OutPutFile << "Node Number" << "," << "OOB Number" <<std::endl;
		for (auto Itr:m_Trees)
		{
			STreeInfo TreeInfo;
			Itr->fetchTreeInfo(TreeInfo);
			OutPutFile << TreeInfo.m_NumOfLeafNodes << "," << Itr->getOOBIndexSet().size() << std::endl;
		}
	}
}
