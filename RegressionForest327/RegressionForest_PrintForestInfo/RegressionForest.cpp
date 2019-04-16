#include "RegressionForest.h"
#include <omp.h>
#include <numeric>
#include <fstream>
#include <algorithm>
#include <map>
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
#include "PathNodeMethod.h"
#include "BasePredictionMethod.h"
#include "MpCompute.h"

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
void CRegressionForest::rebuildForest(const std::string & vConfigFile)
{
	_ASSERTE(!vConfigFile.empty());
	bool IsConfigParsed = hiveConfig::hiveParseConfig(vConfigFile, hiveConfig::EConfigType::XML, CRegressionForestConfig::getInstance());
	_ASSERTE(IsConfigParsed);
}

//******************************************************************************
//FUNCTION:  for single response
std::vector<float> CRegressionForest::predict(const std::vector<std::vector<float>>& vTestFeatureSet, const std::vector<float>& vTestResponseSet) const
{
	_ASSERTE(!vTestFeatureSet.empty() && !vTestResponseSet.empty());
	std::string PredictionMethodSig = CRegressionForestConfig::getInstance()->getAttribute<std::string>(KEY_WORDS::PREDICTION_METHOD);
	IBasePredictionMethod* pPredictionMethod = dynamic_cast<IBasePredictionMethod*>(hiveOO::CProductFactoryData::getInstance()->createProduct(PredictionMethodSig));
	_ASSERTE(pPredictionMethod);
	std::vector<CTree*> TreeSet = getTreeSet();
	std::vector<float> PredictionSet(vTestFeatureSet.size());
	CTrainingSet* pTrainingSet = CTrainingSet::getInstance();

	for (auto i = 0; i < vTestFeatureSet.size(); i++)
		PredictionSet[i] = pPredictionMethod->predictCertainResponseV(vTestFeatureSet[i], vTestResponseSet[i], TreeSet);
	return PredictionSet;
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
//FUNCTION: for multiResponse
void CRegressionForest::predict(const std::vector<float>& vFeatures, unsigned int vNumOfUsingTrees, bool vIsWeightedPrediction, unsigned int vNumResponse, std::vector<float>& voPredictValue) const
{
	_ASSERTE(!vFeatures.empty());

	voPredictValue.resize(vNumResponse); 
	for (int i = 0; i < vNumResponse; ++i)
		voPredictValue[i] = __predictCertainResponse(vFeatures, vNumOfUsingTrees, vIsWeightedPrediction, i);
}

//****************************************************************************************************
//FUNCTION:
float CRegressionForest::__predictCertainResponse(const std::vector<float>& vFeatures, unsigned int vNumOfUsingTrees, bool vIsWeightedPrediction, unsigned int vResponseIndex) const
{
	_ASSERTE(!vFeatures.empty() && vNumOfUsingTrees > 0);

	float PredictValue = 0.0f;
	std::vector<float> PredictValueOfTree(vNumOfUsingTrees, 0.0f);
	std::vector<float> NodeWeight(vNumOfUsingTrees, 0.0f);

	static std::vector<const CNode*> LeafNodeSet;
	LeafNodeSet.resize(this->getNumOfTrees());
	for (int i = 0; i < vNumOfUsingTrees; ++i)
	{
		if (vResponseIndex == 0)	LeafNodeSet[i] = m_Trees[i]->locateLeafNode(vFeatures);
		PredictValueOfTree[i] = m_Trees[i]->predict(*LeafNodeSet[i], vFeatures, NodeWeight[i], vResponseIndex);
	}

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
		PredictValue = std::accumulate(PredictValueOfTree.begin(), PredictValueOfTree.end(), 0.0f);
		return PredictValue / vNumOfUsingTrees;
	}
}

//****************************************************************************************************
//FUNCTION:
float CRegressionForest::__predictCertainResponse(const std::vector<float>& vFeatures, float vResponse, unsigned int vNumOfUsingTrees, bool vIsWeightedPrediction, float& voMPPredictSet, unsigned int vResponseIndex) const
{
	_ASSERTE(!vFeatures.empty() && vNumOfUsingTrees > 0);

	float PredictValue = 0.0f;
	std::vector<float> PredictValueOfTree(vNumOfUsingTrees, 0.0f), PredictValueOfTreeTemp(vNumOfUsingTrees, 0.0f);
	std::vector<float> NodeWeight(vNumOfUsingTrees, 0.0f);
	std::vector<float> MPDistanceWeight(vNumOfUsingTrees, 0.0f);
	std::vector<std::pair<int, float>> PredictBias(vNumOfUsingTrees);

	std::vector<const CNode*> LeafNodeSet;
	LeafNodeSet.resize(this->getNumOfTrees());
	std::vector<std::vector<SPathNodeInfo>> AllTreePath(vNumOfUsingTrees);
	std::vector<std::vector<float>> OutLeafFeatureRange(vNumOfUsingTrees);
	std::vector<std::vector<float>> OutLeafFeatureSplitRange(vNumOfUsingTrees);
	CPathNodeMethod* pPathNode = CPathNodeMethod::getInstance();
	CMpCompute* pMpCompute = nullptr;
	bool IsPrint =  CTrainingSetConfig::getInstance()->getAttribute<bool>(hiveRegressionForest::KEY_WORDS::IS_PRINT_LEAF_NODE);
	for (int i = 0; i < vNumOfUsingTrees; ++i)
	{
		if (vResponseIndex == 0)
		{
			if (IsPrint)
			{
				std::vector<SPathNodeInfo> CurrentTreePathInfo;
				std::vector<float> TempOutLeafFeatureRange;
				std::vector<float> TempOutLeafFeatureSplitRange;
				LeafNodeSet[i] = m_Trees[i]->recordPathNodeInfo(vFeatures, CurrentTreePathInfo, TempOutLeafFeatureRange, TempOutLeafFeatureSplitRange, i);
				PredictValueOfTree[i] = LeafNodeSet[i]->getNodeMeanV();
				AllTreePath[i] = CurrentTreePathInfo;
				OutLeafFeatureRange[i] = TempOutLeafFeatureRange;
				OutLeafFeatureSplitRange[i] = TempOutLeafFeatureSplitRange;
			}
			else
			{
				LeafNodeSet[i] = m_Trees[i]->locateLeafNode(vFeatures);
				PredictValueOfTree[i] = LeafNodeSet[i]->getNodeMeanV();
				//voMPPredictSet += pPathNode->predictWithMinMPOnWholeDimension(m_Trees[i], vFeatures);
				//MPDistanceWeight[i] = pMpCompute->calMPOutOfFeatureAABB(m_Trees[i], LeafNodeSet[i], vFeatures);
				//MPDistanceWeight[i] = pMpCompute->calMPDissimilarityGlobal(m_Trees[i], LeafNodeSet[i]->getNodeDataIndexV(), vFeatures, PredictValueOfTree[i]);
				std::vector<std::pair<float, float>> ResponseRange;
				std::vector<float> ResponseVariance;
				std::vector<int> ReponseNum = m_Trees[i]->calFeatureRangeResponse(LeafNodeSet[i]->getFeatureRange(), ResponseRange, ResponseVariance);
				MPDistanceWeight[i] = std::accumulate(ResponseVariance.begin(), ResponseVariance.end(), 0.f);
			}
		}
		PredictBias[i] = std::make_pair(i, std::abs(PredictValueOfTree[i] - vResponse) / vResponse);
	}
	if (IsPrint)
	{
		sort(PredictBias.begin(), PredictBias.end(), [](std::pair<int, float>& vFirst, std::pair<int, float>& vSecond) {return vFirst.second < vSecond.second; });
		printAllInfo(vFeatures, PredictBias, LeafNodeSet, AllTreePath, OutLeafFeatureRange, OutLeafFeatureSplitRange);
	}
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
		//voMPPredictSet = voMPPredictSet / vNumOfUsingTrees;
		float SumWeight = std::accumulate(MPDistanceWeight.begin(), MPDistanceWeight.end(), 0.f);
		float MeanWeight = SumWeight / vNumOfUsingTrees, SumWeightUsed = 0.f;
		for (int k = 0; k < vNumOfUsingTrees; ++k)
		{
			if (MPDistanceWeight[k] <= MeanWeight)
			{
				voMPPredictSet += PredictValueOfTree[k] * 1 / MPDistanceWeight[k];
				SumWeightUsed += 1 / MPDistanceWeight[k];
			}
		}
		voMPPredictSet = voMPPredictSet / SumWeightUsed;
		PredictValue = std::accumulate(PredictValueOfTree.begin(), PredictValueOfTree.end(), 0.0f);
		return PredictValue / vNumOfUsingTrees;
	}
	_SAFE_DELETE(pMpCompute);
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

//****************************************************************************************************
//FUNCTION:
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

//******************************************************************************
//FUNCTION:
void CRegressionForest::outputPathNodeInfo(const std::string& vOutputFileName, const std::vector<SPathNodeInfo>& vPathNodeInfo) const
{
	std::ofstream OutputFile;
	OutputFile.open(vOutputFileName, std::ios::app);
	if (OutputFile.is_open())
	{
		for (auto iterNode : vPathNodeInfo)
		{
			OutputFile << iterNode.m_NodeLevel << "," << iterNode.m_SplitFeature << "," << iterNode.m_SplitLocation << std::endl;
			std::vector<float> SplitMin = iterNode.m_FeatureSplitRange.first;
			std::vector<float> SplitMax = iterNode.m_FeatureSplitRange.second;
			std::vector<float> FeatureMin = iterNode.m_FeatureRange.first;
			std::vector<float> FeatureMax = iterNode.m_FeatureRange.second;
			OutputFile << ",";
			for (auto i = 0; i < SplitMin.size(); i++)
				OutputFile << SplitMin[i] << "~" << SplitMax[i] << ",";
			OutputFile << std::endl;
			OutputFile << ",";
			for (auto j = 0; j < FeatureMin.size(); j++)
				OutputFile << FeatureMin[j] << "~" << FeatureMax[j] << ",";
			OutputFile << std::endl;
		}
	}
}

//******************************************************************************
//FUNCTION:
void CRegressionForest::printAllInfo(const std::vector<float>& vFeatures, const std::vector<std::pair<int, float>>& vPredictBias, const std::vector<const CNode*>& vLeafNodeSet, const std::vector<std::vector<SPathNodeInfo>>& vAllTreePath, const std::vector<std::vector<float>>& vOutRange, const std::vector<std::vector<float>>& vOutSplitRange) const
{
	std::string BestTreeFilePath = CTrainingSetConfig::getInstance()->getAttribute<std::string>(hiveRegressionForest::KEY_WORDS::BEST_TREE_PATH);
	std::string BadTreeFilePath = CTrainingSetConfig::getInstance()->getAttribute<std::string>(hiveRegressionForest::KEY_WORDS::BAD_TREE_PATH);
	std::ofstream BestTreeFile;
	std::ofstream BadTreeFile;
	BestTreeFile.open(BestTreeFilePath, std::ios::app);
	BadTreeFile.open(BadTreeFilePath, std::ios::app);
	int NumOfUsingTrees = vPredictBias.size();
	int PrintNum = CTrainingSetConfig::getInstance()->getAttribute<int>(hiveRegressionForest::KEY_WORDS::PRINT_TREE_NUMBER);
	_ASSERTE(PrintNum <= NumOfUsingTrees);
	for (int i = 0; i < PrintNum; i++)
	{
		int BestTreeIndex = vPredictBias[i].first;
		int BadTreeIndex = vPredictBias[NumOfUsingTrees - 1 - i].first;
		BestTreeFile << "Tree" << BestTreeIndex << std::endl;
		BadTreeFile << "Tree" << BadTreeIndex << std::endl;
		//vLeafNodeSet[BestTreeIndex]->outputLeafNodeInfo(BestTreeFilePath);
		m_Trees[BestTreeIndex]->printYRangeWithLeafXRange(vFeatures, BestTreeFilePath, vLeafNodeSet[BestTreeIndex]);
		//outputPathNodeInfo(BestTreeFilePath, vAllTreePath[BestTreeIndex]);
		//outputOutFeatureRange(BestTreeFilePath, vOutSplitRange[BestTreeIndex]);
		//outputOutFeatureRange(BestTreeFilePath, vOutRange[BestTreeIndex]);
		m_Trees[BestTreeIndex]->printResponseInfoInAABB(vFeatures, BestTreeFilePath, vLeafNodeSet[BestTreeIndex]);
		//vLeafNodeSet[BadTreeIndex]->outputLeafNodeInfo(BadTreeFilePath);
		m_Trees[BadTreeIndex]->printYRangeWithLeafXRange(vFeatures, BadTreeFilePath, vLeafNodeSet[BadTreeIndex]);
		//outputPathNodeInfo(BadTreeFilePath, vAllTreePath[BadTreeIndex]);
		//outputOutFeatureRange(BadTreeFilePath, vOutSplitRange[BadTreeIndex]);
		//outputOutFeatureRange(BadTreeFilePath, vOutRange[BadTreeIndex]);
		//m_Trees[BadTreeIndex]->printResponseInfoInAABB(vFeatures, BadTreeFilePath, vLeafNodeSet[BadTreeIndex]);
	}
	BestTreeFile << std::endl;
	BadTreeFile << std::endl;
	BestTreeFile.close();
	BadTreeFile.close();
}

//******************************************************************************
//FUNCTION:
void hiveRegressionForest::CRegressionForest::outputOutFeatureRange(const std::string & vOutputFileName, const std::vector<float>& vOutRange) const
{
	std::ofstream OutputFile;
	OutputFile.open(vOutputFileName, std::ios::app);
	if (OutputFile.is_open())
	{
		OutputFile << "OutFeature" << ",";
		for (auto iterNode : vOutRange)
			OutputFile << iterNode << ",";
		OutputFile << std::endl;
	}
}
