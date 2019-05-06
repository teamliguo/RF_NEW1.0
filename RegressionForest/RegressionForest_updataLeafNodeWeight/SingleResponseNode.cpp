#include "SingleResponseNode.h"
#include <fstream>
#include "RegressionForestCommon.h"
#include "math/AverageOutputRegression.h"
#include "common/HiveCommonMicro.h"
#include "common/ProductFactory.h"
#include "Utility.h"

using namespace hiveRegressionForest;

hiveOO::CProductFactory<CSingleResponseNode> theCreator(KEY_WORDS::SINGLE_RESPONSE_NODE);

CSingleResponseNode::CSingleResponseNode()
{
}

CSingleResponseNode::~CSingleResponseNode()
{
}

//****************************************************************************************************
//FUNCTION:Native version
void CSingleResponseNode::createAsLeafNodeV(const std::pair<std::vector<std::vector<float>>, std::vector<float>>& vBootstrapDataset)
{
	_ASSERTE(CTrainingSet::getInstance()->getNumOfResponse() == 1);

	m_IsLeafNode = true;

	std::string LeafNodeModelSig = CRegressionForestConfig::getInstance()->getAttribute<std::string>(KEY_WORDS::LEAF_NODE_MODEL_SIGNATURE);
	if (LeafNodeModelSig.empty()) LeafNodeModelSig = KEY_WORDS::REGRESSION_MODEL_AVERAGE;

#pragma omp critical
	{
		calStatisticsV(vBootstrapDataset);
		m_pRegressionModel = hiveRegressionAnalysis::hiveTrainRegressionModel(vBootstrapDataset.first, vBootstrapDataset.second, LeafNodeModelSig);
	}
	setBestSplitFeatureAndGap(0, FLT_MAX);
}

//****************************************************************************************************
//FUNCTION:for record data index
void CSingleResponseNode::createAsLeafNodeV(const std::pair<std::vector<std::vector<float>>, std::vector<float>>& vBootstrapDataset, const std::vector<int>& vDataSetIndex, const std::pair<int, int>& vIndexRange)
{
	_ASSERTE(CTrainingSet::getInstance()->getNumOfResponse() == 1);

	m_IsLeafNode = true;
	setLeafNodeWeight(1.0f);

	std::string LeafNodeModelSig = CRegressionForestConfig::getInstance()->getAttribute<std::string>(KEY_WORDS::LEAF_NODE_MODEL_SIGNATURE);
	if (LeafNodeModelSig.empty()) LeafNodeModelSig = KEY_WORDS::REGRESSION_MODEL_AVERAGE;
	CTrainingSet::getInstance()->recombineBootstrapDataset(vDataSetIndex, vIndexRange, m_DataSetIndex);

#pragma omp critical
	{
		calStatisticsV(vBootstrapDataset);

		m_pRegressionModel = hiveRegressionAnalysis::hiveTrainRegressionModel(vBootstrapDataset.first, vBootstrapDataset.second, LeafNodeModelSig);
	}

	setBestSplitFeatureAndGap(0, FLT_MAX);
}

//****************************************************************************************************
//FUNCTION:
float CSingleResponseNode::predictV(const std::vector<float>& vFeatureInstance, unsigned int vResponseIndex) const
{
	_ASSERTE(!vFeatureInstance.empty());

	return hiveRegressionAnalysis::hiveExecuteRegression(m_pRegressionModel, vFeatureInstance);
}

//****************************************************************************************************
//FUNCTION:
float CSingleResponseNode::getNodeVarianceV(unsigned int vResponseIndex /*= 0*/) const
{
	return m_NodeVariance;
}

//****************************************************************************************************
//FUNCTION:
float CSingleResponseNode::getNodeMeanV(unsigned int vResponseIndex /*= 0*/) const
{
	return m_NodeMean;
}

//****************************************************************************************************
//FUNCTION:
void CSingleResponseNode::calStatisticsV(const std::pair<std::vector<std::vector<float>>, const std::vector<float>>& vBootstrapDataset)
{
	m_NodeVariance	= var(vBootstrapDataset.second);
	m_NodeMean		= mean(vBootstrapDataset.second);
	calFeatureRange(vBootstrapDataset.first, m_FeatureRange);
}

//******************************************************************************
//FUNCTION:
void CSingleResponseNode::outputLeafNodeInfoV(const std::string & vFilePath) const
{
	const std::vector<int>& LeafNodeDataIndex = getNodeDataIndex();
	std::pair<std::vector<std::vector<float>>, std::vector<float>> BootstrapDataset;
	CTrainingSet* pTrainingSet = CTrainingSet::getInstance();
	for (auto iter : LeafNodeDataIndex)
	{
		BootstrapDataset.first.push_back(pTrainingSet->getFeatureInstanceAt(iter));
		BootstrapDataset.second.push_back(pTrainingSet->getResponseValueAt(iter));
	}
	std::vector<std::vector<float>> FeatureSet = BootstrapDataset.first;
	std::vector<float> ResponseSet = BootstrapDataset.second;
	_ASSERT(FeatureSet.size() == ResponseSet.size() && FeatureSet.size() != 0);
	int DataNum = FeatureSet.size(), FeatureNum = FeatureSet[0].size();
	std::ofstream PrintFile;
	PrintFile.open(vFilePath, std::ios::app);

	if (PrintFile.is_open())
	{
		PrintFile << "TrainData" << ",";
		for (int i = 0; i < FeatureSet[0].size(); i++)
			PrintFile << "x" << i << ",";
		PrintFile << "y" << std::endl;
		for (int i = 0; i < ResponseSet.size(); i++)
		{
			PrintFile << i << ",";
			for (int k = 0; k < FeatureSet[0].size(); k++)
				PrintFile << FeatureSet[i][k] << ",";
			PrintFile << ResponseSet[i] << std::endl;
		}

		std::vector<float> FeatureMean(FeatureSet[0].size(), 0.f);
		std::vector<float> FeatureVar(FeatureSet[0].size(), 0.f);
		for (int i = 0; i < FeatureSet[0].size(); i++)
		{
			for (int k = 0; k < FeatureSet.size(); k++)
				FeatureMean[i] += FeatureSet[k][i];
			FeatureMean[i] /= FeatureSet.size();
			for (int k = 0; k < FeatureSet.size(); k++)
				FeatureVar[i] += (FeatureSet[k][i] - FeatureMean[i])*(FeatureSet[k][i] - FeatureMean[i]);
			FeatureVar[i] = FeatureVar[i] / FeatureSet.size();
		}

		PrintFile << "mean" << ",";
		for (int i = 0; i < FeatureMean.size(); i++) PrintFile << FeatureMean[i] << ",";
		PrintFile << getNodeMeanV() << std::endl;
		PrintFile << "var" << ",";
		for (int i = 0; i < FeatureVar.size(); i++) PrintFile << FeatureVar[i] << ",";
		PrintFile << getNodeVarianceV() << std::endl;
	}
}
