#include "SingleResponseNode.h"
#include "RegressionForestCommon.h"
#include "math/AverageOutputRegression.h"
#include "common/HiveCommonMicro.h"
#include "common/ProductFactory.h"

using namespace hiveRegressionForest;

hiveOO::CProductFactory<CSingleResponseNode> theCreator(KEY_WORDS::SINGLE_RESPONSE_NODE);

CSingleResponseNode::CSingleResponseNode()
{
}

CSingleResponseNode::~CSingleResponseNode()
{
}

//****************************************************************************************************
//FUNCTION:
void CSingleResponseNode::createAsLeafNodeV(const std::pair<std::vector<std::vector<float>>, std::vector<float>>& vBootstrapDataset)
{
	_ASSERTE(CTrainingSet::getInstance()->getNumOfResponse() == 1);

	m_IsLeafNode = true;
	m_BootstrapDataset = vBootstrapDataset;

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
//FUNCTION:
void CSingleResponseNode::createAsLeafNodeV(const std::pair<std::vector<std::vector<float>>, std::vector<float>>& vBootstrapDataset, const std::vector<int>& vDataSetIndex, const std::pair<int, int>& vIndexRange)
{
	_ASSERTE(CTrainingSet::getInstance()->getNumOfResponse() == 1);

	m_IsLeafNode = true;

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
void CSingleResponseNode::calStatisticsV(const std::pair<std::vector<std::vector<float>>, std::vector<float>>& vBootstrapDataset)
{
	m_NodeVariance = _calculateVariance(vBootstrapDataset.second);
	m_NodeMean = _calculateMean(vBootstrapDataset.second);
	m_FeatureRange = calFeatureRange(vBootstrapDataset.first);
}

//****************************************************************************************************
//FUNCTION:
std::vector<int> CSingleResponseNode::getNodeDataIndexV() const
{
	return m_DataSetIndex;
}