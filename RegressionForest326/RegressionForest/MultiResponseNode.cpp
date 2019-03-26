#include "MultiResponseNode.h"
#include "RegressionForestCommon.h"
#include "common/HiveCommonMicro.h"
#include "common/ProductFactory.h"

using namespace hiveRegressionForest;

hiveOO::CProductFactory<CMultiResponseNode> theCreator(KEY_WORDS::MULTI_RESPONSES_NODE);

CMultiResponseNode::CMultiResponseNode()
{
}

CMultiResponseNode::~CMultiResponseNode()
{
}

//****************************************************************************************************
//FUNCTION:
void CMultiResponseNode::createAsLeafNodeV(const std::pair<std::vector<std::vector<float>>, std::vector<float>>& vBootstrapDataset)
{
	m_IsLeafNode = true;
	m_NumResponse = CTrainingSet::getInstance()->getNumOfResponse();	 
	int NumInstance = vBootstrapDataset.second.size() / m_NumResponse;
	_ASSERTE(NumInstance == vBootstrapDataset.first.size());

	m_NodeAvgValuePtr = new float[m_NumResponse]();
	m_NodeVariancePtr = new float[m_NumResponse]();

#pragma omp critical
	{
		for (auto ResponseIndex = 0; ResponseIndex < m_NumResponse; ResponseIndex++)
		{
			for (auto i = 0; i < NumInstance; ++i) m_NodeAvgValuePtr[ResponseIndex] += vBootstrapDataset.second[NumInstance*ResponseIndex + i];
			m_NodeAvgValuePtr[ResponseIndex] /= NumInstance;

			for (auto k = 0; k < NumInstance; ++k) m_NodeVariancePtr[ResponseIndex] += std::pow(vBootstrapDataset.second[NumInstance*ResponseIndex + k] - m_NodeAvgValuePtr[ResponseIndex], 2.0f);
			m_NodeVariancePtr[ResponseIndex] /= NumInstance;
		}
	}

	setBestSplitFeatureAndGap(0, FLT_MAX);
}

//****************************************************************************************************
//FUNCTION:
float CMultiResponseNode::predictV(const std::vector<float>& vFeatureInstance, unsigned int vResponseIndex) const
{
	return m_NodeAvgValuePtr[vResponseIndex];
}

//****************************************************************************************************
//FUNCTION:
float CMultiResponseNode::_getNodeVarianceV(unsigned int vResponseIndex /*= 0*/) const
{
	return m_NodeVariancePtr[vResponseIndex];
}