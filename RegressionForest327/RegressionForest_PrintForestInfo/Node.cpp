#include "Node.h"
#include <numeric>
#include "common/OOInterface.h"
#include "common/HiveCommonMicro.h"
#include "common/ProductFactoryData.h"
#include "TrainingSet.h"
#include "ObjectPool.h"
#include "RegressionForestConfig.h"
#include "RegressionForestCommon.h"
#include <numeric>
#include <fstream>

using namespace hiveRegressionForest;

CNode::CNode()
{
}

CNode::CNode(unsigned int vLevel) : m_Level(vLevel)
{
}

CNode::~CNode()
{
	// NOTES : boost object_pool auto-release memory
}

//********************************************************************************************************
//FUNCTION:
bool CNode::operator==(const CNode& vNode) const
{
	if (m_Level != vNode.getLevel()) return false;
	if (fabs(m_BestGap - vNode.getBestGap()) >= FLT_EPSILON) return false;
	if (m_BestSplitFeatureIndex != vNode.getBestSplitFeatureIndex()) return false;

	if (m_IsLeafNode && vNode.isLeafNode())
	{
		return (m_NodeSize == vNode.getNodeSize()); //TODO: 如何比较
	}
	else
	{
		return m_pLeftChild->operator==(vNode.getLeftChild()) &&
			m_pRightChild->operator==(vNode.getRightChild());
	}
}

//****************************************************************************************************
//FUNCTION:
bool CNode::setLeftChild(CNode* vNode)
{
	if (m_pLeftChild) return false;
	m_pLeftChild = vNode;
	return true;
}

//****************************************************************************************************
//FUNCTION:
bool CNode::setRightChild(CNode* vNode)
{
	if (m_pRightChild) return false;
	m_pRightChild = vNode;
	return true;
}

//******************************************************************************
//FUNCTION:
void CNode::setSubEachFeatureSplitRange(const std::pair<std::vector<float>, std::vector<float>>& vSplitRange)
{
	m_SplitRange = vSplitRange;
}

//******************************************************************************
//FUNCTION:
void CNode::updataFeatureSplitRange(const std::pair<std::vector<float>, std::vector<float>>& vParentRange, int vFeatureIndex, float vSplitLocaiton, bool vIsMin)
{
	m_SplitRange = vParentRange;
	if (vIsMin)
		m_SplitRange.second[vFeatureIndex] = vSplitLocaiton;
	else
		m_SplitRange.first[vFeatureIndex] = vSplitLocaiton;
}

//******************************************************************************
//FUNCTION:
void CNode::outputLeafNodeInfo(const std::string & vFilePath) const
{
	if (isLeafNode())
	{
		const std::pair<std::vector<std::vector<float>>, std::vector<float>>& BootstrapDataset = getBootstrapDataset();
		const std::vector<std::vector<float>>& FeatureSet = BootstrapDataset.first;
		const std::vector<float>& ResponseSet = BootstrapDataset.second;
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
}

//****************************************************************************************************
//FUNCTION:
bool CNode::isUnfitted() const
{
	return m_IsLeafNode && m_NodeSize > CRegressionForestConfig::getInstance()->getAttribute<int>(KEY_WORDS::MAX_LEAF_NODE_INSTANCE_SIZE);
}

//****************************************************************************************************
//FUNCTION:
std::pair<std::vector<float>, std::vector<float>> CNode::calFeatureRange(const std::vector<std::vector<float>>& vFeatureSet)
{
	_ASSERT(vFeatureSet.size() > 0);
	
	std::vector<float> MinFeature, MaxFeature;
	for (int i = 0; i < vFeatureSet[0].size(); i++)
	{
		std::vector<float> Column(vFeatureSet.size());
		for (int k = 0; k < vFeatureSet.size(); k++)
			Column[k] = vFeatureSet[k][i];
		float max = *std::max_element(Column.begin(), Column.end());
		float min = *std::min_element(Column.begin(), Column.end());
		MinFeature.push_back(min);
		MaxFeature.push_back(max);
	}
	return std::make_pair(MinFeature, MaxFeature);
}

//******************************************************************************
//FUNCTION:
std::vector<float> CNode::calOutFeatureRange(const std::vector<float>& vFeatures) const
{
	std::vector<float> OutFeatureRange(vFeatures.size());
	std::pair<std::vector<float>, std::vector<float>> LeafFeatureSet = getFeatureRange();
	for (auto i = 0; i < LeafFeatureSet.first.size(); i++)
	{
		if (vFeatures[i] >= LeafFeatureSet.first[i] && vFeatures[i] <= LeafFeatureSet.second[i])
			OutFeatureRange[i] = 0.f;
		else if (vFeatures[i] < LeafFeatureSet.first[i])
			OutFeatureRange[i] = LeafFeatureSet.first[i] - vFeatures[i];
		else
			OutFeatureRange[i] = vFeatures[i] - LeafFeatureSet.second[i];
	}
	return OutFeatureRange;
}

//******************************************************************************
//FUNCTION:
std::vector<float> CNode::calOutFeatureSplitRange(const std::vector<float>& vFeatures) const
{
	std::vector<float> OutFeatureSplitRange(vFeatures.size());
	const std::pair<std::vector<float>, std::vector<float>> LeafFeatureSet = getFeatureSplitRange();
	for (auto i = 0; i < LeafFeatureSet.first.size(); i++)
	{
		if (vFeatures[i] >= LeafFeatureSet.first[i] && vFeatures[i] <= LeafFeatureSet.second[i])
			OutFeatureSplitRange[i] = 0.f;
		else if (vFeatures[i] < LeafFeatureSet.first[i])
			OutFeatureSplitRange[i] = LeafFeatureSet.first[i] - vFeatures[i];
		else
			OutFeatureSplitRange[i] = vFeatures[i] - LeafFeatureSet.second[i];
	}
	return OutFeatureSplitRange;
}

//****************************************************************************************************
//FUNCTION:
float CNode::_calculateVariance(const std::vector<float>& vNumVec)
{
	_ASSERTE(!vNumVec.empty());
		
	float Mean = std::accumulate(vNumVec.begin(), vNumVec.end(), 0.0f) / vNumVec.size();

	float Variance = 0.0f;
	for (auto Num : vNumVec) Variance += std::pow(std::abs(Num - Mean), 2.0f);

	return Variance / vNumVec.size();
}

//****************************************************************************************************
//FUNCTION:
float CNode::_calculateMean(const std::vector<float>& vNumVec)
{
	_ASSERTE(!vNumVec.empty());

	float Mean = std::accumulate(vNumVec.begin(), vNumVec.end(), 0.0f) / vNumVec.size();

	return Mean;
}

//****************************************************************************************************
//FUNCTION:
float CNode::calculateNodeWeight(unsigned int vResponseIndex /*= 0*/) const
{
	float Delta = 1.0f;

	return (1.0f / (getNodeVarianceV(vResponseIndex) + Delta));
}
