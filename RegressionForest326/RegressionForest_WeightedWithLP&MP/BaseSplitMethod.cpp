#include "BaseSplitMethod.h"
#include "common/HiveCommonMicro.h"
#include "common/ProductFactoryData.h"
#include "common/CommonInterface.h"
#include "TrainingSet.h"
#include "RegressionForestConfig.h"
#include "RegressionForestCommon.h"

using namespace hiveRegressionForest;

INodeSpliter::INodeSpliter()
{
}

INodeSpliter::~INodeSpliter()
{
}

//****************************************************************************************************
//FUNCTION:
bool INodeSpliter::splitNode(CNode* vCurrentNode, const std::pair<int, int>& vBootstrapRange, std::vector<int>& vBootstrapIndex, const std::vector<int>& vCurrentFeatureIndexSubSet, int& voRangeSplitPos)
{
	// NOTES : 初始化为-1，如果执行子特征划分查找以后仍然为-1，则表示：
	//		   选择的子特征集合中各子特征所有的值均相等
	SSplitHyperplane AxisAlignedSplitHyperplane({-1, FLT_MAX});
	__findBestSplitHyperplane(vBootstrapIndex, vBootstrapRange, vCurrentFeatureIndexSubSet, AxisAlignedSplitHyperplane);
	if (AxisAlignedSplitHyperplane.m_AxisAlignedSplitHyperplane.first == -1) return false;
	
	voRangeSplitPos = __processBootstrapRange(vBootstrapIndex, vBootstrapRange, AxisAlignedSplitHyperplane);
	vCurrentNode->setBestSplitFeatureAndGap(AxisAlignedSplitHyperplane.m_AxisAlignedSplitHyperplane.first, AxisAlignedSplitHyperplane.m_AxisAlignedSplitHyperplane.second);

	return true;
}

//********************************************************************************************************
//FUNCTION:
void INodeSpliter::_generateSortedFeatureResponsePairSetV(std::vector<int>& vBootstrapIndex, const std::pair<int, int>& vBootstrapRange, unsigned int vFeatureIndex, std::vector<std::pair<float, float>>& voSortedFeatureResponseSet)
{
	_ASSERTE(!vBootstrapIndex.empty());

	voSortedFeatureResponseSet.resize(vBootstrapRange.second - vBootstrapRange.first);

	const CTrainingSet *pTrainingSet = CTrainingSet::getInstance();
	int count = 0;
	for (int i = vBootstrapRange.first; i<vBootstrapRange.second; ++i)
	{
		voSortedFeatureResponseSet[count++] = std::make_pair(pTrainingSet->getFeatureValueAt(vBootstrapIndex[i], vFeatureIndex), pTrainingSet->getResponseValueAt(vBootstrapIndex[i]));
	}

	std::sort(voSortedFeatureResponseSet.begin(), voSortedFeatureResponseSet.end(), [](const std::pair<float, float>& P1, const std::pair<float, float>& P2) {return P1.first < P2.first; });
}

//****************************************************************************************************
//FUNCTION:
void INodeSpliter::__findBestSplitHyperplane(std::vector<int>& vBootstrapIndex, const std::pair<int, int>& vBootstrapRange, const std::vector<int>& vFeatureIndexSubset, SSplitHyperplane& voSplitHyperplane)
{
	_ASSERTE((vBootstrapRange.second - vBootstrapRange.first) > 0 && !vBootstrapIndex.empty() && !vFeatureIndexSubset.empty());

	float MaxObjFuncVal = -FLT_MAX, SumY = 0.0f;
	const CTrainingSet* pTrainingSet = CTrainingSet::getInstance();
	for (int i = vBootstrapRange.first; i < vBootstrapRange.second; ++i) SumY += pTrainingSet->getResponseValueAt(vBootstrapIndex[i]);

	std::vector<std::pair<float, float>> FeatureResponseSet;
	for (auto FeatureIndex : vFeatureIndexSubset)
	{
		_generateSortedFeatureResponsePairSetV(vBootstrapIndex, vBootstrapRange, FeatureIndex, FeatureResponseSet);

		if (FeatureResponseSet[0].first >= FeatureResponseSet[FeatureResponseSet.size() - 1].first) continue; 
	
		float CurrentFeatureMaxObjVal = -FLT_MAX, BestGap = 0.0f;
		__findLocalBestSplitHyperplaneV(FeatureResponseSet, SumY, CurrentFeatureMaxObjVal, BestGap);

		if (CurrentFeatureMaxObjVal > MaxObjFuncVal)
		{
			MaxObjFuncVal = CurrentFeatureMaxObjVal;

			voSplitHyperplane.m_AxisAlignedSplitHyperplane.first = FeatureIndex;
			voSplitHyperplane.m_AxisAlignedSplitHyperplane.second = BestGap;
		}
	}
}

//****************************************************************************************************
//FUNCTION:
int INodeSpliter::__processBootstrapRange(std::vector<int>& vBootstrapIndex, const std::pair<int, int>& vBootstrapRange, SSplitHyperplane& vSplitHyperplane)
{
	const CTrainingSet* pTrainingSet = CTrainingSet::getInstance();
	int L = vBootstrapRange.first, R = vBootstrapRange.second - 1;
	bool IsCurLInLeftSpace = false, IsCurRInLeftSpace = false;
	while (L <= R)
	{		
		IsCurLInLeftSpace = vSplitHyperplane.IsInstanceInLeftSpace(CTrainingSet::getInstance()->getFeatureInstanceAt(vBootstrapIndex[L]));
		IsCurRInLeftSpace = vSplitHyperplane.IsInstanceInLeftSpace(CTrainingSet::getInstance()->getFeatureInstanceAt(vBootstrapIndex[R]));

		if (IsCurLInLeftSpace)
		{
			L++;
			if (!IsCurRInLeftSpace) R--;
		}
		else
		{
			if (IsCurRInLeftSpace)
			{
				std::swap(vBootstrapIndex[L], vBootstrapIndex[R]);
				L++;
			}
			R--;
		}
	}

	_ASSERTE((L > vBootstrapRange.first) && (L <= vBootstrapRange.second)); //NOTES : L指向尾后元素，故这样Assert

	return L;
}
