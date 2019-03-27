#include "MultiInfoGainSplit.h"
#include "RegressionForestCommon.h"
#include "TrainingSet.h"
#include "common/productfactory.h"

using namespace hiveRegressionForest;

hiveOO::CProductFactory<hiveRegressionForest::CMultiInfoGainSpliter> theCreator(KEY_WORDS::MULTI_INFO_GAIN_METHOD);

CMultiInfoGainSpliter::CMultiInfoGainSpliter()
{
}

CMultiInfoGainSpliter::~CMultiInfoGainSpliter()
{
}

//********************************************************************************************************
//FUNCTION:
void CMultiInfoGainSpliter::_generateSortedFeatureResponsePairSetV(std::vector<int>& vBootstrapIndex, const std::pair<int, int>& vBootstrapRange, unsigned int vFeatureIndex, std::vector<std::pair<float, float>>& voSortedFeatureResponseSet)
{
	_ASSERTE(!vBootstrapIndex.empty());

	int NumOfResponses = CTrainingSet::getInstance()->getNumOfResponse();
	int Range = vBootstrapRange.second - vBootstrapRange.first;
	voSortedFeatureResponseSet.resize(Range * NumOfResponses);

	const CTrainingSet *pTrainingSet = CTrainingSet::getInstance();
	int count = 0;

	for (int ResponseIndex = 0; ResponseIndex < NumOfResponses; ++ResponseIndex)
	{
		for (int i = vBootstrapRange.first; i < vBootstrapRange.second; ++i)
		{
			voSortedFeatureResponseSet[count++] = std::make_pair(pTrainingSet->getFeatureValueAt(vBootstrapIndex[i], vFeatureIndex), pTrainingSet->getResponseValueAt(vBootstrapIndex[i], ResponseIndex));
		}
		std::sort(voSortedFeatureResponseSet.begin() + (Range * ResponseIndex), voSortedFeatureResponseSet.begin() + (Range * (ResponseIndex + 1)), [](const std::pair<float, float>& P1, const std::pair<float, float>& P2) {return P1.first < P2.first; });
	}
	// NOTE: voSortedFeatureResponseSet.second 中连续存储m维响应的
}

//****************************************************************************************************
//FUNCTION:
void CMultiInfoGainSpliter::__findLocalBestSplitHyperplaneV(const std::vector<std::pair<float, float>>& vFeatureResponseSet, float vSum, float& voCurrentFeatureMaxObjVal, float& voCurBestGap)
{
	_ASSERTE(!vFeatureResponseSet.empty());
	int NumOfResponses = CTrainingSet::getInstance()->getNumOfResponse();
	int Range = vFeatureResponseSet.size() / NumOfResponses;
	std::vector<float> SumVec(NumOfResponses);

	SumVec[0] = vSum;
	for (auto ResponseIndex = 1; ResponseIndex < NumOfResponses; ++ResponseIndex)
		for (auto k = Range * ResponseIndex; k < Range * (ResponseIndex + 1); ++k) SumVec[ResponseIndex] += vFeatureResponseSet[k].second;

	std::vector<float> SumL(3, 0.0), SumR = SumVec;
	int NumL = 0, NumR = Range;

	std::vector<float> CurrentNodeEntropy(NumOfResponses);
	for (auto ResponseIndex = 0; ResponseIndex < NumOfResponses; ++ResponseIndex) CurrentNodeEntropy[ResponseIndex] = SumVec[ResponseIndex] * SumVec[ResponseIndex] / vFeatureResponseSet.size();

	for (auto InstanceIndex = 0; InstanceIndex < Range - 1; ++InstanceIndex)
	{
		for (auto ResponseIndex = 0; ResponseIndex < NumOfResponses; ++ResponseIndex)
		{
			SumL[ResponseIndex] += vFeatureResponseSet[Range*ResponseIndex + InstanceIndex].second;
			SumR[ResponseIndex] -= vFeatureResponseSet[Range*ResponseIndex + InstanceIndex].second;
		}
		++NumL;
		--NumR;

		if (vFeatureResponseSet[InstanceIndex].first < vFeatureResponseSet[InstanceIndex + 1].first) // NOTES: ! this line is very important !!!!!!!!!
		{
			float CurrentObjFuncVal = 0;
			for (auto ResponseIndex = 0; ResponseIndex < NumOfResponses; ++ResponseIndex)
				CurrentObjFuncVal += (SumL[ResponseIndex] * SumL[ResponseIndex] / NumL) + (SumR[ResponseIndex] * SumR[ResponseIndex] / NumR) - CurrentNodeEntropy[ResponseIndex];

			if (CurrentObjFuncVal > voCurrentFeatureMaxObjVal)
			{
				voCurrentFeatureMaxObjVal = CurrentObjFuncVal;
				voCurBestGap = (vFeatureResponseSet[InstanceIndex].first + vFeatureResponseSet[InstanceIndex + 1].first) / 2.0f;
			}
		}
	}
}