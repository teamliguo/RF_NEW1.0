#include "MpCompute.h"
#include "Utility.h"

using namespace hiveRegressionForest;

CMpCompute::CMpCompute()
{
}

CMpCompute::~CMpCompute()
{
}

//****************************************************************************************************
//FUNCTION:
float CMpCompute::computeMpOfTwoFeatures(const CTree* vTree, const std::vector<float>& vLeafDate, const std::vector<float>& vTestData, float vPredictResponse)
{
	std::vector<std::pair<float, float>> MaxMinValue;
	CTrainingSet *pTrainingSet = CTrainingSet::getInstance();
	for (int j = 0; j < vLeafDate.size(); j++)
	{
		float StandardDeviation = pTrainingSet->getEachDimStandard()[j];
		MaxMinValue.push_back({ std::max(vLeafDate[j], vTestData[j]) + StandardDeviation, std::min(vLeafDate[j], vTestData[j]) - StandardDeviation });
	}

	return __calMPValue(vTree, MaxMinValue);
}

//****************************************************************************************************
//FUNCTION:
std::pair<int, float> CMpCompute::calMinMPAndIndex(const CTree* vTree, const std::vector<int>& vDataIndex, const std::vector<float>& vFeature)
{
	_ASSERT(!vDataIndex.empty());

	float MinMP = FLT_MAX;
	int MinMPIndex = vDataIndex[0];
	std::vector<float> MP(vDataIndex.size(), 0.f);
	CTrainingSet *pTrainingSet = CTrainingSet::getInstance();
	for (int i = 0; i < vDataIndex.size(); ++i)
	{
		MP[i] = computeMpOfTwoFeatures(vTree, pTrainingSet->getFeatureInstanceAt(vDataIndex[i]), vFeature);
		if (MP[i] < MinMP)
		{
			MinMP = MP[i];
			MinMPIndex = vDataIndex[i];
		}
	}

	return std::make_pair(MinMPIndex, MinMP);
}

//****************************************************************************************************
//FUNCTION:
float CMpCompute::calMPDissimilarityGlobal(const CTree* vTree, const std::vector<int>& vLeafIndex, const std::vector<float>& vFeature, float vPredictResponse)
{
	float SumMP = 0.f;
	CTrainingSet* pTrainingSet = CTrainingSet::getInstance();
	for (int i = 0; i < vLeafIndex.size(); i++)
	{
		std::vector<float> LeafNodeFeatureValue = pTrainingSet->getFeatureInstanceAt(vLeafIndex[i]);
		float MPValue = computeMpOfTwoFeatures(vTree, LeafNodeFeatureValue, vFeature, vPredictResponse);
		SumMP += MPValue;
	}
	return SumMP / vLeafIndex.size();
}

//****************************************************************************************************
//FUNCTION:
float CMpCompute::calMPOutOfFeatureAABB(const CTree* vTree, const CNode* vNode, const std::vector<float>& vFeature)
{
	std::pair<std::vector<float>, std::vector<float>> FeatureRange = vNode->getFeatureRange();
	std::vector<std::pair<float, float>> MaxMinValue;
	for (int i = 0; i < vFeature.size(); ++i)
	{
		if (vFeature[i] < FeatureRange.first[i])
			MaxMinValue.push_back(std::make_pair(vFeature[i], FeatureRange.first[i]));
		else if (vFeature[i] > FeatureRange.second[i])
			MaxMinValue.push_back(std::make_pair(FeatureRange.second[i], vFeature[i]));
		else
			MaxMinValue.push_back(std::make_pair(0.f, 0.f));
	}

	return __calMPValue(vTree, MaxMinValue);
}

//****************************************************************************************************
//FUNCTION:
void CMpCompute::generateSortedFeatureResponsePairSet(const std::vector<std::vector<float>>& vFeatureSet, const std::vector<float>& vResponseSet, unsigned int vFeatureIndex, std::vector<std::pair<float, float>>& voSortedFeatureResponseSet)
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
void CMpCompute::__countIntervalNode(const CTree * vTree, const std::vector<std::pair<float, float>>& vMaxMinValue, std::vector<int>& voIntervalCount, std::vector<std::pair<float, float>>& voIntervalResponseRange)
{
	std::vector<std::vector<std::pair<float, float>>> SortedFeatureResponsePairSet = vTree->getSortedFeatureResponsePairSet();
	for (int i = 0; i < vMaxMinValue.size(); i++)
	{
		std::vector<float> IntervalResponses = calSecondParRange(vMaxMinValue[i].first, vMaxMinValue[i].second, SortedFeatureResponsePairSet[i]);
		float MinResponse = *std::min_element(IntervalResponses.begin(), IntervalResponses.end());
		float MaxResponse = *std::max_element(IntervalResponses.begin(), IntervalResponses.end());
		voIntervalResponseRange[i] = std::make_pair(MinResponse, MaxResponse);
		voIntervalCount[i] = IntervalResponses.size();
	}
}

//****************************************************************************************************
//FUNCTION:
float CMpCompute::__calMPValue(const CTree * vTree, const std::vector<std::pair<float, float>>& vMaxMinValue)
{
	float MPParam = 2.0f, MPValueSum = 0.f;
	std::vector<int> EachFeatureIntervalCount(vMaxMinValue.size(), 0);
	std::vector<std::pair<float, float>> ResponseRange(vMaxMinValue.size());
	CTrainingSet* pTrainingSet = CTrainingSet::getInstance();

	__countIntervalNode(vTree, vMaxMinValue, EachFeatureIntervalCount, ResponseRange);

	for (int i = 0; i < EachFeatureIntervalCount.size(); i++)
	{
		MPValueSum += pow(((float)EachFeatureIntervalCount[i] / (float)pTrainingSet->getNumOfInstances()) /** MPParam*/ /**((ResponseRange[i].first - ResponseRange[i].second) / m_ResponseRange)*/, MPParam);
	}
	return pow(MPValueSum, 1 / MPParam);
}