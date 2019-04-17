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
float CMpCompute::computeMPOfTwoFeatures(const CTree* vTree, const std::vector<float>& vLeafDate, const std::vector<float>& vTestData, float vPredictResponse)
{
	std::vector<std::pair<float, float>> MaxMinValue;
	for (int j = 0; j < vLeafDate.size(); j++)
		MaxMinValue.push_back({ std::max(vLeafDate[j], vTestData[j]), std::min(vLeafDate[j], vTestData[j])});

	return __calMPValue(vTree, MaxMinValue);
}

//****************************************************************************************************
//FUNCTION:
float CMpCompute::calMPOutOfFeatureAABB(const CTree* vTree, const CNode* vNode, const std::vector<float>& vFeature)
{
	_ASSERTE(!vFeature.empty());

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
void CMpCompute::__countIntervalNode(const CTree * vTree, const std::vector<std::pair<float, float>>& vMaxMinValue, std::vector<int>& voIntervalCount, std::vector<std::pair<float, float>>& voIntervalResponseRange)
{
	_ASSERTE(!vMaxMinValue.empty());

	std::vector<std::vector<std::pair<float, float>>> SortedFeatureResponsePairSet = vTree->getSortedFeatureResponsePairSet();
	for (int i = 0; i < vMaxMinValue.size(); i++)
	{
		std::vector<float> IntervalResponses = __obtainIntervalDataInRange(vMaxMinValue[i].first, vMaxMinValue[i].second, SortedFeatureResponsePairSet[i]);
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
	_ASSERTE(!vMaxMinValue.empty());

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