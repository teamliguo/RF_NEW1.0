#include "BalancedBootstrapSelector.h"
#include "RegressionForestCommon.h"
#include "TrainingSet.h"
#include "common/ProductFactory.h"
#include "common/HiveCommonMicro.h"
#include "math/RandomInterface.h"
#include <random>

using namespace hiveRegressionForest;

hiveOO::CProductFactory<CBalancedBootstrapSelector> theCreator(KEY_WORDS::BALANCED_BOOTSTRAP_SELECTOR);

CBalancedBootstrapSelector::CBalancedBootstrapSelector()
{
}

CBalancedBootstrapSelector::~CBalancedBootstrapSelector()
{
}

//****************************************************************************************************
//FUNCTION:
void CBalancedBootstrapSelector::generateBootstrapIndexSetV(unsigned int vInstanceNum, std::vector<int>& voBootstrapIndexSet)
{
	if (m_MaxGroupSize == 0) m_MaxGroupSize = __divideResponseByRange(vInstanceNum); // NOTES : 控制仅建立一次

	voBootstrapIndexSet.resize(vInstanceNum);
	for (auto &InstanceIndex : voBootstrapIndexSet)
	{
		unsigned int GroupIndex = hiveRandom::hiveGenerateRandomInteger(0, DEFAULT_GROUP_NUM - 1);
		unsigned int GroupMemberIndex = hiveRandom::hiveGenerateRandomInteger(0, m_MaxGroupSize-1);
		InstanceIndex = m_ResponseAndModIndexSet[GroupIndex].first[m_ResponseAndModIndexSet[GroupIndex].second[GroupMemberIndex]];
	}
}

//****************************************************************************************************
//FUNCTION:
unsigned int CBalancedBootstrapSelector::__divideResponseByRange(unsigned int vInstanceNum)
{
	const CTrainingSet *pTrainingSet = CTrainingSet::getInstance();
	const std::vector<float>& ResponseSet = pTrainingSet->getResponseSet();

	float MaxResponse = *std::max_element(ResponseSet.begin(), ResponseSet.end());
	float MinResponse = *std::min_element(ResponseSet.begin(), ResponseSet.end());
	float RangeLength = ceil((MaxResponse - MinResponse) / DEFAULT_GROUP_NUM);

	for (auto i = 0; i < ResponseSet.size(); ++i) m_ResponseAndModIndexSet[(ResponseSet[i] - MinResponse) / RangeLength].first.push_back(i);

	unsigned int MaxGroupSize = 0;
	for (auto RangePair : m_ResponseAndModIndexSet)	MaxGroupSize = _MAX(MaxGroupSize, RangePair.second.first.size());

	std::vector<int> RandPermutation(MaxGroupSize);
	for (auto i = 0; i < MaxGroupSize; ++i) RandPermutation[i] = i;
	
	std::random_device RandomDevice;
	std::mt19937 Generator(RandomDevice());
	for (auto i = 0; i < DEFAULT_GROUP_NUM; ++i)
	{
		std::shuffle(RandPermutation.begin(), RandPermutation.end(), Generator);

		for (auto k = 0; k < MaxGroupSize; ++k) 
			m_ResponseAndModIndexSet[i].second.push_back(RandPermutation[k] % m_ResponseAndModIndexSet[i].first.size());
	}

	return MaxGroupSize;
}