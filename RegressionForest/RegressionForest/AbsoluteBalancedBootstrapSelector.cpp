#include "AbsoluteBalancedBootstrapSelector.h"
#include "RegressionForestCommon.h"
#include "TrainingSet.h"
#include "common/ProductFactory.h"
#include "math/RandomInterface.h"

using namespace hiveRegressionForest;

hiveOO::CProductFactory<CAbsoluteBalancedBootstrapSelector> theCreator(KEY_WORDS::ABSOLUTE_BALANCED_BOOTSTRAP_SELECTOR);

CAbsoluteBalancedBootstrapSelector::CAbsoluteBalancedBootstrapSelector()
{
}

CAbsoluteBalancedBootstrapSelector::~CAbsoluteBalancedBootstrapSelector()
{
}

//****************************************************************************************************
//FUNCTION:
void CAbsoluteBalancedBootstrapSelector::generateBootstrapIndexSetV(unsigned int vInstanceNum, std::vector<int>& voBootstrapIndexSet)
{
	const CTrainingSet *pTrainingSet = CTrainingSet::getInstance();
	const std::vector<float>& ResponseSet = pTrainingSet->getResponseSet();

	float MaxResponse = *std::max_element(ResponseSet.begin(), ResponseSet.end());
	float MinResponse = *std::min_element(ResponseSet.begin(), ResponseSet.end());
	float RangeLength = ceil((MaxResponse - MinResponse) / DEFAULT_GROUP_NUM);

	if (m_GroupedResponseIndex.size() == 0)
	{
		m_GroupedResponseIndex.resize(DEFAULT_GROUP_NUM);
		for (auto i = 0; i < ResponseSet.size(); ++i)
			m_GroupedResponseIndex[(ResponseSet[i] - MinResponse) / RangeLength].push_back(i);
	}

	voBootstrapIndexSet.resize(vInstanceNum);
	for (auto i = 0; i < vInstanceNum; ++i)
	{
		unsigned int GroupIndex = hiveRandom::hiveGenerateRandomInteger(0, m_GroupedResponseIndex.size() - 1);
		unsigned int GroupMemberIndex = hiveRandom::hiveGenerateRandomInteger(0, m_GroupedResponseIndex[GroupIndex].size() - 1);
		voBootstrapIndexSet[i] = m_GroupedResponseIndex[GroupIndex][GroupMemberIndex];
	}
}