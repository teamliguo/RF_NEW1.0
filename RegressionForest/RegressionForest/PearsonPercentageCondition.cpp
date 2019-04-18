#include "PearsonPercentageCondition.h"
#include "RegressionForestConfig.h"
#include "RegressionForestCommon.h"
#include "common/ProductFactory.h"

using namespace hiveRegressionForest;

hiveOO::CProductFactory<CPearsonPercentageCondition> theCreator(KEY_WORDS::PEARSON_PERCENTAGE_CONDITION);

CPearsonPercentageCondition::CPearsonPercentageCondition()
{
}

CPearsonPercentageCondition::~CPearsonPercentageCondition()
{
}

//****************************************************************************************************
//FUNCTION:
bool CPearsonPercentageCondition::isMeetTerminateConditionV(const std::vector<std::vector<float>>& vFeatureSet, const std::vector<float>& vResponseSet, boost::any vExtra)
{
	float PearsonPercentaget = boost::any_cast<float>(vExtra);

	float PresentOverallProportion = CRegressionForestConfig::getInstance()->getAttribute<float>(KEY_WORDS::PRESENT_OVERALL_PROPORTION);
	unsigned int MaxLeafNodeSamples = CRegressionForestConfig::getInstance()->getAttribute<int>(KEY_WORDS::MAX_LEAF_NODE_INSTANCE_SIZE);

	_ASSERTE(PearsonPercentaget >= 0.0f && PearsonPercentaget <= 1.0f);
	return (PearsonPercentaget >= PresentOverallProportion || vFeatureSet.size() <= MaxLeafNodeSamples);
}