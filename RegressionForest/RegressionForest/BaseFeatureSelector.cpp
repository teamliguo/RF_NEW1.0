#include "BaseFeatureSelector.h"
#include "RegressionForestCommon.h"
#include "RegressionForestConfig.h"

using namespace hiveRegressionForest;

IFeatureSelector::IFeatureSelector()
{
	__init();
}

//****************************************************************************************************
//FUNCTION:
void IFeatureSelector::__init()
{
	if (CRegressionForestConfig::getInstance()->isAttributeExisted(KEY_WORDS::NUMBER_CANDIDATE_FEATURE))
	{
		m_NumCandidataFeature = CRegressionForestConfig::getInstance()->getAttribute<int>(KEY_WORDS::NUMBER_CANDIDATE_FEATURE);
	}
	else
	{
		m_NumCandidataFeature = 0;
	}
}
