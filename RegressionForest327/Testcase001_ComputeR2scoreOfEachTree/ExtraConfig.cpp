#include "ExtraConfig.h"
#include "ExtraCommon.h"
#include "../RegressionForest_ComputeR2scoreOfEachTree/RegressionForestCommon.h"

using namespace hiveRegressionForestExtra;

CExtraConfig::CExtraConfig()
{
	__defineAcceptableAttributes();
}

CExtraConfig::~CExtraConfig()
{
}

//****************************************************************************************************
//FUNCTION:
void CExtraConfig::__defineAcceptableAttributes()
{
	//define extra attributes
	defineAttribute(KEY_WORDS::IS_SERIALIZE_MODEL, hiveConfig::ATTRIBUTE_BOOL);
	defineAttribute(KEY_WORDS::SERIALIZATION_MODEL_PATH, hiveConfig::ATTRIBUTE_STRING);
	defineAttribute(KEY_WORDS::PREDICT_RESULT_PATH, hiveConfig::ATTRIBUTE_STRING);
	defineAttribute(KEY_WORDS::STATISTICAL_RESULT_PATH, hiveConfig::ATTRIBUTE_STRING);
}