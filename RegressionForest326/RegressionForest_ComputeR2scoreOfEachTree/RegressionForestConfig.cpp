#include "RegressionForestConfig.h"
#include "RegressionForestCommon.h"

using namespace hiveRegressionForest;

CRegressionForestConfig::CRegressionForestConfig()
{
	__defineAcceptableAttributes();
}

CRegressionForestConfig::~CRegressionForestConfig()
{
}

//****************************************************************************************************
//FUNCTION:
bool CRegressionForestConfig::isConfigParsed()
{
	return !getInstance()->getName().empty();
}

//****************************************************************************************************
//FUNCTION:
void CRegressionForestConfig::__defineAcceptableAttributes()
{
	//define basic attributes
	defineAttribute(KEY_WORDS::NUMBER_OF_TREE,								hiveConfig::ATTRIBUTE_INT);
	defineAttribute(KEY_WORDS::MAX_TREE_DEPTH,								hiveConfig::ATTRIBUTE_INT);
	defineAttribute(KEY_WORDS::MAX_LEAF_NODE_INSTANCE_SIZE,					hiveConfig::ATTRIBUTE_INT);
	defineAttribute(KEY_WORDS::NUMBER_CANDIDATE_FEATURE,					hiveConfig::ATTRIBUTE_INT);
	defineAttribute(KEY_WORDS::NODE_SPLIT_METHOD,							hiveConfig::ATTRIBUTE_STRING);	
	defineAttribute(KEY_WORDS::LEAF_NODE_MODEL_SIGNATURE,					hiveConfig::ATTRIBUTE_STRING);
	defineAttribute(KEY_WORDS::FEATURE_SELECTOR,							hiveConfig::ATTRIBUTE_STRING);
	defineAttribute(KEY_WORDS::FEATURE_WEIGHT_CALCULATE_METHOD,				hiveConfig::ATTRIBUTE_STRING);
	defineAttribute(KEY_WORDS::LIVE_UPDATE_FEATURES_WEIGHT,					hiveConfig::ATTRIBUTE_BOOL);
	defineAttribute(KEY_WORDS::BOOTSTRAP_SELECTOR,							hiveConfig::ATTRIBUTE_STRING);
	defineAttribute(KEY_WORDS::INSTANCE_WEIGHT_CALCULATE_METHOD,			hiveConfig::ATTRIBUTE_STRING);
	defineAttribute(KEY_WORDS::MAX_MSE_FITTING_THRESHOLD,					hiveConfig::ATTRIBUTE_FLOAT);
	defineAttribute(KEY_WORDS::LEAF_NODE_CONDITION,							hiveConfig::ATTRIBUTE_STRING);
	defineAttribute(KEY_WORDS::OPENMP_PARALLEL_BUILD_TREE,					hiveConfig::ATTRIBUTE_BOOL);
	defineAttribute(KEY_WORDS::BUILD_TREE_TYPE,								hiveConfig::ATTRIBUTE_STRING);
	defineAttribute(KEY_WORDS::CREATE_NODE_TYPE,							hiveConfig::ATTRIBUTE_STRING);
}