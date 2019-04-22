#include "TrainingSetConfig.h"
#include "TrainingSetCommon.h"

using namespace hiveRegressionForest;

CTrainingSetConfig::CTrainingSetConfig()
{
	__defineAcceptableAttributes();
}

CTrainingSetConfig::~CTrainingSetConfig()
{
}

//****************************************************************************************************
//FUNCTION:
void CTrainingSetConfig::__defineAcceptableAttributes()
{
	defineAttribute(KEY_WORDS::IS_BINARY_TRAINGSETFILE,		hiveConfig::ATTRIBUTE_BOOL);
	defineAttribute(KEY_WORDS::TRAININGSET_PATH,			hiveConfig::ATTRIBUTE_STRING);
	defineAttribute(KEY_WORDS::NUM_OF_INSTANCE,				hiveConfig::ATTRIBUTE_INT);
	defineAttribute(KEY_WORDS::NUM_OF_FEATURE,				hiveConfig::ATTRIBUTE_INT);
	defineAttribute(KEY_WORDS::NUM_OF_RESPONSE,				hiveConfig::ATTRIBUTE_INT);
	defineAttribute(KEY_WORDS::TESTSET_PATH,				hiveConfig::ATTRIBUTE_STRING);
	defineAttribute(KEY_WORDS::IS_NORMALIZE,				hiveConfig::ATTRIBUTE_BOOL);
	defineAttribute(KEY_WORDS::IS_DIVIDE_FILE,				hiveConfig::ATTRIBUTE_BOOL);
	defineAttribute(KEY_WORDS::START_DIVIDE_POINT,          hiveConfig::ATTRIBUTE_INT);
	defineAttribute(KEY_WORDS::NEW_FILE_DATA_SIZE,			hiveConfig::ATTRIBUTE_INT);
	defineAttribute(KEY_WORDS::GOOD_TEST_FILE,				hiveConfig::ATTRIBUTE_STRING);
	defineAttribute(KEY_WORDS::BAD_TEST_FILE,				hiveConfig::ATTRIBUTE_STRING);
}