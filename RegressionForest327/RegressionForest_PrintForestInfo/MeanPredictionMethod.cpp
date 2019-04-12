#include "MeanPredictionMethod.h"
#include "RegressionForestCommon.h"
#include "common/ProductFactory.h"

using namespace hiveRegressionForest;

hiveOO::CProductFactory<CMeanPredictionMethod> theCreator(KEY_WORDS::MEAN_PREDICTION_METHOD);

float CMeanPredictionMethod::predictCertainResponseV(const std::vector<float>& vTestFeatureInstance, float vTestResponse, unsigned int vTreeNumber)
{
	return 0.0f;
}