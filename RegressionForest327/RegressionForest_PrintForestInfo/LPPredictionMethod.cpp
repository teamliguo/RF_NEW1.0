#include "LPPredictionMethod.h"
#include "RegressionForestCommon.h"
#include "common/ProductFactory.h"

using namespace hiveRegressionForest;

hiveOO::CProductFactory<CLPPredictionMethod> theCreator(KEY_WORDS::LP_PREDICTION_METHOD);

float CLPPredictionMethod::predictCertainResponseV(const std::vector<float>& vTestFeatureInstance, float vTestResponse, unsigned int vTreeNumber)
{
	return 0.0f;
}