#include "MPPredictionMethod.h"
#include "RegressionForestCommon.h"
#include "common/ProductFactory.h"

using namespace hiveRegressionForest;

hiveOO::CProductFactory<CMPPredictionMethod> theCreator(KEY_WORDS::MP_PREDICTION_METHOD);

float CMPPredictionMethod::predictCertainResponseV(const std::vector<float>& vTestFeatureInstance, float vTestResponse, unsigned int vTreeNumber)
{
	return 0.0f;
}
