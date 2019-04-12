#include "VariancePredictionMethod.h"
#include "RegressionForestCommon.h"
#include "common/ProductFactory.h"

using namespace hiveRegressionForest;

hiveOO::CProductFactory<CVariancePredictionMethod> theCreator(KEY_WORDS::VARIANCE_PREDICTION_METHOD);

float CVariancePredictionMethod::predictCertainResponseV(const std::vector<float>& vTestFeatureInstance, float vTestResponse, unsigned int vTreeNumber)
{
	return 0.0f;
}
