#include "InternalNodePredictionMethod.h"
#include "RegressionForestCommon.h"
#include "common/ProductFactory.h"

using namespace hiveRegressionForest;

hiveOO::CProductFactory<CInternalNodePredictionMethod> theCreator(KEY_WORDS::INTERNAL_NODE_PREDICTION_METHOD); 

float CInternalNodePredictionMethod::predictCertainResponseV(const std::vector<float>& vTestFeatureInstance, float vTestResponse, unsigned int vTreeNumber)
{
	return 0.0f;
}
