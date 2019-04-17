#include "InternalNodePredictionMethod.h"
#include <numeric>
#include "RegressionForestCommon.h"
#include "common/ProductFactory.h"
#include "RegressionForest.h"
#include "PathNodeMethod.h"

using namespace hiveRegressionForest;

hiveOO::CProductFactory<CInternalNodePredictionMethod> theCreator(KEY_WORDS::INTERNAL_NODE_PREDICTION_METHOD);

float CInternalNodePredictionMethod::predictCertainResponseV(const std::vector<float>& vTestFeatureInstance, float vTestResponse, const std::vector<CTree*>& vTreeSet)
{
	_ASSERTE(!vTestFeatureInstance.empty() && !vTreeSet.empty());

	int TreeNumber = vTreeSet.size();
	std::vector<float> PredictValueOfTree(TreeNumber, 0.0f);
	CPathNodeMethod* pPathNodeMethod = CPathNodeMethod::getInstance();

#pragma omp parallel for
	for (int i = 0; i < TreeNumber; ++i)
	{
		PredictValueOfTree[i] = pPathNodeMethod->prediceWithInternalNode(vTreeSet[i], vTestFeatureInstance);
	}
	float PredictValue = std::accumulate(PredictValueOfTree.begin(), PredictValueOfTree.end(), 0.0f);

	return PredictValue / TreeNumber;
}