#include "MPPredictionMethod.h"
#include <numeric>
#include <float.h>
#include "common/HiveCommonMicro.h"
#include "RegressionForestCommon.h"
#include "common/ProductFactory.h"
#include "MpCompute.h"

using namespace hiveRegressionForest;

hiveOO::CProductFactory<CMPPredictionMethod> theCreator(KEY_WORDS::MP_PREDICTION_METHOD);

float CMPPredictionMethod::predictCertainTestV(const std::vector<float>& vTestFeatureInstance, float vTestResponse, const std::vector<CTree*>& vTreeSet)
{
	_ASSERTE(!vTestFeatureInstance.empty() && !vTreeSet.empty());

	int TreeNumber = vTreeSet.size();
	std::vector<float> PredictValueOfTree(TreeNumber, 0.0f);
	std::vector<float> NodeWeight(TreeNumber, 0.0f);
	std::vector<float> MPValue(TreeNumber, 0.0f);
	std::vector<const CNode*> LeafNodeSet(TreeNumber);

	CMpCompute* pMPCompute = nullptr;
#pragma omp parallel for
	for (int i = 0; i < TreeNumber; ++i)
	{
		LeafNodeSet[i] = vTreeSet[i]->locateLeafNode(vTestFeatureInstance);
		PredictValueOfTree[i] = vTreeSet[i]->predict(*LeafNodeSet[i], vTestFeatureInstance, NodeWeight[i]);
		MPValue[i] = 1.0 / (pMPCompute->calMPOutOfFeatureAABB(vTreeSet[i], LeafNodeSet[i], vTestFeatureInstance) + FLT_EPSILON);		
	}
	_SAFE_DELETE(pMPCompute);

	float PredictValue = 0.f;
	for (int i = 0; i < TreeNumber; ++i)
	{
		PredictValue += PredictValueOfTree[i] * MPValue[i];
	}
	float WeightSum = std::accumulate(MPValue.begin(), MPValue.end(), 0.0f);
	return PredictValue / WeightSum;
}