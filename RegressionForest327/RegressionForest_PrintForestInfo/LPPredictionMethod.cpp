#include "LPPredictionMethod.h"
#include <numeric>
#include "RegressionForestCommon.h"
#include "common/ProductFactory.h"

using namespace hiveRegressionForest;

hiveOO::CProductFactory<CLPPredictionMethod> theCreator(KEY_WORDS::LP_PREDICTION_METHOD);

float CLPPredictionMethod::predictCertainResponseV(const std::vector<float>& vTestFeatureInstance, float vTestResponse, const std::vector<CTree*>& vTreeSet)
{
	_ASSERTE(!vTestFeatureInstance.empty() && !vTreeSet.empty());

	int TreeNumber = vTreeSet.size();
	std::vector<float> PredictValueOfTree(TreeNumber, 0.0f);
	std::vector<float> NodeWeight(TreeNumber, 0.0f);
	std::vector<float> EuclideanDistance(TreeNumber, 0.0f);
	static std::vector<const CNode*> LeafNodeSet(TreeNumber);
	float PredictValue = 0.f;
	for (int i = 0; i < TreeNumber; ++i)
	{
		LeafNodeSet[i] = vTreeSet[i]->locateLeafNode(vTestFeatureInstance);
		PredictValueOfTree[i] = vTreeSet[i]->predict(*LeafNodeSet[i], vTestFeatureInstance, NodeWeight[i]);
		EuclideanDistance[i] = 1.0 / (__calEuclideanDistance(LeafNodeSet[i]->getFeatureRange(), vTestFeatureInstance) + 1e-6);
		PredictValue += PredictValueOfTree[i] * EuclideanDistance[i];
	}
	float WeightSum = std::accumulate(EuclideanDistance.begin(), EuclideanDistance.end(), 0.0f);
	return PredictValue / WeightSum;
}

//****************************************************************************************************
//FUNCTION:
float CLPPredictionMethod::__calEuclideanDistance(const std::pair<std::vector<float>, std::vector<float>>& vFeatureRange, const std::vector<float>& vTestFeatureInstance)
{
	float EuclideanDistance = 0.f;
	for (int i = 0; i < vTestFeatureInstance.size(); ++i)
	{
		if (vTestFeatureInstance[i] < vFeatureRange.first[i])
			EuclideanDistance += (vFeatureRange.first[i] - vTestFeatureInstance[i])*(vFeatureRange.first[i] - vTestFeatureInstance[i]);
		else if (vTestFeatureInstance[i] > vFeatureRange.second[i])
			EuclideanDistance += (vTestFeatureInstance[i] - vFeatureRange.second[i])*(vTestFeatureInstance[i] - vFeatureRange.second[i]);
	}
	return EuclideanDistance;
}