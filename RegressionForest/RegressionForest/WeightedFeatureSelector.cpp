#include "WeightedFeatureSelector.h"
#include "RegressionForestCommon.h"
#include "math/RandomInterface.h"
#include "common/ProductFactory.h"

using namespace hiveRegressionForest;

hiveOO::CProductFactory<CWeightedFeatureSelector> theCreator(KEY_WORDS::WEIGHTED_FEATURE_SELECTOR);

CWeightedFeatureSelector::CWeightedFeatureSelector()
{
}

CWeightedFeatureSelector::~CWeightedFeatureSelector()
{
}

//****************************************************************************************************
//FUNCTION:
void CWeightedFeatureSelector::generateFeatureIndexSetV(unsigned int vFeatureNum, std::vector<int>& voFeatureIndexSubset, const std::vector<float>& vWeightSet)
{
	_ASSERTE(!vWeightSet.empty());
	if (m_NumCandidataFeature == 0) m_NumCandidataFeature = vFeatureNum / 3;
	hiveRandom::hiveGenerateSamplingWithoutReplacement(0, vFeatureNum - 1, m_NumCandidataFeature, vWeightSet, voFeatureIndexSubset);
}