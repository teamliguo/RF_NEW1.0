#include "UniformFeatureSelector.h"
#include "RegressionForestCommon.h"
#include "math/RandomInterface.h"
#include "common/ProductFactory.h"

using namespace hiveRegressionForest;

hiveOO::CProductFactory<CUniformFeatureSelector> theCreator(KEY_WORDS::UNIFORM_FEATURE_SELECTOR);

CUniformFeatureSelector::CUniformFeatureSelector()
{
}

CUniformFeatureSelector::~CUniformFeatureSelector()
{
}

//****************************************************************************************************
//FUNCTION:
void CUniformFeatureSelector::generateFeatureIndexSetV(unsigned int vFeatureNum, std::vector<int>& voFeatureIndexSubset, const std::vector<float>& vWeightSet)
{
	//_ASSERTE(!vWeightSet.empty());
	if (m_NumCandidataFeature == 0) m_NumCandidataFeature = vFeatureNum / 3;
	hiveRandom::hiveGenerateSamplingWithoutReplacement(0, vFeatureNum - 1, m_NumCandidataFeature, voFeatureIndexSubset);
}