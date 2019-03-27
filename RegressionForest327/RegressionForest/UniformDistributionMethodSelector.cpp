#include "UniformDistributionMethodSelector.h"
#include "math/RandomInterface.h"
#include "common/ProductFactory.h"
#include "RegressionForestCommon.h"

using namespace hiveRegressionForest;

hiveOO::CProductFactory<CUniformDistributionMethod> TheCreator(KEY_WORDS::UNIFORM_DISTRIBUTION_METHOD);

CUniformDistributionMethod::CUniformDistributionMethod()
{
}

CUniformDistributionMethod::~CUniformDistributionMethod()
{
}

//****************************************************************************************************
//FUNCTION:
void CUniformDistributionMethod::generateFeatureIndexSubsetV(unsigned int vFeatureNum, std::vector<int>& voFeatureIndexSubset, const std::vector<float>& vWeightSet)
{
	hiveRandom::hiveGenerateSamplingWithoutReplacement(0, vFeatureNum - 1, vFeatureNum / 3, voFeatureIndexSubset);
}

//****************************************************************************************************
//FUNCTION:
void CUniformDistributionMethod::generateBootstrapIndexSetV(unsigned int vInstanceNum, std::vector<int>& voBootstrapSet, const std::vector<float>& vWeightSet)
{
	hiveRandom::hiveGenerateRandomIntegerSet(0, vInstanceNum - 1, vInstanceNum, voBootstrapSet);
}