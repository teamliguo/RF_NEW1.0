#include "UniformBootstrapSelector.h"
#include "RegressionForestCommon.h"
#include "common/ProductFactory.h"
#include "math/RandomInterface.h"

using namespace hiveRegressionForest;

hiveOO::CProductFactory<CUniformBootstrapSelector> theCreator(KEY_WORDS::UNIFORM_BOOTSTRAP_SELECTOR);

CUniformBootstrapSelector::CUniformBootstrapSelector()
{
}

CUniformBootstrapSelector::~CUniformBootstrapSelector()
{
}

//****************************************************************************************************
//FUNCTION:
void CUniformBootstrapSelector::generateBootstrapIndexSetV(unsigned int vInstanceNum, std::vector<int>& voBootstrapIndexSet, const std::vector<float>& vWeightSet)
{
	hiveRandom::hiveGenerateRandomIntegerSet(0, vInstanceNum - 1, vInstanceNum, voBootstrapIndexSet);
}