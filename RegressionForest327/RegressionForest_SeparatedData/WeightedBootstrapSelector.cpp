#include "WeightedBootstrapSelector.h"
#include "RegressionForestCommon.h"
#include "TrainingSet.h"
#include "common/ProductFactory.h"
#include "math/RandomInterface.h"

using namespace hiveRegressionForest;

hiveOO::CProductFactory<CWeightedBootstrapSelector> theCreator(KEY_WORDS::WEIGHTED_BOOTSTRAP_SELECTOR);

CWeightedBootstrapSelector::CWeightedBootstrapSelector()
{
}

CWeightedBootstrapSelector::~CWeightedBootstrapSelector()
{
}

//****************************************************************************************************
//FUNCTION:
void CWeightedBootstrapSelector::generateBootstrapIndexSetV(unsigned int vInstanceNum, std::vector<int>& voBootstrapIndexSet, const std::vector<float>& vWeightSet)
{
	hiveRandom::hiveGenerateRandomIntegerSet(0, vInstanceNum - 1, vInstanceNum, vWeightSet, voBootstrapIndexSet);
}