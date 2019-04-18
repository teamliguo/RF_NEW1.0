#include "EarlyFittingCondition.h"
#include "RegressionForestCommon.h"
#include "RegressionForestConfig.h"
#include "TrainingSet.h"
#include "common/ProductFactory.h"
#include "common/HiveCommonMicro.h"
#include "math/RegressionAnalysisInterface.h"
#include "math/BaseRegression.h"

using namespace hiveRegressionForest;

hiveOO::CProductFactory<CEarlyFittingCondition> theCreator(KEY_WORDS::EARLY_FITTING_CONDITION);

CEarlyFittingCondition::CEarlyFittingCondition()
{
}

CEarlyFittingCondition::~CEarlyFittingCondition()
{
}

//****************************************************************************************************
//FUNCTION:
bool CEarlyFittingCondition::isMeetTerminateConditionV(const std::vector<std::vector<float>>& vFeatureSet, const std::vector<float>& vResponseSet, boost::any vExtra)
{
	unsigned int NodeLevel = boost::any_cast<unsigned int>(vExtra);

	bool IsFittingTerminateBegin = (vFeatureSet.size() <= 2 * CTrainingSet::getInstance()->getNumOfFeatures());
	float MeanSquareError = FLT_MAX;
	if (IsFittingTerminateBegin)
		MeanSquareError = __calculateMeanSquaredError(vFeatureSet, vResponseSet);

	unsigned int MaxTreeDepth = CRegressionForestConfig::getInstance()->getAttribute<int>(KEY_WORDS::MAX_TREE_DEPTH);
	unsigned int MaxLeafNodeSamples = CRegressionForestConfig::getInstance()->getAttribute<int>(KEY_WORDS::MAX_LEAF_NODE_INSTANCE_SIZE);
	float MaxFittingMeanSquareError = CRegressionForestConfig::getInstance()->getAttribute<float>(KEY_WORDS::MAX_MSE_FITTING_THRESHOLD);

	return (NodeLevel >= MaxTreeDepth || vFeatureSet.size() <= MaxLeafNodeSamples || MeanSquareError < MaxFittingMeanSquareError);
}

//****************************************************************************************************
//FUNCTION:
float CEarlyFittingCondition::__calculateMeanSquaredError(const std::vector<std::vector<float>>& vFeatureSet, const std::vector<float>& vResponseSet)
{
	std::string LeafNodeModelSig = CRegressionForestConfig::getInstance()->getAttribute<std::string>(KEY_WORDS::LEAF_NODE_MODEL_SIGNATURE);
	if (LeafNodeModelSig.empty()) LeafNodeModelSig = KEY_WORDS::REGRESSION_MODEL_LEAST_SQUARES;

 	hiveRegressionAnalysis::IBaseRegression* pRegressionModel = hiveRegressionAnalysis::hiveTrainRegressionModel(vFeatureSet, vResponseSet, LeafNodeModelSig);
	float MeanSquareError = 0.0;
 	for (auto i = 0; i < vFeatureSet.size(); ++i)
 		MeanSquareError += pow((hiveRegressionAnalysis::hiveExecuteRegression(pRegressionModel, vFeatureSet[i]) - vResponseSet[i]), 2);
	_SAFE_DELETE(pRegressionModel);

	return MeanSquareError / vFeatureSet.size();
}