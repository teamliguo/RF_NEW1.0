#include "VariancePredictionMethod.h"
#include <numeric>
#include "RegressionForestCommon.h"
#include "common/ProductFactory.h"
#include "Tree.h"
#include "Utility.h"

using namespace hiveRegressionForest;

hiveOO::CProductFactory<CVariancePredictionMethod> theCreator(KEY_WORDS::VARIANCE_PREDICTION_METHOD);

float CVariancePredictionMethod::predictCertainResponseV(const std::vector<float>& vTestFeatureInstance, float vTestResponse, const std::vector<CTree*>& vTreeSet)
{
	_ASSERTE(!vTestFeatureInstance.empty());

	int TreeNum = vTreeSet.size();
	std::vector<float> PredictValueOfTree(TreeNum);
	std::vector<float> TreeResponseVar(TreeNum);
	std::vector<std::vector<float>> TreeFeatureVar(TreeNum), TreeFeatureVarRatio(TreeNum);
	CTrainingSet* pTrainingSet = CTrainingSet::getInstance();

#pragma omp parallel for
	for (auto i = 0; i < TreeNum; i++)
	{
		std::vector<float> TreeResponseSet;
		std::vector<std::vector<float>> TreeFeatureSet;
		std::vector<int> TreeResponseIndexSet;
		const CNode* CurrentLeafNode = vTreeSet[i]->locateLeafNode(vTestFeatureInstance);
		PredictValueOfTree[i] = CurrentLeafNode->getNodeMeanV();
		TreeResponseIndexSet  = CurrentLeafNode->getNodeDataIndex();
		for (auto iter : TreeResponseIndexSet)
		{
			TreeFeatureSet.push_back(pTrainingSet->getFeatureInstanceAt(iter));
			TreeResponseSet.push_back(pTrainingSet->getResponseValueAt(iter));
		}
		TreeResponseVar[i] = var(TreeResponseSet);
		__calVarChangedRatio(TreeFeatureSet, vTestFeatureInstance, TreeFeatureVar[i], TreeFeatureVarRatio[i]);//当前这棵树的各个特征方差，现在存储的是一个16维的条目
	}

	std::vector<float>				WeightByTreeResponseVar		= __calWeightByResponseVar(TreeResponseVar, -1);//元素个数为树的数量
	std::vector<std::vector<float>> WeightByTreeFeatureVar		= __calTreeWeightByFeatureVar(TreeFeatureVar, -1);//若有N颗树，则有N个向量，
	std::vector<std::vector<float>> WeightByTreeFeatureVarRatio = __calTreeWeightByFeatureVar(TreeFeatureVarRatio, -1);
	std::vector<float> FinalWeight = __calFinalWeight(WeightByTreeFeatureVar, WeightByTreeFeatureVarRatio, WeightByTreeResponseVar);
	
	float SumOfWeightPrediction = 0.f;
	for (auto i = 0; i < FinalWeight.size(); i++)
		SumOfWeightPrediction += FinalWeight[i] * PredictValueOfTree[i];
	return SumOfWeightPrediction / std::accumulate(FinalWeight.begin(), FinalWeight.end(), 0.f);
}

//******************************************************************************
//FUNCTION:
void CVariancePredictionMethod::__calVarChangedRatio(const std::vector<std::vector<float>>& vData, const std::vector<float>& vAddValue, std::vector<float>& voNativeVar, std::vector<float>& voChangedRatio)
{
	_ASSERTE(!vData.empty() && !vAddValue.empty());

	int DataNum = vData.size(), FeatureNum = vData[0].size();
	std::vector<float> FeatureSum(FeatureNum, 0.f), FeatureWithNewDataSum(FeatureNum, 0.f);
	voNativeVar.resize(FeatureNum, 0.f);
	voChangedRatio.resize(FeatureNum, 0.f);

	for (auto i = 0; i < FeatureNum; i++)
	{
		for (auto j = 0; j < DataNum; j++)
			FeatureSum[i] += vData[j][i];
		FeatureWithNewDataSum[i] += FeatureSum[i] + vAddValue[i];
	}
	for (auto k = 0; k < FeatureNum; k++)
	{
		for (auto j = 0; j < DataNum; j++)
		{
			voNativeVar[k] += pow(vData[j][k] - FeatureSum[k] / DataNum, 2);
			voChangedRatio[k] += pow(vData[j][k] - FeatureWithNewDataSum[k] / (DataNum + 1), 2);
		}
		voChangedRatio[k] += pow(vAddValue[k] - FeatureWithNewDataSum[k] / (DataNum + 1), 2);
		voNativeVar[k] /= DataNum;
		voChangedRatio[k] = abs(voChangedRatio[k] / (DataNum + 1) - voNativeVar[k]);
	}
}

//******************************************************************************
//FUNCTION:
std::vector<float> CVariancePredictionMethod::__calWeightByResponseVar(const std::vector<float>& vTreeVar, int vParadigmValue)
{
	_ASSERTE(!vTreeVar.empty());

	std::vector<float> TreeWeight;
	float SubstituteOfZero = FLT_MAX;
	for (auto iter : vTreeVar)
		if (iter > FLT_EPSILON && iter < SubstituteOfZero)
			SubstituteOfZero = iter;
	SubstituteOfZero -= FLT_EPSILON;
	for (auto iter : vTreeVar)
	{
		float Temp = (iter > FLT_EPSILON) ? pow(iter, vParadigmValue) : pow(SubstituteOfZero, vParadigmValue);
		TreeWeight.push_back(Temp);
	}
	float SumOfTreeWeight = std::accumulate(TreeWeight.begin(), TreeWeight.end(), 0.f);
	for (auto i = 0; i < TreeWeight.size(); i++)
		TreeWeight[i] /= SumOfTreeWeight;

	return TreeWeight;
}

//******************************************************************************
//FUNCTION:
std::vector<std::vector<float>> CVariancePredictionMethod::__calTreeWeightByFeatureVar(const std::vector<std::vector<float>>& vFeatureVar, int vParadigmValue)
{
	_ASSERTE(!vFeatureVar.empty());

	std::vector<std::vector<float>> WeigthByFeatureVar(vFeatureVar[0].size()), TransWeight;
	std::vector<std::vector<float>> TransposeFeature;
	transpose(vFeatureVar, TransposeFeature);
	for (auto i = 0; i < TransposeFeature.size(); i++)
	{
		WeigthByFeatureVar[i] = __calWeightByResponseVar(TransposeFeature[i], -1);
	}
	transpose(WeigthByFeatureVar, TransWeight);

	return TransWeight;
}

//******************************************************************************
//FUNCTION:
std::vector<float> CVariancePredictionMethod::__calFinalWeight(const std::vector<std::vector<float>>& vWeigthByFeature, const std::vector<std::vector<float>>& vWeightByFeatureWithTest, const std::vector<float>& vWeightByResponse)
{
	_ASSERTE(!vWeightByResponse.empty() && !vWeightByFeatureWithTest.empty() && vWeigthByFeature.empty());

	int DataNum = vWeigthByFeature.size();
	std::vector<float> FinalWeight(DataNum);
	std::vector<float> MeanOfFeatureWeight(DataNum), MeanOfFeatureWithTestVar(DataNum);
	for (auto i = 0; i < DataNum; i++)
	{
		MeanOfFeatureWithTestVar[i] = std::accumulate(vWeightByFeatureWithTest[i].begin(), vWeightByFeatureWithTest[i].end(), 0.f) / vWeightByFeatureWithTest[i].size();
		MeanOfFeatureWeight[i] = std::accumulate(vWeigthByFeature[i].begin(), vWeigthByFeature[i].end(), 0.f) / vWeigthByFeature[i].size();
		FinalWeight[i] = (MeanOfFeatureWithTestVar[i] + MeanOfFeatureWeight[i] + vWeightByResponse[i]) / 3;
	}

	return FinalWeight;
}