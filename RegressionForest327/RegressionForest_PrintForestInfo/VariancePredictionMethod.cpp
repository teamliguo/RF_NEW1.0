#include <numeric>
#include "VariancePredictionMethod.h"
#include "RegressionForestCommon.h"
#include "common/ProductFactory.h"
#include "Tree.h"
#include "Utility.h"

using namespace hiveRegressionForest;

hiveOO::CProductFactory<CVariancePredictionMethod> theCreator(KEY_WORDS::VARIANCE_PREDICTION_METHOD);//问题：要不要写成私有函数？

float CVariancePredictionMethod::predictCertainResponseV(const std::vector<float>& vTestFeatureInstance, float vTestResponse, const std::vector<CTree*>& vTreeSet)
{
	_ASSERTE(!vTestFeatureInstance.empty());
	std::vector<float> PredictValueOfTree(vTreeSet.size());
	std::vector<float> TreeResponseVar(vTreeSet.size());
	std::vector<std::vector<float>> TreeFeatureVar(vTreeSet.size()), TreeFeatureVarRatio(vTreeSet.size());
	CTrainingSet* pTrainingSet = CTrainingSet::getInstance();
#pragma omp parallel for
	for (auto i = 0; i < vTreeSet.size(); i++)
	{
		std::vector<float> TreeResponseSet;
		std::vector<std::vector<float>> TreeFeatureSet;
		std::vector<int> TreeResponseIndexSet, TreeFeatureIndexSet;
		const CNode* CurrentLeafNode = vTreeSet[i]->locateLeafNode(vTestFeatureInstance);
		PredictValueOfTree[i] = CurrentLeafNode->getNodeMeanV();
		TreeResponseIndexSet = CurrentLeafNode->getNodeDataIndex();
		for (auto iter : TreeResponseIndexSet)
		{
			TreeFeatureSet.push_back(pTrainingSet->getFeatureInstanceAt(iter));
			TreeResponseSet.push_back(pTrainingSet->getResponseValueAt(iter));
		}
		TreeResponseVar[i] = var(TreeResponseSet);
		__calVarChangedRatio(TreeFeatureSet, vTestFeatureInstance, TreeFeatureVar[i], TreeFeatureVarRatio[i]);//当前这棵树的各个特征方差，现在存储的是一个16维的条目
	}
	std::vector<float> WeightByTreeResponseVar = __calWeightByResponseVar(TreeResponseVar, -1);//元素个数为树的数量
	std::vector<std::vector<float>> WeightByTreeFeatureVar = __calTreeWeightByFeatureVar(TreeFeatureVar, -1);//若有N颗树，则有N个向量，
	std::vector<std::vector<float>> WeightByTreeFeatureVarRatio = __calTreeWeightByFeatureVar(TreeFeatureVarRatio, -1);
	std::vector<float> FinalWeight = __calFinalWeight(WeightByTreeFeatureVar, WeightByTreeFeatureVarRatio, TreeResponseVar);
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
	std::vector<float> FeatureSum(vData[0].size(), 0.f), FeatureWithNewDataSum(vData[0].size(), 0.f);//初始化，改i，k
	voNativeVar.resize(vData[0].size(), 0.f);
	voChangedRatio.resize(vData[0].size(), 0.f);
	for (auto i = 0; i < vData[0].size(); i++)
	{
		for (auto j = 0; j < vData.size(); j++)
			FeatureSum[i] += vData[j][i];
		FeatureWithNewDataSum[i] += FeatureSum[i] + vAddValue[i];
	}
	for (auto k = 0; k < vData[0].size(); k++)
	{
		for (auto j = 0; j < vData.size(); j++)
		{
			voNativeVar[k] += pow(vData[j][k] - FeatureSum[k] / vData.size(), 2);
			voChangedRatio[k] += pow(vData[j][k] - FeatureWithNewDataSum[k] / (vData.size() + 1), 2);
		}
		voChangedRatio[k] += pow(vAddValue[k] - FeatureWithNewDataSum[k] / (vData.size() + 1), 2);
		voNativeVar[k] /= vData.size();
		voChangedRatio[k] = abs(voChangedRatio[k] / (vData.size() + 1) - voNativeVar[k]);
	}
}

//******************************************************************************
//FUNCTION:
std::vector<float> CVariancePredictionMethod::__calWeightByResponseVar(const std::vector<float>& vTreeVar, int vParadigmValue)
{
	_ASSERTE(!vTreeVar.empty());
	std::vector<float> TreeWeight;
	float SumOfTreeWeight = 0.0f, SubstituteOfZero = FLT_MAX;
	for (auto iter : vTreeVar)
		SubstituteOfZero = (iter < SubstituteOfZero && iter != 0) ? iter : SubstituteOfZero;
	SubstituteOfZero -= FLT_EPSILON;
	for (auto iter : vTreeVar)
	{
		float Temp = (iter != 0) ? pow(iter, vParadigmValue) : pow(SubstituteOfZero, vParadigmValue);
		TreeWeight.push_back(Temp);
	}
	SumOfTreeWeight = std::accumulate(TreeWeight.begin(), TreeWeight.end(), 0.f);
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
	for (auto i = 0; i < TransposeFeature.size(); i++)//第一个维度
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
	std::vector<float> FinalWeight(vWeigthByFeature.size());
	std::vector<float> MeanOfFeatureWeight(vWeigthByFeature.size()), MeanOfFeatureWithTestVar(vWeigthByFeature.size());
	for (auto i = 0; i < vWeigthByFeature.size(); i++)
	{
		MeanOfFeatureWithTestVar[i] = std::accumulate(vWeightByFeatureWithTest[i].begin(), vWeightByFeatureWithTest[i].end(), 0.f) / vWeightByFeatureWithTest[i].size();
		MeanOfFeatureWeight[i] = std::accumulate(vWeigthByFeature[i].begin(), vWeigthByFeature[i].end(), 0.f) / vWeigthByFeature[i].size();
		FinalWeight[i] = (MeanOfFeatureWithTestVar[i] + MeanOfFeatureWeight[i] + vWeightByResponse[i]) / 3;
	}
	return FinalWeight;
}
