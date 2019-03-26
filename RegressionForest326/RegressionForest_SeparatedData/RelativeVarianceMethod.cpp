#include "RelativeVarianceMethod.h"
#include "common/ProductFactory.h"
#include "math/RandomInterface.h"
#include "RegressionForestCommon.h"
#include "TrainingSet.h"

using namespace hiveRegressionForest;

hiveOO::CProductFactory<CRelativeVarianceMethod> theCreator(KEY_WORDS::RELATIVE_VARIANCE_METHODE);

//****************************************************************************************************
//FUNCTION:
void CRelativeVarianceMethod::__findBestSplitHyperplaneV(std::vector<int>& vBootstrapIndex, const std::pair<int, int>& vBootstrapRange, const std::vector<int>& vFeatureIndexSubset, SSplitHyperplane& voSplitHyperplane)
{
	// NOTES : 这个方法改了统一的接口在这里不好做，而这个方法不常用
	_ASSERTE(!vBootstrapIndex.empty() && !vFeatureIndexSubset.empty());

	int NumInstance = vBootstrapRange.second - vBootstrapRange.first, NumL = 0, NumR = 0;
	float MeanL = 0.0f, MeanR = 0.0f;
	float MaxCurrentFeatureObjVal = -FLT_MAX, CurrentFeatureObjVal = 0.0f;
	float Variance = 0.0f, Mean = 0.0f;

	std::vector<float> ResponseSet(NumInstance);
	int k = 0;
	for (int i = vBootstrapRange.first; i < vBootstrapRange.second; ++i)
	{
		ResponseSet[k++] = CTrainingSet::getInstance()->getResponseAt(vBootstrapIndex[i]);
	}

	__calculateVarianceAndMeanVal(ResponseSet, Variance, Mean);
	if (Variance == 0)
	{
		m_BestGap = CTrainingSet::getInstance()->getFeatureValueAt(0, vFeatureIndexSubset[0]) + CTrainingSet::getInstance()->getFeatureValueAt(1, vFeatureIndexSubset[0]) / 2.0f;
		m_BestSplitFeatureIndex = vFeatureIndexSubset[0];
		return;
	}
	
	std::vector<std::pair<float, float>> FeatureResponseSet;
	for (auto FeatureIndex : vFeatureIndexSubset)
	{
		_generateFeatureResponsePairSet(vBootstrapIndex, vBootstrapRange, FeatureIndex, FeatureResponseSet);
		_sortFeatureResponsePairSetByFeature(FeatureResponseSet);
		for (auto k = 0; k < FeatureResponseSet.size() - 1; ++k)
		{
			MeanL = (MeanL*NumL + FeatureResponseSet[k].second) / (k + 1);
			MeanR = (MeanR*(NumR - k) - FeatureResponseSet[k].second) / (NumR - k - 1);
			++NumL, --NumR;
			float LeftVariable = 0.0, RightVariable = 0.0;
			for (auto n = 0; n <= k; ++n)
			{
				float Distance = FeatureResponseSet[n].second - MeanL;
				LeftVariable += std::pow(Distance, 2.0f);
			}
			LeftVariable /= NumL;
			for (auto n = k + 1; n < NumInstance; ++n)
			{
				float Distance = FeatureResponseSet[n].second - MeanR;
				RightVariable += std::pow(Distance, 2.0f);
			}
			RightVariable /= NumR;

			CurrentFeatureObjVal = (Variance - ((float(NumL) / float(NumInstance)) * LeftVariable) - ((float(NumR) / float(NumInstance)) * RightVariable)) / Variance;
			if (CurrentFeatureObjVal > MaxCurrentFeatureObjVal)
			{
				MaxCurrentFeatureObjVal = CurrentFeatureObjVal;
				m_BestGap = (FeatureResponseSet[k].first + FeatureResponseSet[k + 1].first) / 2.0f;
				m_BestSplitFeatureIndex = FeatureIndex;
			}
		}
	}
}

//****************************************************************************************************
//FUNCTION:
void CRelativeVarianceMethod::__calculateVarianceAndMeanVal(const std::vector<float>& vInput, float& voVariance, float& voMeanVal)
{
	_ASSERTE(vInput.size() > 0);
	voVariance = 0.0;
	voMeanVal = 0.0;
	float Sum = 0.0;

	std::for_each(vInput.begin(), vInput.end(), [&Sum](const float& vItr) { Sum += vItr; });

	if (Sum == 0) voVariance = voMeanVal = 0.0;
	else
	{
		voMeanVal = Sum / vInput.size();
		std::for_each(vInput.begin(), vInput.end(), [&voMeanVal, &voVariance](const float& vElem) {voVariance += std::pow(std::fabs(vElem - voMeanVal), 2); });
	}
	voVariance = voVariance / vInput.size();
}
