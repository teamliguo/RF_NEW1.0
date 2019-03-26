#include "FeatureWeightRSSMethod.h"
#include <numeric>
#include "common/ProductFactory.h"
#include "TrainingSet.h"
#include "RegressionForestCommon.h"

using namespace hiveRegressionForest;

hiveOO::CProductFactory<CRSSFeatureWeightMethod> theCreator(KEY_WORDS::RSS_METHOD);

CRSSFeatureWeightMethod::CRSSFeatureWeightMethod()
{
}

CRSSFeatureWeightMethod::~CRSSFeatureWeightMethod()
{
}

//****************************************************************************************************
//FUNCTION:
void CRSSFeatureWeightMethod::__calculateFeatureWeightV(const std::vector<std::vector<float>>& vInstanceSet, const std::vector<float>& vResponseSet, std::vector<std::pair<unsigned int, float>>& voFeatureWeightSet)
{
	_ASSERTE(!vInstanceSet.empty() && !vResponseSet.empty() && vInstanceSet.size() == vResponseSet.size());

	float SumResponse = 0.0;
	for (unsigned int i = 0; i < vResponseSet.size(); ++i)
		SumResponse += vResponseSet[i];

	unsigned int FeatureNum = CTrainingSet::getInstance()->getNumOfFeatures();
	std::vector<std::pair<float, float>> FeatureResponseSet(vInstanceSet.size());
	for (auto i = 0; i < FeatureNum; ++i)
	{
		FeatureResponseSet.clear();
		__getSortedFeatureResponsePairSet(i, vInstanceSet, vResponseSet, FeatureResponseSet);

		if(FeatureResponseSet[i].first >= FeatureResponseSet[FeatureResponseSet.size()-1].first) continue;
		voFeatureWeightSet.push_back(std::make_pair(i, __calculateMaxObjectFuncValue(FeatureResponseSet, SumResponse)));
	}
	std::sort(voFeatureWeightSet.begin(), voFeatureWeightSet.end(),
		[](const std::pair<unsigned int, float>& vLeft, const std::pair<unsigned int, float>& vRight) { return vLeft.second > vRight.second; });
}

//****************************************************************************************************
//FUNCTION:
void CRSSFeatureWeightMethod::__getSortedFeatureResponsePairSet(unsigned int vFeatureIndex, const std::vector<std::vector<float>>& vInstanceSet, const std::vector<float>& vResponseSet, std::vector<std::pair<float, float>>& voFeatureResponseSet)
{
	voFeatureResponseSet.clear();
	for (auto i = 0; i < vInstanceSet.size(); ++i)
	{
		voFeatureResponseSet.push_back(std::make_pair(vInstanceSet[i][vFeatureIndex], vResponseSet[i]));
	}
	std::sort(voFeatureResponseSet.begin(), voFeatureResponseSet.end());
}

//****************************************************************************************************
//FUNCTION:
float CRSSFeatureWeightMethod::__calculateMaxObjectFuncValue(const std::vector<std::pair<float, float>>& vFeatureResponseSet, float vSumResponse)
{
	float MaxObjectFunctionValue = -FLT_MAX;
	unsigned int NumSamples = vFeatureResponseSet.size();
	float SumL = 0.0, SumR = vSumResponse;
	unsigned int NumL = 0, NumR = NumSamples;
	float CritParent = vSumResponse * vSumResponse / NumSamples;
	for (auto i = 0; i < NumSamples - 1; ++i)
	{
		SumL += vFeatureResponseSet[i].second;
		SumR -= vFeatureResponseSet[i].second;
		NumL++;
		NumR--;

		_ASSERTE(NumL != 0 && NumR != 0);

		if (vFeatureResponseSet[i].first < vFeatureResponseSet[i + 1].first) // NOTES: ! this line is very important !!!!!!!!!
		{
			float CurrentCrit = (SumL * SumL / NumL) + (SumR * SumR / NumR) - CritParent;
			if (CurrentCrit > MaxObjectFunctionValue) 
				MaxObjectFunctionValue = CurrentCrit;
		}
	}
	return (MaxObjectFunctionValue > FLT_MIN) ? MaxObjectFunctionValue : FLT_MIN;
}