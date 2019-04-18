#include "InfoGainMethod.h"
#include "common/ProductFactory.h"
#include "RegressionForestCommon.h"
#include "TrainingSet.h"

using namespace hiveRegressionForest;

hiveOO::CProductFactory<CInfoGainMethod> theCreator(KEY_WORDS::INFORMATION_GAIN_METHOD);

//********************************************************************************************************
//FUNCTION:
void CInfoGainMethod::__findLocalBestSplitHyperplaneV(const std::vector<std::pair<float, float>>& vFeatureResponseSet, float vSum, float& voCurrentFeatureMaxObjVal, float& voCurBestGap)
{
	_ASSERTE(!vFeatureResponseSet.empty());

	float SumL = 0.0f, SumR = vSum;
	int NumL = 0, NumR = vFeatureResponseSet.size();
	float CurrentNodeEntropy = vSum * vSum / vFeatureResponseSet.size();
	for (auto InstanceIndex = 0; InstanceIndex < vFeatureResponseSet.size() - 1; ++InstanceIndex)
	{
		SumL += vFeatureResponseSet[InstanceIndex].second;
		SumR -= vFeatureResponseSet[InstanceIndex].second;
		++NumL;
		--NumR;

		if (vFeatureResponseSet[InstanceIndex].first < vFeatureResponseSet[InstanceIndex + 1].first) // NOTES: ! this line is very important !!!!!!!!!
		{
			float CurrentObjFuncVal = (SumL * SumL / NumL) + (SumR * SumR / NumR) - CurrentNodeEntropy;
			if (CurrentObjFuncVal > voCurrentFeatureMaxObjVal)
			{
				voCurrentFeatureMaxObjVal = CurrentObjFuncVal;
				voCurBestGap = (vFeatureResponseSet[InstanceIndex].first + vFeatureResponseSet[InstanceIndex + 1].first) / 2.0f;
			}
		}
	}
}