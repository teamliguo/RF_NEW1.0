#include "RSSMethod.h"
#include "common/ProductFactory.h"
#include "RegressionForestCommon.h"

using namespace hiveRegressionForest;

hiveOO::CProductFactory<CResidualSumOfSquaresMethod> theCreator(KEY_WORDS::RESIDUAL_SUM_OF_SQUARES_METHOD);

//****************************************************************************************************
//FUNCTION:
void CResidualSumOfSquaresMethod::__findLocalBestSplitHyperplaneV(const std::vector<std::pair<float, float>>& vFeatureResponseSet, float vSum, float& voCurrentFeatureMaxObjVal, float& voCurBestGap)
{
	_ASSERTE(!vFeatureResponseSet.empty());

	int NumL = 0, NumR = vFeatureResponseSet.size();
	float MeanL = 0.0f, MeanR = vSum / vFeatureResponseSet.size();
	for (auto k = 0; k < vFeatureResponseSet.size() - 1; ++k)
	{
		MeanL = (MeanL*NumL + vFeatureResponseSet[k].second) / (k + 1);
		MeanR = (MeanR*NumR - vFeatureResponseSet[k].second) / (NumR - 1);
		NumL++;
		NumR--;

		if (vFeatureResponseSet[k].first < vFeatureResponseSet[k + 1].first)
		{
			float RssL = 0.0f, RssR = 0.0f;
			for (auto n = 0; n <= k; ++n)
				RssL += std::pow(vFeatureResponseSet[n].second - MeanL, 2.0f);

			for (auto n = k + 1; n < vFeatureResponseSet.size(); ++n)
				RssR += std::pow(vFeatureResponseSet[n].second - MeanR, 2.0f);

			float CurFeatureObjVal = -(RssL + RssR);

			if (CurFeatureObjVal > voCurrentFeatureMaxObjVal)
			{
				voCurrentFeatureMaxObjVal = CurFeatureObjVal;
				voCurBestGap = (vFeatureResponseSet[k].first + vFeatureResponseSet[k + 1].first) / 2.0f;
			}
		}
	}
}