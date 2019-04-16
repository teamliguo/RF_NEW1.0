#pragma once
#include <vector>
#include "RegressionForest_EXPORTS.h"

REGRESSION_FOREST_EXPORTS void calAccuracyByThreshold(const std::vector<float>& vData, const std::vector<float>& vThresholdVec, std::vector<float>& voAccuracyVec);
REGRESSION_FOREST_EXPORTS void calAccuracyByBiasRate(const std::vector<float>& vData, const std::vector<float>& vBiasRateVec, std::vector<float>& voAccuracyVec);
REGRESSION_FOREST_EXPORTS float calMSE(const std::vector<float>& vDeviateData);
REGRESSION_FOREST_EXPORTS float calR2Score(const std::vector<float>& vTestResponseSet, const std::vector<float>& vPredictSet);
float mean(const std::vector<float>& vData);
float var(const std::vector<float>& vData);
float calSumSquareError(const std::vector<float>& vData);
float cov(const std::vector<float>& vDataA, const std::vector<float>& vDataB);
float samplePearsonCorrelationCoefficient(const std::vector<float>& vDataA, const std::vector<float>&vDataB);
void transpose(const std::vector<std::vector<float>>& vNativeMatrix, std::vector<std::vector<float>>& voTransposeMatrix);

template<class T1, class T2>
std::vector<T2> calSecondParRange(T1 vMin, T1 vMax, const std::vector<std::pair<T1, T2>>& vData)
{
	std::vector<T2> SecondPar;
	int MinIndex = 0, MaxIndex = 0;
	for (int k = 0; k < vData.size(); k++)
	{
		if (vData[k].first >= vMin)
		{
			MinIndex = k;
			MaxIndex = k;
			break;
		}
	}
	for (int k = MinIndex; k < vData.size(); k++)
	{
		if (vData[k].first <= vMax)
			MaxIndex = k;
		else break;
	}
	for (int k = MinIndex; k <= MaxIndex; k++)
		SecondPar.push_back(vData[k].second);

	return SecondPar;
}