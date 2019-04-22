#pragma once
#include <vector>
#include <string>
#include "RegressionForest_EXPORTS.h"

REGRESSION_FOREST_EXPORTS void	calAccuracyByThreshold(const std::vector<float>& vData, const std::vector<float>& vThresholdVec, std::vector<float>& voAccuracyVec);
REGRESSION_FOREST_EXPORTS void	calAccuracyByBiasRate(const std::vector<float>& vData, const std::vector<float>& vBiasRateVec, std::vector<float>& voAccuracyVec);
REGRESSION_FOREST_EXPORTS float calMSE(const std::vector<float>& vDeviateData);
REGRESSION_FOREST_EXPORTS float calR2Score(const std::vector<float>& vTestResponseSet, const std::vector<float>& vPredictSet);
float mean(const std::vector<float>& vData);
float var(const std::vector<float>& vData);
float calSumSquareError(const std::vector<float>& vData);
float cov(const std::vector<float>& vDataA, const std::vector<float>& vDataB);
float samplePearsonCorrelationCoefficient(const std::vector<float>& vDataA, const std::vector<float>&vDataB);
void  transpose(const std::vector<std::vector<float>>& vNativeMatrix, std::vector<std::vector<float>>& voTransposeMatrix);