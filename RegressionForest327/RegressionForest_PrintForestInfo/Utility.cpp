#include "Utility.h"
#include <numeric>

//****************************************************************************************************
//FUNCTION:
void calAccuracyByThreshold(const std::vector<float>& vData, const std::vector<float>& vThresholdVec, std::vector<float>& voAccuracyVec)
{
	_ASSERTE(!vData.empty() && !vThresholdVec.empty());

	unsigned int Num = vData.size();
	unsigned int NumOfThresholds = vThresholdVec.size();
	voAccuracyVec.resize(NumOfThresholds);

	for (auto i = 0; i < Num; i++)
		for (auto k = 0; k < NumOfThresholds; k++)
			if (vData[i] < vThresholdVec[k])
				voAccuracyVec[k]++;

	for (auto i = 0; i < voAccuracyVec.size(); i++)
		voAccuracyVec[i] /= Num;
}

//****************************************************************************************************
//FUNCTION:
void calAccuracyByBiasRate(const std::vector<float>& vData, const std::vector<float>& vBiasRateVec, std::vector<float>& voAccuracyVec)
{
	_ASSERTE(!vData.empty() && !vBiasRateVec.empty());

	unsigned Num = vData.size();
	unsigned int NumOfThresholds = vBiasRateVec.size();
	voAccuracyVec.resize(NumOfThresholds);

	for (auto i = 0; i < Num; i++)
		for (auto k = 0; k < NumOfThresholds; k++)
			if (vData[i] < vBiasRateVec[k])
				voAccuracyVec[k]++;

	for (auto i = 0; i < voAccuracyVec.size(); i++)
		voAccuracyVec[i] /= Num;
}

//****************************************************************************************************
//FUNCTION:
float calMSE(const std::vector<float>& vDeviateData)
{
	_ASSERTE(!vDeviateData.empty());

	unsigned int Num = vDeviateData.size();
	float MSE = 0.0f;

	for (unsigned int i = 0; i < Num; i++)
		MSE += pow(vDeviateData[i], 2);

	return MSE / Num;
}

//****************************************************************************************************
//FUNCTION:
float calR2Score(const std::vector<float>& vTestResponseSet, const std::vector<float>& vPredictSet)
{
	_ASSERTE(!vTestResponseSet.empty() && !vPredictSet.empty());

	float SumY = std::accumulate(std::begin(vTestResponseSet), std::end(vTestResponseSet), 0.0f);
	float MeanY = SumY / vTestResponseSet.size();

	float Numerator = 0.0f;
	float Denumerator = 0.0f;
	for (auto i = 0; i < vTestResponseSet.size(); i++)
	{
		Numerator += pow(vTestResponseSet[i] - vPredictSet[i], 2);
		Denumerator += pow(vTestResponseSet[i] - MeanY, 2);
	}
	_ASSERTE(Denumerator != 0);
	float R2Store = 1 - Numerator / Denumerator;
	return R2Store;
}


//****************************************************************************************************
//FUNCTION:
float mean(const std::vector<float>& vData)
{
	_ASSERTE(!vData.empty());
	float Sum = std::accumulate(vData.begin(), vData.end(), 0.f);
	return Sum / vData.size();
}

//****************************************************************************************************
//FUNCTION:
float var(const std::vector<float>& vData)
{
	_ASSERTE(!vData.empty());

	float Sum = 0.f;
	float Mean = mean(vData);
	for (int i = 0; i < vData.size(); i++)
	{
		Sum += (vData[i] - Mean)*(vData[i] - Mean);
	}

	return Sum / vData.size();
}

//******************************************************************************
//FUNCTION:
float calSumSquareError(const std::vector<float>& vData)
{
	_ASSERTE(!vData.empty());

	float Sum = 0.f;
	float Mean = mean(vData);
	for (int i = 0; i < vData.size(); i++)
	{
		Sum += (vData[i] - Mean)*(vData[i] - Mean);
	}

	return Sum;
}

//****************************************************************************************************
//FUNCTION:
float cov(const std::vector<float>& vDataA, const std::vector<float>& vDataB)
{
	_ASSERTE(!vDataA.empty() && vDataA.size() == vDataB.size());

	float Sum = 0.0;
	float MeanA = mean(vDataA);
	float MeanB = mean(vDataB);
	for (int i = 0; i < vDataA.size(); i++)
	{
		Sum += (vDataA[i] - MeanA)*(vDataB[i] - MeanB);
	}

	return Sum;
}

//****************************************************************************************************
//FUNCTION:
float samplePearsonCorrelationCoefficient(const std::vector<float>& vDataA, const std::vector<float>&vDataB)
{
	_ASSERTE(!vDataA.empty() && vDataA.size() == vDataB.size());
	return (cov(vDataA, vDataB)) / (sqrt(calSumSquareError(vDataA)*calSumSquareError(vDataB)));
}

//******************************************************************************
//FUNCTION:
void transpose(const std::vector<std::vector<float>>& vNativeMatrix, std::vector<std::vector<float>>& voTransposeMatrix)
{
	_ASSERTE(!vNativeMatrix.empty());
	voTransposeMatrix.resize(vNativeMatrix[0].size());
	for (auto i = 0; i < vNativeMatrix[0].size(); i++)
		for (auto j = 0; j < vNativeMatrix.size(); j++)
			voTransposeMatrix[i].push_back(vNativeMatrix[j][i]);
}