#pragma once
#include <vector>
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
	float SumY = std::accumulate(std::begin(vTestResponseSet), std::end(vTestResponseSet), 0.0);
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
	float Sum = std::accumulate(vData.begin(), vData.end(), 0.f);

	return Sum / vData.size();
}

//****************************************************************************************************
//FUNCTION:
float var(const std::vector<float>& vData)
{
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
	_ASSERTE(vDataA.size() == vDataB.size());
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
	_ASSERTE(vDataA.size() == vDataB.size());
	return (cov(vDataA, vDataB)) / (sqrt(var(vDataA)*var(vDataB)));
}

//****************************************************************************************************
//FUNCTION:
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