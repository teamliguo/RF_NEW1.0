#include "TrainingSet.h"
#include <fstream>
#include <string>
#include <numeric>
#include <algorithm>
#include <boost/algorithm/string.hpp>
#include "common/CommonInterface.h"
#include "common/ConfigParser.h"
#include "common/HiveCommonMicro.h"
#include "common/OOInterface.h"
#include "common/ProductFactoryData.h"
#include "TrainingSetConfig.h"
#include "TrainingSetCommon.h"
#include "BaseBootstrapSelector.h"
#include "BaseInstanceWeightMethod.h"

using namespace hiveRegressionForest;

CTrainingSet::CTrainingSet()
{
}

CTrainingSet::~CTrainingSet()
{
	if(m_pResponseSet) _SAFE_DELETE_ARRAY(m_pResponseSet);
}

//******************************************************************************
//FUNCTION:
bool CTrainingSet::comparePair(const std::pair<float, int>& vFirst, const std::pair<float, int>& vSecond)
{
	return (vFirst.first < vSecond.first) ? true : false;
}

//****************************************************************************************************
//FUNCTION:
bool CTrainingSet::loadTrainingSet(const std::string& vConfig, bool vHeader)
{
	__initTrainingSetConfig(vConfig);
	_ASSERTE(m_pResponseSet && !m_FeatureSet.empty());
	
	bool IsBinaryFile = CTrainingSetConfig::getInstance()->getAttribute<bool>(hiveRegressionForest::KEY_WORDS::IS_BINARY_TRAINGSETFILE);
	std::string TrainingSetFile = CTrainingSetConfig::getInstance()->getAttribute<std::string>(hiveRegressionForest::KEY_WORDS::TRAININGSET_PATH);
	
	bool IsLoadDataSuccess = IsBinaryFile ? __loadSetFromBinaryFile(TrainingSetFile) : __loadSetFromCSVFile(TrainingSetFile, vHeader);

	bool IsNormalize = false;
	if (CTrainingSetConfig::getInstance()->isAttributeExisted(hiveRegressionForest::KEY_WORDS::IS_NORMALIZE))
		IsNormalize = CTrainingSetConfig::getInstance()->getAttribute<bool>(hiveRegressionForest::KEY_WORDS::IS_NORMALIZE);
	if (IsNormalize)
	{
		normalization(m_FeatureSet);
	}
	
	__calStandardDeviation(m_FeatureSet, m_EachDimStandard);//Add_ljy_12/20
	__calFeatureRange();
	__calResponseRange();
	return IsLoadDataSuccess;
}

//****************************************************************************************************
//FUNCTION:
void CTrainingSet::recombineBootstrapDataset(const std::vector<int>& vBootstrapIndexSet, const std::pair<int, int>& vBootstrapIndexRange, std::pair<std::vector<std::vector<float>>, std::vector<float>>& voFeatureResponseSet)
{
	_ASSERTE(!vBootstrapIndexSet.empty());

	int Range = vBootstrapIndexRange.second - vBootstrapIndexRange.first;
	_ASSERTE(Range > 0);	
	voFeatureResponseSet.first.resize(Range);
	voFeatureResponseSet.second.resize(Range * m_NumResponse);

	int ResponseIndex = 0;
	while (ResponseIndex < m_NumResponse)
	{
		int n = ResponseIndex * Range;
		for (auto i = vBootstrapIndexRange.first; i < vBootstrapIndexRange.second; ++i)
		{
			if (ResponseIndex == 0) voFeatureResponseSet.first[n] = getFeatureInstanceAt(vBootstrapIndexSet[i]);
			voFeatureResponseSet.second[n] = getResponseValueAt(vBootstrapIndexSet[i], ResponseIndex);
			++n;
		}
		++ResponseIndex;
	}
}

//******************************************************************************
//FUNCTION:根据范围输出总表的下标
void CTrainingSet::recombineBootstrapDataset(const std::vector<int>& vBootstrapIndexSet, const std::pair<int, int>& vBootstrapIndexRange, std::vector<int>& voRangeIndex)
{
	_ASSERTE(!vBootstrapIndexSet.empty());

	int Range = vBootstrapIndexRange.second - vBootstrapIndexRange.first;
	_ASSERTE(Range > 0);
	for (auto i = vBootstrapIndexRange.first; i < vBootstrapIndexRange.second; ++i)
		voRangeIndex.push_back(vBootstrapIndexSet[i]);
}

//****************************************************************************************************
//FUNCTION:
void CTrainingSet::dumpFeatureValueSetAt(const std::vector<int>& vInstanceIndexSet, unsigned int vFeatureIndex, std::vector<float>& voValueSet)
{
	_ASSERTE(!vInstanceIndexSet.empty());

	voValueSet.resize(vInstanceIndexSet.size());
	for (unsigned int i = 0; i < vInstanceIndexSet.size(); ++i)
		voValueSet[i] = getFeatureValueAt(vInstanceIndexSet[i], vFeatureIndex);
}

//******************************************************************************
//FUNCTION:计算标准差
void CTrainingSet::__calStandardDeviation(const std::vector<std::vector<float>>& vFeatureDataSet, std::vector<float>& voEachDimStandard)
{
	_ASSERTE(!vFeatureDataSet.empty());
	std::vector<std::vector<float>> DimFeature;
	calDimFeatures(vFeatureDataSet, DimFeature);
	std::vector<float> StandardDeviationSum(DimFeature.size(), 0.f);
	for (int i = 0; i < DimFeature.size(); i++)
	{
		float Average = accumulate(DimFeature[i].begin(), DimFeature[i].end(), 0.f) / DimFeature[0].size();
		for (int j = 0; j < DimFeature[0].size(); j++)
		{
			StandardDeviationSum[i] += pow(DimFeature[i][j] - Average, 2.0f);
		}
		voEachDimStandard.push_back(pow(StandardDeviationSum[i] / DimFeature[0].size(), 0.5f));
	}
}

//****************************************************************************************************
//FUNCTION:
void CTrainingSet::__initTrainingSetConfig(const std::string& vConfig)
{
	_ASSERTE(!vConfig.empty());

	bool IsConfigParsed = hiveConfig::hiveParseConfig(vConfig, hiveConfig::EConfigType::XML, CTrainingSetConfig::getInstance());
	_ASSERTE(IsConfigParsed);

	const CTrainingSetConfig* pTrainingSetConfig = CTrainingSetConfig::getInstance();
	unsigned int NumInstance = pTrainingSetConfig->getAttribute<int>(hiveRegressionForest::KEY_WORDS::NUM_OF_INSTANCE);
	unsigned int NumFeature  = pTrainingSetConfig->getAttribute<int>(hiveRegressionForest::KEY_WORDS::NUM_OF_FEATURE);
	m_NumResponse = pTrainingSetConfig->getAttribute<int>(hiveRegressionForest::KEY_WORDS::NUM_OF_RESPONSE);

	m_FeatureSet.resize(NumInstance);
	for (auto& Itr : m_FeatureSet) Itr.resize(NumFeature);

	m_pResponseSet = new float[NumInstance * m_NumResponse];
}

//*********************************************************************
//FUNCTION:
bool CTrainingSet::__loadSetFromCSVFile(const std::string& vCSVFile, bool vHeader)
{
	_ASSERTE(!vCSVFile.empty());

	std::ifstream DataFile(vCSVFile);
	if (DataFile.is_open())
	{
		std::string Line;

		if (vHeader)
		{
			getline(DataFile, Line);
			std::cout << "Ignore header of [" + vCSVFile + "] by default.\n";
		}

		std::vector<std::string> InstanceString;
		for (int i = 0; i < m_FeatureSet.size(); ++i)
		{
			getline(DataFile, Line);
			InstanceString.clear();
			boost::split(InstanceString, Line, boost::is_any_of(", "));
			_ASSERTE(!InstanceString.empty());

			if (!m_FeatureSet.empty() && (InstanceString.size() != (m_FeatureSet[0].size() + m_NumResponse)))
			{
				DataFile.close();
				m_FeatureSet.clear();
				_SAFE_DELETE_ARRAY(m_pResponseSet);

				return false;
			}

			int InstanceIndex = 0;
			m_FeatureSet[i].resize(m_FeatureSet[0].size());
			for (InstanceIndex = 0; InstanceIndex < m_FeatureSet[0].size(); ++InstanceIndex)
				m_FeatureSet[i][InstanceIndex] = std::stof(InstanceString[InstanceIndex].c_str());

			for (int m = 0; m < m_NumResponse; ++m)
				m_pResponseSet[i*m_NumResponse + m] = std::stof(InstanceString[InstanceIndex + m]);
		}
		DataFile.close();

		return true;
	}
	else
	{
		std::cout << "Failed to open file [" + vCSVFile + "], check file path.\n";

		return false;
	}
}

//****************************************************************************************************
//FUNCTION:
void CTrainingSet::__calResponseRange()
{
	float MaxResponse = (std::numeric_limits<float>::min)();
	float MinResponse = (std::numeric_limits<float>::max)();
	for (int i = 0; i < m_FeatureSet.size(); ++i)
	{
		for (int m = 0; m < m_NumResponse; ++m)
		{
			MaxResponse = (m_pResponseSet[i*m_NumResponse + m] > MaxResponse) ? m_pResponseSet[i*m_NumResponse + m] : MaxResponse;
			MinResponse = (m_pResponseSet[i*m_NumResponse + m] < MinResponse) ? m_pResponseSet[i*m_NumResponse + m] : MinResponse;
		}
	}
	m_ResponseRange = std::make_pair(MinResponse, MaxResponse);
}

//****************************************************************************************************
//FUNCTION:
void CTrainingSet::__calFeatureRange()
{
	std::vector<float> MinFeature, MaxFeature;
	for (int i = 0; i < m_FeatureSet[0].size(); i++)
	{
		std::vector<float> Column(m_FeatureSet.size());
		for (int k = 0; k < m_FeatureSet.size(); k++)
			Column[k] = m_FeatureSet[k][i];
		float max = *std::max_element(Column.begin(), Column.end());
		float min = *std::min_element(Column.begin(), Column.end());
		MinFeature.push_back(min);
		MaxFeature.push_back(max);
	}
	m_FeatureRange = std::make_pair(MinFeature, MaxFeature);
}

//****************************************************************************************************
//FUNCTION:
float CTrainingSet::__calMPValue(const std::vector<float>& vLeafDate, const std::vector<float>& vTestData, float vPredictResponse)
{
	float MPParam = 2.0f, MPValueSum = 0.f;
	std::vector<std::pair<float, float>> MaxMinValue;
	std::vector<int> EachFeatureIntervalCount(vLeafDate.size(), 0);
	std::vector<std::pair<float, float>> ResponseRange(vLeafDate.size(), { vPredictResponse, vPredictResponse });

	for (int j = 0; j < vLeafDate.size(); j++)
	{
		float StandardDeviation = getEachDimStandard()[j];
		MaxMinValue.push_back({ std::max(vLeafDate[j], vTestData[j]) + StandardDeviation, std::min(vLeafDate[j], vTestData[j]) - StandardDeviation });
	} 

	__countIntervalNode(MaxMinValue, EachFeatureIntervalCount, ResponseRange);

	for (int i = 0; i < EachFeatureIntervalCount.size(); i++)
	{
		MPValueSum += pow(((float)EachFeatureIntervalCount[i] / (float)getNumOfInstances()) /** MPParam*/ /**((ResponseRange[i].first - ResponseRange[i].second) / m_ResponseRange)*/, MPParam);
	}
	return pow(MPValueSum, 1 / MPParam);
}

//******************************************************************************
//FUNCTION:
void CTrainingSet::calDimFeatures(const std::vector<std::vector<float>>& vFeatureDataSet, std::vector<std::vector<float>>& voDimFeatureDataSet)
{
	voDimFeatureDataSet.resize(vFeatureDataSet[0].size());
	for (int i = 0; i < vFeatureDataSet[0].size(); i++)
		for (int j = 0; j < vFeatureDataSet.size(); j++)
			voDimFeatureDataSet[i].push_back(vFeatureDataSet[j][i]);
}

//*********************************************************************
//FUNCTION:
bool CTrainingSet::__loadSetFromBinaryFile(const std::string& vBinaryFile)
{
	std::ifstream BinaryDataFile(vBinaryFile, std::ios::binary | std::ios::in);

	unsigned int InstanceSize = m_FeatureSet[0].size() + m_NumResponse;
	float *TempData = new float[m_FeatureSet.size()*InstanceSize];
	_ASSERTE(TempData);

	if (BinaryDataFile.is_open())
	{
		BinaryDataFile.read((char*)TempData, sizeof(float)*m_FeatureSet.size()*InstanceSize);
		for (int i = 0; i < m_FeatureSet.size(); ++i)
		{
			for (int k = 0; k < InstanceSize; k++)
			{
				if (k < m_FeatureSet[0].size())
					m_FeatureSet[i][k] = TempData[i*InstanceSize + k];
				else
					m_pResponseSet[i*m_NumResponse+(k - m_FeatureSet[0].size())] = TempData[i*InstanceSize + k];
			}
		}

		_SAFE_DELETE_ARRAY(TempData);
		BinaryDataFile.close();

		return true;
	}
	else
	{
		std::cout << "Failed to open file [" + vBinaryFile + "], check file path.\n";

		return false;
	}
}

//****************************************************************************************************
//FUNCTION:
void CTrainingSet::normalization(std::vector<std::vector<float>>& voFeatureSet)
{
	_ASSERT(voFeatureSet.size() > 0);
	if (m_FeatureMean.size() == 0 && m_FeatureStd.size() == 0)
	{
		m_FeatureMean.resize(voFeatureSet[0].size(), 0.0f);
		m_FeatureStd.resize(voFeatureSet[0].size(), 0.0f);
		for (int i = 0; i < voFeatureSet[0].size(); i++)
		{
			for (int k = 0; k < voFeatureSet.size(); k++)
			{
				m_FeatureMean[i] += voFeatureSet[k][i];
			}
			m_FeatureMean[i] /= voFeatureSet.size();
			for (int k = 0; k < voFeatureSet.size(); k++)
			{
				m_FeatureStd[i] += (voFeatureSet[k][i] - m_FeatureMean[i])*(voFeatureSet[k][i] - m_FeatureMean[i]);
			}
			m_FeatureStd[i] = std::sqrt(m_FeatureStd[i] / voFeatureSet.size());
		}
	}
	
	for (int i = 0; i < voFeatureSet[0].size(); i++)
	{
		for (int k = 0; k < voFeatureSet.size(); k++)
		{
			voFeatureSet[k][i] = (voFeatureSet[k][i] - m_FeatureMean[i]) / m_FeatureStd[i];
		}
	}
}

//******************************************************************************
//FUNCTION:计算MP相异度,全局
float CTrainingSet::calMPDissimilarityGlobal(const std::vector<int>& vLeafIndex, const std::vector<float>& vFeatures, float vPredictResponse)
{
	float SumMP = 0.f;
	for (int i = 0; i < vLeafIndex.size(); i++)
	{
		std::vector<float> LeafNodeFeatureValue = getFeatureInstanceAt(vLeafIndex[i]);
		float MPValue = __calMPValue(LeafNodeFeatureValue, vFeatures, vPredictResponse);
		SumMP += MPValue;
	}

	return SumMP / vLeafIndex.size();
}

//****************************************************************************************************
//FUNCTION:
float CTrainingSet::calEuclideanDistance(const std::vector<int>& vLeafIndex, const std::vector<float>& vFeature)
{
	float LPParam = 2.0f;
	std::vector<float> FeatureValue;
	float SumLP = 0.f;
	for (int i = 0; i < vLeafIndex.size(); i++)
	{
		float LPValueSum = 0.f;
		FeatureValue = getFeatureInstanceAt(vLeafIndex[i]);
		for (int k = 0; k < FeatureValue.size(); k++)
		{
			LPValueSum += pow(vFeature[k] - FeatureValue[k], LPParam);
		}
		SumLP += pow(LPValueSum, 1.0 / LPParam);
	}
	return SumLP / vLeafIndex.size();
}

//****************************************************************************************************
//FUNCTION:
std::pair<int, float> CTrainingSet::calMinMPAndIndex(const std::vector<int>& vDataIndex, const std::vector<float>& vFeature)
{
	_ASSERT(!vDataIndex.empty());

	float MinMP = FLT_MAX;
	int MinMPIndex = vDataIndex[0];
	std::vector<float> MP(vDataIndex.size(), 0.f);
	for (int i = 0; i < vDataIndex.size(); ++i)
	{
		MP[i] = __calMPValue(getFeatureInstanceAt(vDataIndex[i]), vFeature);
		if (MP[i] < MinMP)
		{
			MinMP = MP[i];
			MinMPIndex = vDataIndex[i];
		}
	}

	return std::make_pair(MinMPIndex, MinMP);
}

//******************************************************************************
//FUNCTION:返回某个范围内点落入的点数
void CTrainingSet::__countIntervalNode(const std::vector<std::pair<float, float>>& vMaxMinValue, std::vector<int>& voIntervalCount, std::vector<std::pair<float, float>>& voInterResponseRange)
{
	_ASSERTE(vMaxMinValue.size() == m_FeatureSet[0].size());
    #pragma omp parallel for
	for (int i = 0; i < m_FeatureSet[0].size(); i++)//确定某一维
	{
		int count = 0;
		for (int j = 0; j < m_FeatureSet.size(); j++)
		{
			if (m_FeatureSet[j][i] <= vMaxMinValue[i].first && m_FeatureSet[j][i] >= vMaxMinValue[i].second)
			{
				count++;
				voInterResponseRange[i].first = (voInterResponseRange[i].first < getResponseValueAt(j)) ? getResponseValueAt(j) : voInterResponseRange[i].first;
				voInterResponseRange[i].second = (voInterResponseRange[i].second > getResponseValueAt(j)) ? getResponseValueAt(j) : voInterResponseRange[i].second;
			}
		}
		voIntervalCount[i] = count;
	}
	/*int count = 0;
	for (int i = 0; i < m_FeatureSet.size(); i++)
	{
		for (int j = 0; j < m_FeatureSet[0].size(); j++)
		{
			if (m_FeatureSet[i][j] >= vMaxMinValue[j].first || m_FeatureSet[i][j] <= vMaxMinValue[j].second)
			{
				break;
			}
			if (j == m_FeatureSet[0].size() - 1)
			{
				voInterResponseRange[0].first = (voInterResponseRange[0].first < getResponseValueAt(i)) ? getResponseValueAt(i) : voInterResponseRange[0].first;
				voInterResponseRange[0].second = (voInterResponseRange[0].second > getResponseValueAt(i)) ? getResponseValueAt(i) : voInterResponseRange[0].second;
				count++;
			}
		}
	}
	voIntervalCount[0] = count;*/
}
