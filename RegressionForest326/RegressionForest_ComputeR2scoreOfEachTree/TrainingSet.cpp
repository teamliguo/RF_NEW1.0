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
	float MPParam = 2.0f;
	std::vector<std::vector<float>> LeafNodeFeatureValue;
	float MinMPDissimilarity = (std::numeric_limits<float>::max)();
	std::vector<std::pair<float, int>> OrderStandardDeviation;
	float SumMP = 0.f;
	for (int i = 0; i < vLeafIndex.size(); i++)
	{
		LeafNodeFeatureValue.push_back(getFeatureInstanceAt(vLeafIndex[i]));//LeafNodeFeatureValue[i]的所有维度求最大最小
		std::vector<std::pair<float, float>> MaxMinValue;
		std::vector<int> EachFeatureIntervalCount;
		std::vector<int> ResponseCount;

		for (int j = 0; j < LeafNodeFeatureValue[0].size(); j++)//16 Dims
		{
			float StandardDeviation = getEachDimStandard()[j];
			MaxMinValue.push_back({ std::max(LeafNodeFeatureValue[i][j], vFeatures[j])/* + StandardDeviation*/, std::min(LeafNodeFeatureValue[i][j], vFeatures[j])/* - StandardDeviation*/ });//加标准差,计算当前样本与测试集每一维的范围
		}
		float MPValueSum = 0.f;
		std::vector<std::pair<float, float>> ResponseRange(MaxMinValue.size(), { vPredictResponse, vPredictResponse });
		EachFeatureIntervalCount.resize(MaxMinValue.size(), 0);
		ResponseCount.resize(MaxMinValue.size(), 0);
		__countIntervalNode(MaxMinValue, m_FeatureSet, EachFeatureIntervalCount, ResponseRange);//获得每个维度在对应区间的样本个数，16 Dims

		/*for (int i = 0; i < m_FeatureSet[0].size(); i++)//确定某一维
		{
			int count = 0;
			for (int j = 0; j < m_FeatureSet.size(); j++)
			{
				if (m_pResponseSet[j] <= ResponseRange[i].first && m_pResponseSet[j] >= ResponseRange[i].second)
				{
					count++;
				}
			}
			ResponseCount[i] = count;
		}*/

		for (int k = 0; k < EachFeatureIntervalCount.size(); k++)
		{
			MPValueSum += pow(((float)EachFeatureIntervalCount[k] / (float)getNumOfInstances()) /**((ResponseRange[k].first - ResponseRange[k].second) / (getResponseRange().second - getResponseRange().first))*/, MPParam);
		}
		//std::cout << MPValueSum << std::endl;
		SumMP += pow(MPValueSum, 1 / MPParam);
	}
	return SumMP / vLeafIndex.size();
}

//******************************************************************************
//FUNCTION:计算MP相异度,叶子节点内部计算概率
//float CTrainingSet::calMPDissimilarity(const std::vector<int>& vLeafIndex, const std::vector<float>& vFeatures)
//{
//	float MPParam = 2.0f;
//	std::vector<std::vector<float>> LeafNodeFeatureValue;
//	float MinMPValueResponse;
//	float MinMPDissimilarity = (std::numeric_limits<float>::max)();
//	for (int i = 0; i < vLeafIndex.size(); i++)
//		LeafNodeFeatureValue.push_back(getFeatureInstanceAt(vLeafIndex[i]));
//	std::vector<float> EachDimStandardDevation;
//	__calStandardDeviation(LeafNodeFeatureValue, EachDimStandardDevation);
//	std::vector<std::pair<float, int>> OrderStandardDeviation;
//	for (int i = 0; i < vLeafIndex.size(); i++)
//	{
//		float CurrentSampleMPValue;
//		std::vector<std::pair<float, float>> EachDimMaxMinValue;
//		std::vector<int> EachFeatureIntervalCount;
//		for (int j = 0; j < LeafNodeFeatureValue[0].size(); j++)
//		{
//			EachDimMaxMinValue.push_back({ std::max(LeafNodeFeatureValue[i][j], vFeatures[j]) + EachDimStandardDevation[j], std::min(LeafNodeFeatureValue[i][j], vFeatures[j]) - EachDimStandardDevation[j] });//加标准差,计算当前样本与测试集每一维的范围
//		}
//		__countIntervalNode(EachDimMaxMinValue, LeafNodeFeatureValue, EachFeatureIntervalCount);
//		float MPValueSum = 0.f;
//		for (int i = 0; i < EachFeatureIntervalCount.size(); i++)
//			MPValueSum += pow((float)EachFeatureIntervalCount[i] / (float)vLeafIndex.size(), MPParam);
//		CurrentSampleMPValue = pow(MPValueSum, 1.0f / MPParam);
//		OrderStandardDeviation.push_back({CurrentSampleMPValue, vLeafIndex[i]});
//
//		if (vLeafIndex.size() <= 5 && CurrentSampleMPValue < MinMPDissimilarity)
//		{
//		    MinMPDissimilarity = CurrentSampleMPValue;
//			MinMPValueResponse = getResponseValueAt(vLeafIndex[i]);
//		}
//	}
//	if(vLeafIndex.size() > 5)
//	{
//		sort(OrderStandardDeviation.begin(), OrderStandardDeviation.end(), comparePair);
//		float sumPredictValue = 0.f;
//		for (auto i = 0; i < 5; i++)
//		{
//			sumPredictValue += getResponseValueAt(OrderStandardDeviation[i].second);
//			//std::cout << "local Index " << i << "th = " << OrderStandardDeviation[i].second << std::endl;
//		}
//		MinMPValueResponse = sumPredictValue / 5.0f;
//	}
//
//	return MinMPValueResponse;
//}

//******************************************************************************
//FUNCTION:返回某个范围内点落入的点数
void CTrainingSet::__countIntervalNode(const std::vector<std::pair<float, float>>& vMaxMinValue, const std::vector<std::vector<float>>& vFeatureDataSet, std::vector<int>& voIntervalCount, std::vector<std::pair<float, float>>& voInterResponseRange)
{
	_ASSERTE(vMaxMinValue.size() == vFeatureDataSet[0].size());
    //#pragma omp parallel for
	for (int i = 0; i < vFeatureDataSet[0].size(); i++)//确定某一维
	{
		int count = 0;
		for (int j = 0; j < vFeatureDataSet.size(); j++)
		{
			if (vFeatureDataSet[j][i] <= vMaxMinValue[i].first && vFeatureDataSet[j][i] >= vMaxMinValue[i].second)
			{
				count++;
				voInterResponseRange[i].first = (voInterResponseRange[i].first < getResponseValueAt(j)) ? getResponseValueAt(j) : voInterResponseRange[i].first;
				voInterResponseRange[i].second = (voInterResponseRange[i].second > getResponseValueAt(j)) ? getResponseValueAt(j) : voInterResponseRange[i].second;
			}
		}
		voIntervalCount[i] = count;
	}
}
