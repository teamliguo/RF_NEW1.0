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

//****************************************************************************************************
//FUNCTION:
bool CTrainingSet::compareTuple(const std::tuple<std::vector<float>, float, float>& vFirst, std::tuple<std::vector<float>, float, float>& vSecond)
{
	return std::get<2>(vFirst) < std::get<2>(vSecond);
}

//******************************************************************************
//FUNCTION:
bool CTrainingSet::comparePair(const std::pair<float, float>& vFirst, const std::pair<float, float>& vSecond)
{
	return vFirst.second < vSecond.second;
}
//******************************************************************************
//FUNCTION:
bool CTrainingSet::comparePair2(const std::pair<float, float>& vFirst, const std::pair<float, float>& vSecond)
{
	return vFirst.second > vSecond.second;
}

bool CTrainingSet::cmpByIndexDesc(const std::vector<float> &a, const std::vector<float> &b, int index)
{
	return a[index] > b[index];
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
	for (int i = 0; i < m_EachDimStandard.size(); i++)
	std::cout << m_EachDimStandard[i] << " ";
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

//****************************************************************************************************
//FUNCTION:
float CTrainingSet::__calMPValue(const std::vector<std::vector<float>>& vAllLeafFeatureSet, const std::vector<float>& vAllLeafResponseSet, const std::vector<float>& vLeafDate, const std::vector<float>& vTestData, float vPredictResponse)
{
	float MPParam = 2.0f, MPValueSum = 0.f;
	std::vector<std::pair<float, float>> MaxMinValue;
	std::vector<int> EachFeatureIntervalCount(vLeafDate.size(), 0);
	std::vector<std::pair<float, float>> ResponseRange(vLeafDate.size(), { vPredictResponse, vPredictResponse });

	for (int j = 0; j < vLeafDate.size(); j++)
	{
		/*float StandardDeviation = getEachDimStandard()[j];*/
		float StandardDeviation = 0;
		MaxMinValue.push_back({ std::max(vLeafDate[j], vTestData[j]) + StandardDeviation, std::min(vLeafDate[j], vTestData[j]) - StandardDeviation });
	}
	__calIntervalSample(MaxMinValue, vAllLeafFeatureSet, vAllLeafResponseSet, EachFeatureIntervalCount, ResponseRange); 

	for (int i = 0; i < EachFeatureIntervalCount.size(); i++)
	{
		/*float n=(float)getNumOfInstances();*/
		MPValueSum += pow(((float)EachFeatureIntervalCount[i] / vAllLeafResponseSet.size()) /** MPParam*/ /**((ResponseRange[i].first - ResponseRange[i].second) / m_ResponseRange)*/, MPParam);
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

//******************************************************************************
//FUNCTION:局部
std::vector<std::tuple<std::vector<float>, float, float>> CTrainingSet::calReCombineDataMP(const std::vector<std::vector<float>>& vAllLeafFeatureSet, const std::vector<float>& vAllLeafResponseSet, const std::vector<float>& vFeatures, float vPredictResponse)
{
	float MPParam = 2.0f;
	std::vector<std::tuple<std::vector<float>, float, float>> MPSet;//first 为训练样本Y值，second 为MP值
	
	float AllREsponseRange = *std::max_element(vAllLeafResponseSet.begin(), vAllLeafResponseSet.end()) - *std::min_element(vAllLeafResponseSet.begin(), vAllLeafResponseSet.end());
	int InstanceNoRepeat = CTrainingSetConfig::getInstance()->getAttribute<int>(KEY_WORDS::INSTANCE_NO_REPEAT);
	 
	std::vector<std::vector<float>> vAllLeafFeatureSetChanged;//排序
	
	//行列互换
	for (int j = 0;j < vAllLeafFeatureSet[0].size();j++)
	{
		std::vector<float> vAllLeafFeatureRow;
		for (int i = 0;i < vAllLeafFeatureSet.size();i++)
		{
			vAllLeafFeatureRow.push_back(vAllLeafFeatureSet[i][j]);
		}
		vAllLeafFeatureSetChanged.push_back(vAllLeafFeatureRow);
	}
	//按行排序-维度-行
	for (int j = 0;j < vAllLeafFeatureSetChanged.size();j++)
	{
 		std::sort(vAllLeafFeatureSetChanged[j].begin(), vAllLeafFeatureSetChanged[j].end());
	}

	for (int i = 0; i < vAllLeafFeatureSet.size(); i++)
	{
		std::vector<float> TrainFeatureValue = vAllLeafFeatureSet[i];
		std::vector<std::pair<float, float>> MaxMinValue;
		std::vector<int> EachFeatureIntervalCount(vAllLeafFeatureSet[i].size(), 0);
		std::vector<std::pair<float, float>> ResponseRange(vAllLeafFeatureSet[i].size(), { vPredictResponse, vPredictResponse });
		float MPValueSum = 0.f;

		for (int j = 0; j < TrainFeatureValue.size(); j++)
		{
			MaxMinValue.push_back({ std::max(TrainFeatureValue[j], vFeatures[j]), std::min(TrainFeatureValue[j], vFeatures[j]) });//暂时去掉标准差,因为加上标准差范围太大
		}

		/*__calIntervalSample(MaxMinValue, vAllLeafFeatureSet, vAllLeafResponseSet, EachFeatureIntervalCount, ResponseRange);*/
		//__countIntervalNode(MaxMinValue, EachFeatureIntervalCount, ResponseRange);
		__calIntervalSampleByOrder(MaxMinValue, vAllLeafFeatureSetChanged, vAllLeafResponseSet, EachFeatureIntervalCount, ResponseRange);

		for (int k = 0; k < EachFeatureIntervalCount.size(); k++)//遍历每一维
		{
			MPValueSum += pow(((float)EachFeatureIntervalCount[k] / vAllLeafResponseSet.size())/**(((ResponseRange[k].first - ResponseRange[k].second) / AllREsponseRange))*/, MPParam);
		}
			 
	    MPSet.push_back(std::make_tuple(vAllLeafFeatureSet[i], vAllLeafResponseSet[i], pow(MPValueSum, 1 / MPParam) ));
	}
	std::sort(MPSet.begin(), MPSet.end(), compareTuple);
	
	//去掉重复值-----zy20190214
	if (InstanceNoRepeat == 1)
	{
		MPSet.erase(unique(MPSet.begin(), MPSet.end()), MPSet.end());

	}

	return MPSet;
}

//******************************************************************************
//FUNCTION:局部
std::vector<std::pair<float, float>> CTrainingSet::calReCombineDataLP(const std::vector<std::vector<float>>& vAllLeafFeatureSet, const std::vector<float>& vAllLeafResponseSet, const std::vector<float>& vFeatures, float vPredictResponse)
{
	float LPParam = 2.0f;
	std::vector<std::pair<float, float>> LPSet;//first 为训练样本Y值，second 为LP值  
	int InstanceNoRepeat = CTrainingSetConfig::getInstance()->getAttribute<int>(KEY_WORDS::INSTANCE_NO_REPEAT);
	float LPValueSum = 0.f;
	for (int i = 0; i < vAllLeafFeatureSet.size(); i++)
	{
		std::vector<float> TrainFeatureValue = vAllLeafFeatureSet[i]; 
		float MPValueSum = 0.f;  
		for (int i = 0; i < TrainFeatureValue.size(); ++i)
		{
			LPValueSum += pow(TrainFeatureValue[i] - vFeatures[i], LPParam);
		}

		LPSet.push_back({ vAllLeafResponseSet[i], pow(LPValueSum, 1 / LPParam) });


	}
	std::sort(LPSet.begin(), LPSet.end(), comparePair);

	//去掉重复值-----zy20190214
	if (InstanceNoRepeat == 1)
	{
		LPSet.erase(unique(LPSet.begin(), LPSet.end()), LPSet.end());

	}

	return LPSet;
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
float CTrainingSet::calEuclideanDistance(int vLeafIndex, const std::vector<float>& vFeature)
{
	float LPParam = 2.0f;
	std::vector<float> FeatureValue;
	float SumLP = 0.f;
	 
		float LPValueSum = 0.f;
		FeatureValue = getFeatureInstanceAt(vLeafIndex);
		for (int k = 0; k < FeatureValue.size(); k++)
		{
			LPValueSum += pow(vFeature[k] - FeatureValue[k], LPParam);
		}
	 
	return pow(LPValueSum,1/LPParam);
}
//****************************************************************************************************
//FUNCTION:  
std::pair<int, float>   CTrainingSet::calMinMPAndIndex(const std::vector<int>& vDataIndex, const std::vector<float>& vFeature)
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

//****************************************************************************************************
//FUNCTION:  
std::vector<std::pair<int, float>>  CTrainingSet::calMinMPAndIndex(const std::vector<int>& vDataIndex, const std::vector<float>& vFeature, int vInstanceMethod, float vInstanceNumber, float vInstanceNumberRatio)
{
	_ASSERT(!vDataIndex.empty());
	std::vector<std::pair<int, float>> MPSet;//first 为训练样本索引，second 为MP值
	std::vector<std::pair<int, float>> MPSetByNumber;//first 为训练样本索引，second 为MP值
	float MinMP = FLT_MAX;
	int MinMPIndex = vDataIndex[0];
	std::vector<float> MP(vDataIndex.size(), 0.f);
	
	for (int i = 0; i < vDataIndex.size(); ++i)
	{
		/*MP[i] = __calMPValue(getFeatureInstanceAt(vDataIndex[i]), vFeature);*/
		MP[i] = __calMPValue(getFeatureInstanceAt(vDataIndex[i]), vFeature);
		/*if (MP[i] < MinMP)
		{
			MinMP = MP[i];
			MinMPIndex = vDataIndex[i];
		}*/
		MPSet.push_back({ vDataIndex[i],MP[i] });
	}
	std::sort(MPSet.begin(), MPSet.end(), comparePair);

	if (vInstanceMethod == 0)
	{
		if (vInstanceNumber <= 0) vInstanceNumber = 1;
		if (vInstanceNumber > MP.size()) vInstanceNumber = MP.size();
	}
	else
	{
		int allResponseNumber = MP.size();
		vInstanceNumber = (int)(allResponseNumber *vInstanceNumberRatio);
	}
	for (int i = 0;i < vInstanceNumber;i++)
	{
		MPSetByNumber.push_back({ MPSet[i].first,MPSet[i].second });
	}
	return MPSetByNumber;
}

//****************************************************************************************************
//FUNCTION:  
std::vector<std::pair<int, float>>  CTrainingSet::calMinMPAndIndex(const std::vector<std::vector<float>>& vAllLeafFeatureSet, const std::vector<float>& vAllLeafResponseSet, const std::vector<int>& vDataIndex, const std::vector<float>& vFeature, int vInstanceMethod, float vInstanceNumber, float vInstanceNumberRatio)
{
	_ASSERT(!vDataIndex.empty());
	std::vector<std::pair<int, float>> MPSet;//first 为训练样本索引，second 为MP值
	std::vector<std::pair<int, float>> MPSetByNumber;//first 为训练样本索引，second 为MP值
	float MinMP = FLT_MAX;
	int MinMPIndex = vDataIndex[0];
	std::vector<float> MP(vDataIndex.size(), 0.f);

	for (int i = 0; i < vDataIndex.size(); ++i)
	{
		 
		MP[i] = __calMPValue(  vAllLeafFeatureSet,  vAllLeafResponseSet, getFeatureInstanceAt(vDataIndex[i]), vFeature);
		/*if (MP[i] < MinMP)
		{
		MinMP = MP[i];
		MinMPIndex = vDataIndex[i];
		}*/
		MPSet.push_back({ vDataIndex[i],MP[i] });
	}
	std::sort(MPSet.begin(), MPSet.end(), comparePair);

	if (vInstanceMethod == 0)
	{
		if (vInstanceNumber <= 0) vInstanceNumber = 1;
		if (vInstanceNumber > MP.size()) vInstanceNumber = MP.size();
	}
	else
	{
		int allResponseNumber = MP.size(); 
		vInstanceNumber = (int)(allResponseNumber *vInstanceNumberRatio);
		if(vInstanceNumber<=0) vInstanceNumber = 1;
	}
	for (int i = 0; i < vInstanceNumber; i++)
	{
		MPSetByNumber.push_back({ MPSet[i].first,MPSet[i].second });
	}
	return MPSetByNumber;
}
//****************************************************************************************************
//FUNCTION:
std::vector<std::pair<int, float>>  CTrainingSet::calMinLPAndIndex(const std::vector<int>& vDataIndex, const std::vector<float>& vFeature, int vInstanceMethod, float vInstanceNumber, float vInstanceNumberRatio)
{
	_ASSERT(!vDataIndex.empty());
	std::vector<std::pair<int, float>> LPSet;//first 为训练样本索引，second 为MP值
	std::vector<std::pair<int, float>> LPSetByNumber;//first 为训练样本索引，second 为MP值
	float MinLP = FLT_MAX;
	int MinLPIndex = vDataIndex[0];
	std::vector<float> LP(vDataIndex.size(), 0.f);
	for (int i = 0; i < vDataIndex.size(); ++i)
	{
		LP[i] = calEuclideanDistance(vDataIndex[i], vFeature);
	/*	if (LP[i] < MinLP)
		{
			MinLP = LP[i];
			MinLPIndex = vDataIndex[i];
		}*/
		LPSet.push_back({ vDataIndex[i],LP[i] });
	}

	std::sort(LPSet.begin(), LPSet.end(), comparePair);

	if (vInstanceMethod == 0)
	{
		if (vInstanceNumber <= 0) vInstanceNumber = 1;
		if (vInstanceNumber > LP.size()) vInstanceNumber = LP.size();
	}
	else
	{
		int allResponseNumber = LP.size();
		vInstanceNumber = (int)(allResponseNumber *vInstanceNumberRatio);
		if (vInstanceNumber <= 0) vInstanceNumber = 1;
	}
	for (int i = 0;i < vInstanceNumber;i++)
	{
		LPSetByNumber.push_back({ LPSet[i].first,LPSet[i].second });
	}
	return LPSetByNumber;
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

//******************************************************************************
//FUNCTION:局部范围内搜索
void CTrainingSet::__calIntervalSample(const std::vector<std::pair<float, float>>& vMaxMinValue, const std::vector<std::vector<float>>& vAllLeafFeatureSet, const std::vector<float>& vAllLeafResponseSet, std::vector<int>& voIntervalCount, std::vector<std::pair<float, float>>& voInterResponseRange)
{
	_ASSERTE(vMaxMinValue.size() == vAllLeafFeatureSet[0].size());
	_ASSERTE(!vAllLeafFeatureSet.empty() && !vAllLeafResponseSet.empty() && !voIntervalCount.empty() && !voInterResponseRange.empty());
#pragma omp parallel for
	for (int i = 0; i < vAllLeafFeatureSet[0].size(); i++)
	{
		int count = 0;
		for (int j = 0; j < vAllLeafFeatureSet.size(); j++)
		{
			if (vAllLeafFeatureSet[j][i] <= vMaxMinValue[i].first && vAllLeafFeatureSet[j][i] >= vMaxMinValue[i].second)
			{
				count++;
				voInterResponseRange[i].first = (voInterResponseRange[i].first < vAllLeafResponseSet[j]) ? vAllLeafResponseSet[j] : voInterResponseRange[i].first;
				voInterResponseRange[i].second = (voInterResponseRange[i].second > vAllLeafResponseSet[j]) ? vAllLeafResponseSet[j] : voInterResponseRange[i].second;
			}
		}
		voIntervalCount[i] = count;
	}
}

 
////******************************************************************************
////FUNCTION:局部范围内搜索-在按特征值升序排序的样本中
void CTrainingSet::__calIntervalSampleByOrder(const std::vector<std::pair<float, float>>& vMaxMinValue, const std::vector<std::vector<float>>& vAllLeafFeatureSetOrdered, const std::vector<float>& vAllLeafResponseSet, std::vector<int>& voIntervalCount, std::vector<std::pair<float, float>>& voInterResponseRange)
{
	_ASSERTE(vMaxMinValue.size() == vAllLeafFeatureSetOrdered.size());
	_ASSERTE(!vAllLeafFeatureSetOrdered.empty() && !vAllLeafResponseSet.empty() && !voIntervalCount.empty() && !voInterResponseRange.empty());

	//特征个数，维度
//#pragma omp parallel for
	for (int i = 0; i < vAllLeafFeatureSetOrdered.size(); i++)
	{
		int count = 0;
		//某个特征，遍历排序的特征值
		int indexMin = 0;
	 
		int indexMax = 0;
		//获取最小值的下标
		for (int j = 0; j < vAllLeafFeatureSetOrdered[i].size(); j++) 
		{
			if (vAllLeafFeatureSetOrdered[i][j] >= vMaxMinValue[i].second)
			{
				indexMin = j; 
				indexMax = indexMin;
				break; 
			} 
		}
		//获取最大值的下标
		for (int k = indexMin; k < vAllLeafFeatureSetOrdered[i].size(); k++)
		{
			if (vAllLeafFeatureSetOrdered[i][k] <= vMaxMinValue[i].first)
			{
				indexMax = k;
				
			}
			else
			{
				break;
			}
		}

		voIntervalCount[i] = indexMax - indexMin + 1;;
	}
}

//********************************************************************************************************
//FUNCTION:
std::vector<std::pair<int, float>>  CTrainingSet::calIfMp(unsigned int vNumOfUsingTrees, std::vector<int> TotalDataIndex, std::vector<std::vector<int>> NodeDataIndex)
{
	 
	//去掉重复点
	sort(TotalDataIndex.begin(), TotalDataIndex.end()); 
	TotalDataIndex.erase(unique(TotalDataIndex.begin(), TotalDataIndex.end()), TotalDataIndex.end());
 
	std::vector<std::pair<int, int>>  DataIndexCount;
	std::vector<std::pair<int, float>>  DataIndexMp;//first 为训练样本index值，second 为MP值
	sort(TotalDataIndex.begin(), TotalDataIndex.end());
	int count = 0;
	for (int i = 0; i < TotalDataIndex.size(); i++)
   {
		int dataIndex = TotalDataIndex[i];
		for (int j = 0; j < vNumOfUsingTrees; j++)
		{
			for (int k = 0;k < NodeDataIndex[j].size();k++)
			{
				if (NodeDataIndex[j][k] == dataIndex)
				{
					DataIndexCount.push_back({ dataIndex, NodeDataIndex[j].size() });
					break;
				}
			}
			 
		}
  	 }

	 //统计每个点的域内样本之和
	
	int mpDataIndex = DataIndexCount[0].first;
	float sumOfDataIndex = 0.0f;
	for (int i = 0; i < DataIndexCount.size(); i++)
	{
		if (DataIndexCount[i].first == mpDataIndex)
		{
			sumOfDataIndex += DataIndexCount[i].second;
		}
		else
		{
			DataIndexMp.push_back({ mpDataIndex,sumOfDataIndex });
			mpDataIndex = DataIndexCount[i].first;
			sumOfDataIndex = 0.0f;
			i = i - 1;
			 
		}
	}
	////获取最相似的点的y：count最大的点
	//int maxCount = DataIndexMp[0].second;
	//int maxIndex = DataIndexMp[0].first;
	//for (int i = 0; i < DataIndexMp.size(); i++)
	//{
	//	if (DataIndexMp[i].second > maxCount)
	//	{
	//		maxCount = DataIndexMp[i].second;
	//		maxIndex= DataIndexMp[i].first;
	//	}
	//}
	std::sort(DataIndexMp.begin(), DataIndexMp.end(), comparePair2);

	return DataIndexMp;
}
