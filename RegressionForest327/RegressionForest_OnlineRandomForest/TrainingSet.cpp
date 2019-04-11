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
//FUNCTION:���ݷ�Χ����ܱ���±�
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
//FUNCTION:�����׼��
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
			if (m_FeatureStd[i] == 0)
			m_FeatureStd[i] = 1.0f;
			voFeatureSet[k][i] = (voFeatureSet[k][i] - m_FeatureMean[i]) / m_FeatureStd[i];
		}
	}
}

//-KMeans����Ԥ��ֵ----------------------------- 
std::vector<std::pair<int, float>> CTrainingSet::kMeans(std::vector<std::vector<float>>& voFeatureSet, int TreeNum,int clusterK,int iterTime,std::vector<int>&clusters)
{
	std::vector<std::vector<float>> randKFeature;
	std::vector <std::vector<std::vector<float>>> clustersFeature;//��¼�����ص���������
	
	std::vector<std::vector<float>> currentMeanCenter;
	std::vector<std::vector<float>> oldMeanCenter;//��¼���ĵ����������� 
	std::vector<std::vector<float>> lpFeature;
	std::vector<std::pair<int,float>> testToMid;//���������������ĵ�ľ���.first-�ر�ţ�second-������ĵľ���
	//std::vector<int> intRIndex;
	//int firstIndex = rand() % (voFeatureSet.size() - 1);
	//intRIndex.push_back({ firstIndex });
	//
	//for (int i = 1;i < clusterK;i++)
	//{
	//	int j = 0;
	//	int tempIndex = rand() % (voFeatureSet.size() - 1);
	//	for (;j < intRIndex.size();j++)
	//	{
	//		if (tempIndex == intRIndex[j])
	//			break;
	//	}
	//	if (j == intRIndex.size())
	//	{
	//		intRIndex.push_back({ tempIndex });
	//	}
	//	else
	//	{
	//		i--;
	//	}
	//}
	//for (int i = 0;i < intRIndex.size();i++)
	//{
	//	randKFeature.push_back({ voFeatureSet[intRIndex[i]] });//���ѡ�������

	//}
	//randKFeature.push_back({ voFeatureSet[intRanIndex] });//���ѡ�������
 //
	////��ʼ�������ѡ����ͬ�����ĵ�
	//for (int i = 1;i < clusterK;i++)
	//{ 
	//	int intRanIndex = ;;
	//	int count = 1;
	//	for (int j = 0;j < randKFeature.size();j++)
	//	{
	//		if(isSameVandV(voFeatureSet[intRanIndex],randKFeature[j]))
	//		{
	//			i--;//���¼���intRanIndex
	//			break; //��ͬ
	//		}
	//		count++;//����ͬ
	//	}
	//	if (count == randKFeature.size())//��randKFeature������ͬ
	//	{
	//		randKFeature.push_back({ voFeatureSet[intRanIndex] });
	//	}
	//	 
	//}
	//��ȡԭʼ����Ҷ�ӽڵ����ĵ���Ϊ��ʼ�ľ�ֵ
	oldMeanCenter.resize(TreeNum);
	int j = voFeatureSet.size() - TreeNum ;
	int oldIndex = 0;
	for (;j < voFeatureSet.size();j++)
	{
	 
		oldMeanCenter[oldIndex].assign(voFeatureSet[j].begin(), voFeatureSet[j].end());
		oldIndex++;
	}
	int iters = 0;
	do
	{ 
		clustersFeature.clear();
		lpFeature.clear(); 
		currentMeanCenter.clear();
		clustersFeature.resize(clusterK);
		//�������������ֵ�����ľ���
		for (int k = 0;k < voFeatureSet.size() - TreeNum-1;k++)
		{
			std::vector<float> lpToMeanFeat;
			for (int j = 0;j < oldMeanCenter.size();j++)
			{
				
				float lp=calLp(voFeatureSet[k], oldMeanCenter[j]);
				lpToMeanFeat.push_back({ lp });
				
			}
			lpFeature.push_back({ lpToMeanFeat });
		}
		//���ݾ��룬�����������ڵĴ�
		for (int i = 0;i < lpFeature.size();i++)
		{
			int minLpIndex = 0;
			for (int j = 1;j < lpFeature[0].size();j++)
			{
				if (lpFeature[i][minLpIndex]> lpFeature[i][j])
				{
					minLpIndex = j;
				}
				

			 }
			clusters[i]= minLpIndex; 

		}
		
        
	    for (int k = 0;k < clusterK;k++)
	   {
		  for (int i = 0;i < clusters.size();i++)
		 {
			
				if (k == clusters[i])
				{
					/*std::vector<float> tempFeat = ;
					std::vector<std::vector<float>> */
					clustersFeature[k].push_back({ voFeatureSet[clusters[i]] });//�������ڴصı�ţ����ִص�����
				}

			}
		}

		//���¼�����صľ�ֵ����
	    for (int k = 0;k < clusterK;k++)
	   {
		 std::vector<float> meanFeatVector;
		 for (int j = 0;j < voFeatureSet[0].size();j++)
		 { 
		  float sumOfFeat = 0.0f;
		  for (int i = 0;i < clustersFeature[k].size();i++)
		  { 
				sumOfFeat += clustersFeature[k][i][j];
		   }
		  if (clustersFeature[k].size() != 0)
		  {
			  sumOfFeat /= clustersFeature[k].size();
		  }
		 
		  meanFeatVector.push_back(sumOfFeat);
		}
		currentMeanCenter.push_back({ meanFeatVector });//��¼�µĴ����ĵ�����
	}
	 if (currentMeanCenter.size() != oldMeanCenter.size())
	 {
		 currentMeanCenter.pop_back();
	 }

    std::vector<bool> cmpFlag = isSameVandV(currentMeanCenter,oldMeanCenter);//�жϵ�ǰ���ĵ� ����һ�����ĵ��Ƿ���ͬ
	int breakFlag = 0;
	for (int j = 0;j < cmpFlag.size();j++)
   {
		  if (cmpFlag[j])
		 {
			  breakFlag++;
			  continue;//��ͬ����
		 } 
		  else
		  {
			  //��ͬ�����¸���
			  oldMeanCenter[j].assign(currentMeanCenter[j].begin(), currentMeanCenter[j].end());

		  }
    }
    if (breakFlag == clusterK)//�����ǰ��ֵ������û�и��£��˳�ѭ��
    {
			break;
    } 	
	iters++;
	}while (iters < iterTime);//�ﵽ�����������˳�ѭ��
   //����������������ĸ��أ��������������������ĵľ��� 
	
	for (int k = 0;k < oldMeanCenter.size();k++)
	{ 
		 float lp = calLp(voFeatureSet[voFeatureSet.size()- TreeNum-1], oldMeanCenter[k]);
		 testToMid.push_back({k,lp });  
	}
	std::sort(testToMid.begin(), testToMid.end(),comparePair);
	return testToMid;
}



//-KMeans����Ԥ��ֵ--�صĸ�����̬����--���ݴص������Ƿ��ȶ������㷨------------------------- 
std::vector<std::pair<int, float>> CTrainingSet::kMeansByDyCls(std::vector<std::vector<float>>& voFeatureSet, int TreeNum, int clusterK, int iterTime, std::vector<int>&clusters)
{
	std::vector<std::vector<float>> randKFeature;
	std::vector <std::vector<std::vector<float>>> clustersFeature;//��¼�����ص���������
	std::vector <std::vector<std::vector<float>>> oldClustersFeature;
	std::vector<std::vector<float>> currentMeanCenter;
	std::vector<std::vector<float>> oldMeanCenter;//��¼���ĵ����������� 
	std::vector<std::vector<float>> lpFeature;
	std::vector<std::pair<int, float>> testToMid;//���������������ĵ�ľ���.first-�ر�ţ�second-������ĵľ���
											 
	oldMeanCenter.resize(TreeNum);
	int j = voFeatureSet.size() - TreeNum;
	int oldIndex = 0;
	for (;j < voFeatureSet.size();j++)
	{

		oldMeanCenter[oldIndex].assign(voFeatureSet[j].begin(), voFeatureSet[j].end());
		oldIndex++;
	}
	int iters = 0;
	do
	{
		oldClustersFeature.clear();
		clustersFeature.clear();
		lpFeature.clear();
		currentMeanCenter.clear();
		clustersFeature.resize(oldMeanCenter.size());
		//�������������ֵ�����ľ���
		for (int k = 0;k < voFeatureSet.size() - TreeNum - 1;k++)
		{
			std::vector<float> lpToMeanFeat;
			for (int j = 0;j < oldMeanCenter.size();j++)
			{
				//������Ե㵽��ֵ��ľ���-LP,�ɳ���MP����
				float lp = calLp(voFeatureSet[k], oldMeanCenter[j]);
				lpToMeanFeat.push_back({ lp });

			}
			lpFeature.push_back({ lpToMeanFeat });
		}
		//���ݾ��룬�����������ڵĴ�
		for (int i = 0;i < lpFeature.size();i++)
		{
			int minLpIndex = 0;
			for (int j = 1;j < lpFeature[0].size();j++)
			{
				if (lpFeature[i][minLpIndex]> lpFeature[i][j])
				{
					minLpIndex = j;
				}


			}
			clusters[i] = minLpIndex;

		}
 
		for (int k = 0;k < oldMeanCenter.size();k++)
		{
			for (int i = 0;i < clusters.size();i++)
			{

				if (k == clusters[i])
				{
					/*std::vector<float> tempFeat = ;
					std::vector<std::vector<float>> */
					clustersFeature[k].push_back({ voFeatureSet[clusters[i]] });//�������ڴصı�ţ����ִص�����
				}

			}
		}

		//ȥ���������Ĵ�------------------
		for (int i = 0;i < clustersFeature.size();i++)
		{
			if (clustersFeature[i].size() != 0)
			{
				oldClustersFeature.push_back({ clustersFeature[i] });
			} 

		}
		//���¼�����صľ�ֵ����
		for (int k = 0;k < oldClustersFeature.size();k++)
		{
			std::vector<float> meanFeatVector;
			for (int j = 0;j < voFeatureSet[0].size();j++)
			{
				float sumOfFeat = 0.0f;
				for (int i = 0;i < oldClustersFeature[k].size();i++)
				{
					sumOfFeat += oldClustersFeature[k][i][j];
				}
				if (oldClustersFeature[k].size() != 0)
				{
					sumOfFeat /= oldClustersFeature[k].size();
				}

				meanFeatVector.push_back(sumOfFeat);
			}
			currentMeanCenter.push_back({ meanFeatVector });//��¼�µĴ����ĵ�����
		}
		int breakFlag = 0;
		if (currentMeanCenter.size() == oldMeanCenter.size())//����ظ����ޱ仯���ж�����һ�ε����ĵ��Ƿ���ͬ
		{
			std::vector<bool> cmpFlag = isSameVandV(currentMeanCenter, oldMeanCenter);//�жϵ�ǰ���ĵ� ����һ�����ĵ��Ƿ���ͬ
			
			for (int j = 0;j < cmpFlag.size();j++)
			{
				if (cmpFlag[j])
				{
					breakFlag++;
					continue;//��ͬ����
				}
				else
				{
					//��ͬ�����¸���
					oldMeanCenter[j].assign(currentMeanCenter[j].begin(), currentMeanCenter[j].end());

				}
			}

		}
		else//����ظ�������
		{
			oldMeanCenter.clear();//���
			oldMeanCenter.resize(currentMeanCenter.size());
			oldMeanCenter.assign(currentMeanCenter.begin(), currentMeanCenter.end());//����  
		}
		
		if (breakFlag == oldMeanCenter.size())//�����ǰ��ֵ������û�и��£��˳�ѭ��
		{
			break;
		}
		iters++;
	} while (iters < iterTime);//�ﵽ�����������˳�ѭ��
							   //����������������ĸ��أ��������������������ĵľ��� 

	for (int k = 0;k < oldMeanCenter.size();k++)
	{
		float lp = calLp(voFeatureSet[voFeatureSet.size() - TreeNum - 1], oldMeanCenter[k]);
		testToMid.push_back({ k,lp });
	}
	std::sort(testToMid.begin(), testToMid.end(), comparePair);
	return testToMid;
}


//-KMeans����Ԥ��ֵ--�صĸ�����̬����--���ݴص�Y�ķ����Ƿ�С����ֵ�����㷨------------------------- 
std::vector<std::pair<int, float>> CTrainingSet::kMeansByResponseVarRatio(std::vector<float>& oldClustersResponseVar, float maxResponseOfTree, float maxBiasRatio, const std::vector<float>& TotalNodeDataResponseSet, std::vector<std::vector<float>>& voFeatureSet, int TreeNum, int clusterK, int iterTime, std::vector<int>& Clusters)
{
	std::vector<std::vector<float>> randKFeature;
	std::vector <std::vector<std::vector<float>>> clustersFeature;//��¼�����ص���������
	std::vector <std::vector<std::vector<float>>> oldClustersFeature;
	std::vector<std::vector<float>> currentMeanCenter;
	std::vector<std::vector<float>> oldMeanCenter;//��¼���ĵ����������� 
	std::vector<float> currentClustersResponseVar;//��¼��ǰ�ص�y�ķ���
	std::vector<std::vector<float>> lpFeature; 
	std::vector<std::pair<int, float>> testToMid;//���������������ĵ�ľ���.first-�ر�ţ�second-������ĵľ���
	 
 
	float maxCurResponseVar = maxResponseOfTree;
	oldMeanCenter.resize(TreeNum);
	int j = voFeatureSet.size() - TreeNum;
	int oldIndex = 0;
	std::vector<std::vector<float>> vAllLeafFeatureSetChanged;
	//���������л���
	for (int j = 0;j < voFeatureSet[0].size();j++)
	{
		std::vector<float> vAllLeafFeatureRow;
		for (int i = 0;i < voFeatureSet.size();i++)
		{
			vAllLeafFeatureRow.push_back(voFeatureSet[i][j]);
		}
		vAllLeafFeatureSetChanged.push_back(vAllLeafFeatureRow);
	}
	//������������-ά��-��
	for (int j = 0;j < vAllLeafFeatureSetChanged.size();j++)
	{
		std::sort(vAllLeafFeatureSetChanged[j].begin(), vAllLeafFeatureSetChanged[j].end());
	}

	//�����׼��
	std::vector<float> voEachDimStandard;
	__calStandardDeviation(voFeatureSet, voEachDimStandard);

	for (;j < voFeatureSet.size();j++)
	{

		oldMeanCenter[oldIndex].assign(voFeatureSet[j].begin(), voFeatureSet[j].end());
		oldIndex++;
	}
	int iters = 0;
	
	do
	{
		oldClustersFeature.clear();
		clustersFeature.clear();
		lpFeature.clear(); 
		clustersFeature.resize(oldMeanCenter.size());
		//�������������ֵ�����ľ���
		for (int k = 0;k < voFeatureSet.size() - TreeNum - 1;k++)
		{
			std::vector<float> lpToMeanFeat;
			for (int j = 0;j < oldMeanCenter.size();j++)
			{
				//������Ե㵽��ֵ��ľ���-LP,�ɳ���MP����
				float lp = calLp(voFeatureSet[k], oldMeanCenter[j]);
				//float lp = calMp(voFeatureSet[k], oldMeanCenter[j], voFeatureSet, vAllLeafFeatureSetChanged,voEachDimStandard);
				lpToMeanFeat.push_back({ lp });

			}
			lpFeature.push_back({ lpToMeanFeat });
		}

		 
		//���ݾ��룬�����������ڵĴ�
		for (int i = 0;i < lpFeature.size();i++)
		{
			int minLpIndex = 0;
			for (int j = 1;j < lpFeature[0].size();j++)
			{
				if (lpFeature[i][minLpIndex]> lpFeature[i][j])
				{
					minLpIndex = j;
				}


			}
			Clusters[i] = minLpIndex;

		}

		for (int k = 0;k < oldMeanCenter.size();k++)
		{
			for (int i = 0;i < Clusters.size();i++)
			{

				if (k == Clusters[i])
				{
					/*std::vector<float> tempFeat = ;
					std::vector<std::vector<float>> */
					clustersFeature[k].push_back({ voFeatureSet[Clusters[i]] });//�������ڴصı�ţ����ִص�����
				}

			}
		}
 
		//���¼�����ص�bias----------------------
		//����ÿ���ص�y�����ֵ��ƫ��,�������ƫ�� 
		currentClustersResponseVar = calResponseVar(clustersFeature.size(), Clusters, TotalNodeDataResponseSet, maxCurResponseVar); 
	
		//���¼�����صľ�ֵ����
		for (int k = 0;k < clustersFeature.size();k++)
		{
			std::vector<float> meanFeatVector;
			for (int j = 0;j < voFeatureSet[0].size();j++)
			{
				float sumOfFeat = 0.0f;
				for (int i = 0;i < clustersFeature[k].size();i++)
				{
					sumOfFeat += clustersFeature[k][i][j];
				}
				if (clustersFeature[k].size() != 0)//��ذ�������
				{
					sumOfFeat /= clustersFeature[k].size();
				}
				else//��ز���������
				{
					sumOfFeat = FLT_MAX;
				}
				meanFeatVector.push_back(sumOfFeat);
			}
			currentMeanCenter.push_back({ meanFeatVector });//��¼�µĴ����ĵ�����
		}
 
		//�ж����ص�ƫ���Ƿ�С���޶�����ֵ-,�ǣ��˳�ѭ��----------------------------
		if (maxCurResponseVar<maxBiasRatio)
		{   
			//�������ĵ�
			for (int i = 0;i < currentMeanCenter.size();i++)
			{
				//��ͬ�����¸���
				oldMeanCenter[i].assign(currentMeanCenter[i].begin(), currentMeanCenter[i].end());
			}
			break;
		}

		int breakFlag = 0;
		//�жϴص�y�ķ����Ƿ�仯-------
		  for (int i = 0;i < currentClustersResponseVar.size();i++)
		  {
			 
				if (currentClustersResponseVar[i] < oldClustersResponseVar[i] || currentClustersResponseVar[i]== FLT_MAX)//��ǰƫ��С����һ�֣���������
				{
					//��ͬ�����¸���
					oldMeanCenter[i].assign(currentMeanCenter[i].begin(), currentMeanCenter[i].end());
					oldClustersResponseVar[i]= currentClustersResponseVar[i];//����bias
				 } 
				else
				{
					 
					breakFlag++;
					 
				}
				 
			}
 
		
		if (breakFlag == oldMeanCenter.size())//�����ǰ��ֵ������û�и��£��˳�ѭ��
		{
			break;
		}
		iters++;
		 
	} while (iters < iterTime);//�ﵽ�����������˳�ѭ��
							   //����������������ĸ��أ��������������������ĵľ��� 

	for (int k = 0;k < oldMeanCenter.size();k++)
	{
		 //���ж�oldMeanCenter�Ƿ�ȫΪFLTMAX,�ǣ�����Ϊ���޴�
		if (oldMeanCenter[k][0]!= FLT_MAX)
		{
			float lp = calLp(voFeatureSet[voFeatureSet.size() - TreeNum - 1], oldMeanCenter[k]);//LP����
	     	//float lp = calMp(voFeatureSet[voFeatureSet.size() - TreeNum - 1], oldMeanCenter[k], voFeatureSet, vAllLeafFeatureSetChanged,voEachDimStandard);//MP����
			testToMid.push_back({ k,lp });
		}
	/*	else
		{
			testToMid.push_back({ k,FLT_MAX });
		}*/
	
	}
	std::sort(testToMid.begin(), testToMid.end(), comparePair);
	return testToMid;
}


//�����������ĵ�Ϊ���޴�
void  CTrainingSet::setCenterFLTMAX(std::vector<float>& FeatureCenterMax)
{
	for (int i = 0;i < FeatureCenterMax.size();i++)
	{

		FeatureCenterMax[i] = FLT_MAX;
	}

}

 
//����ÿ���ص�y�ķ���-----------
std::vector<float>  CTrainingSet::calResponseVar(int  currentClusterNum, std::vector<int>&  Clusters, const std::vector<float>& TotalNodeDataResponseSet,float& maxResponseVar)
{
	std::vector<float> currentClustersVar(currentClusterNum,0.0f);
	std::vector<float> meanOfClusters(currentClusterNum, 0.0f);
	meanOfClusters = calMeanOfClusters(currentClusterNum, Clusters, TotalNodeDataResponseSet);//ÿ���صľ�ֵ
	float maxCurrentVar = 0.0f;
	for (int i = 0;i < meanOfClusters.size();i++)
	{
		float sumOfCurrentClusterVar = 0.0;
		int countOfCurClustersVar = 0;
		for (int j = 0;j < Clusters.size();j++)
		{
			if (Clusters[j] == i && meanOfClusters[i]!=0)
			{
				sumOfCurrentClusterVar +=pow((TotalNodeDataResponseSet[j]- meanOfClusters[i]),2);
				countOfCurClustersVar++;
			}
		}
		if (countOfCurClustersVar != 0)
		{
			currentClustersVar[i] = sumOfCurrentClusterVar / countOfCurClustersVar;
			if (currentClustersVar[i]>maxCurrentVar)
				maxCurrentVar = currentClustersVar[i];
		}
		else
		{
			currentClustersVar[i] =FLT_MAX; //Ĭ�������֯
		}
	
	}
	 
	maxResponseVar = maxCurrentVar;
	return currentClustersVar;
}

//����ÿ���ص�y�ľ�ֵ-----------------------
std::vector<float> CTrainingSet::calMeanOfClusters(int currentClusterNum, std::vector<int>&  Clusters, const std::vector<float>& TotalNodeDataResponseSet)
{
	std::vector<float> meanOfCurCls(currentClusterNum,0.0f);
	for (int i = 0;i< currentClusterNum;i++)
	{
		 
		float sumOfCurrentCluster = 0.0;
		int countOfCurClusters = 0;
		for (int j = 0;j < Clusters.size();j++)
		{

			if (Clusters[j] == i)
			{
				sumOfCurrentCluster += TotalNodeDataResponseSet[j];
				countOfCurClusters++;
			}
		}
		if (countOfCurClusters != 0)
		{
			meanOfCurCls[i] = sumOfCurrentCluster / countOfCurClusters;
		}
		else
		{
			meanOfCurCls[i] = 0.0f;
		}
	}
	return meanOfCurCls;
}
//�ж����������������Ƿ���ͬ,��ͬ����-True
//-�������------------------------
std::vector<bool> CTrainingSet::isSameVandV(std::vector<std::vector<float>> currentMeanCenter,std::vector<std::vector<float>> oldMeanCenter)
{
	std::vector<bool> cmpFlag(currentMeanCenter.size(),false);
	for (int i = 0;i < currentMeanCenter.size();i++)
	{
		int count = 0;
		for (int j = 0;j < currentMeanCenter[0].size();j++)
		{
			if (currentMeanCenter[i][j] == oldMeanCenter[i][j])
				count++;
			else
				break;
		}
		if (count == currentMeanCenter[0].size())
		{
			cmpFlag[i]=true;
	   }
	}

	return cmpFlag;
}


//�ж��������������Ƿ���ͬ,��ͬ����-True
//-�������------------------------
bool CTrainingSet::isSameVandV(std::vector<float> currentMeanCenter, std::vector<float> oldMeanCenter)
{
	 
	bool cmpFlag = false;
		int count = 1;
		for (int j = 0;j < currentMeanCenter.size();j++)
		{
			if (currentMeanCenter[j] == oldMeanCenter[j])
				count++;
			else
				break;
		}
		if (count == currentMeanCenter.size())
		{
			cmpFlag=true;
		}
	 

	return cmpFlag;
}
//-���������������֮���LP����-----------------------
float CTrainingSet::calLp(std::vector<float> voFeature, std::vector<float> objFeature)
{
	float LPParam = 2.0f;
	float LPValueSum = 0.0f;
	float SumLP = 0.0f;
	for (int k = 0; k < voFeature.size(); k++)
	{
		LPValueSum += pow(voFeature[k] - objFeature[k], LPParam);
	}
	SumLP += pow(LPValueSum, 1.0 / LPParam);
	return SumLP;
}




//-���������������֮���MP����,������Ϊ����Ҷ�ӽڵ������-----------------------
float CTrainingSet::calMp(std::vector<float> voFeature, std::vector<float> objFeature, std::vector<std::vector<float>>& vAllLeafFeatureSet, std::vector<std::vector<float>> vAllLeafFeatureSetChanged, std::vector<float> voEachDimStandard)
{
	float MPParam = 2.0f; 
	std::vector<std::pair<float, float>> MaxMinValue; 
	
	//����Test�����ĵ��MP
	std::vector<int> EachFeatureIntervalCount(voFeature.size(), 0);
	float MPValueSum = 0.f; 

	for (int j = 0; j < voFeature.size(); j++)
	{
		MaxMinValue.push_back({ std::max(voFeature[j], objFeature[j]) + voEachDimStandard[j], std::min(voFeature[j], objFeature[j]) - voEachDimStandard[j] });//��ʱȥ����׼��,��Ϊ���ϱ�׼�Χ̫��
	}
	
	
	__calIntervalSampleByMid(MaxMinValue, vAllLeafFeatureSetChanged, EachFeatureIntervalCount);

	for (int k = 0; k < EachFeatureIntervalCount.size(); k++)//����ÿһά
	{
		MPValueSum += pow(((float)EachFeatureIntervalCount[k] / vAllLeafFeatureSet.size())/**(((ResponseRange[k].first - ResponseRange[k].second) / AllREsponseRange))*/, MPParam);
	}
	 
	return pow(MPValueSum, 1 / MPParam);
}

//��һ��--��׼��--------------------------------
void CTrainingSet::standard(std::vector<std::vector<float>>& voFeatureSet)
{
	_ASSERT(voFeatureSet.size() > 0);
	std::vector<float> FeatureMeans(voFeatureSet[0].size(), 0.0f);
	std::vector<float> FeatureStd(voFeatureSet[0].size(), 0.0f);
	for (int i = 0; i < voFeatureSet[0].size(); i++)//��
	{
		for (int k = 0; k < voFeatureSet.size(); k++)
		{
			FeatureMeans[i] += voFeatureSet[k][i];
		}
		FeatureMeans[i] /= voFeatureSet.size();//��ֵ
		for (int k = 0; k < voFeatureSet.size(); k++)
		{
			FeatureStd[i] += (voFeatureSet[k][i] - FeatureMeans[i])*(voFeatureSet[k][i] - FeatureMeans[i]);
		}
		FeatureStd[i] = std::sqrt(FeatureStd[i] / voFeatureSet.size());//��׼��
	}

	for (int i = 0; i < voFeatureSet[0].size(); i++)
	{
		for (int k = 0; k < voFeatureSet.size(); k++)
		{
			if (FeatureStd[i] == 0)
				FeatureStd[i] = 1.0f;
			voFeatureSet[k][i] = (voFeatureSet[k][i] - FeatureMeans[i]) / FeatureStd[i];
		}
	}

}


 
//******************************************************************************
//FUNCTION:����MP�����,ȫ��
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
//FUNCTION:�ֲ�
std::vector<std::pair<float, float>>  CTrainingSet::calLeafAndBotherMP(const std::vector<std::vector<float>>& vLeafFeatureSet, const std::vector<float>& vLeafResponseSet, 
	                                                                   const std::vector<std::vector<float>>& vBotherFeatureSet, const std::vector<float>& vBotherResponseSet,
	                                                                   const std::vector<std::vector<float>>& vAllLeafFeatureSet, const std::vector<float>& vAllLeafResponseSet,
	                                                                   const std::vector<float>& vFeatures, float vPredictResponse)
{
	float MPParam = 2.0f;
	std::vector<std::pair<float, float>> leafMPSet;//first Ϊѵ������Yֵ��second ΪMPֵ
	std::vector<std::pair<float, float>> BotherMPSet;//first Ϊѵ������Yֵ��second ΪMPֵ
	std::vector<std::pair<float, float>> AllMPSet;//first Ϊѵ������Yֵ��second ΪMPֵ
											   //�������ֵ����
	std::vector<float> LeafFeatureMid(vLeafFeatureSet[0].size(), 0);
	std::vector<float> BotherFeatureMid(vBotherFeatureSet[0].size(), 0);

	float LeafResponseMid = 0.0f;
	float BotherResponseMid = 0.0f;
	//Ҷ��ά�����ĵ�
	for (int j = 0;j < vLeafFeatureSet[0].size();j++)//��
	{
		for (int i = 0; i < vLeafFeatureSet.size(); i++)//��
		{
			LeafFeatureMid[j] += vLeafFeatureSet[i][j];
		}
		LeafFeatureMid[j] = LeafFeatureMid[j] / vLeafFeatureSet.size();//ÿ��/ά�ȵľ�ֵ
	}
	//Ҷ�ӵ�y��ֵ
	for (int j = 0;j < vLeafResponseSet.size();j++)//��
	{
		LeafResponseMid += vLeafResponseSet[j];
	}
	LeafResponseMid = LeafResponseMid / vLeafResponseSet.size();

	 
	//�ֵ�ά�����ĵ�
	for (int j = 0;j < vBotherFeatureSet[0].size();j++)//��
	{
		for (int i = 0; i < vBotherFeatureSet.size(); i++)//��
		{
			BotherFeatureMid[j] += vBotherFeatureSet[i][j];
		}
		BotherFeatureMid[j] = BotherFeatureMid[j] / vBotherFeatureSet.size();//ÿ��/ά�ȵľ�ֵ
	}
	//�ֵܵ�y��ֵ
	for (int j = 0;j < vBotherResponseSet.size();j++)//��
	{
		BotherResponseMid += vBotherResponseSet[j];
	}
	BotherResponseMid= BotherResponseMid/ vBotherResponseSet.size();


	std::vector<std::vector<float>> vAllLeafFeatureSetChanged;//����

	//���������л���
	for (int j = 0;j < vAllLeafFeatureSet[0].size();j++)
	{
		std::vector<float> vAllLeafFeatureRow;
		for (int i = 0;i < vAllLeafFeatureSet.size();i++)
		{
			vAllLeafFeatureRow.push_back(vAllLeafFeatureSet[i][j]);
		}
		vAllLeafFeatureSetChanged.push_back(vAllLeafFeatureRow);
	}
	//������������-ά��-��
	for (int j = 0;j < vAllLeafFeatureSetChanged.size();j++)
	{
		std::sort(vAllLeafFeatureSetChanged[j].begin(), vAllLeafFeatureSetChanged[j].end());
	}


	leafMPSet = calMPwithMidPoint(LeafFeatureMid,  vFeatures, vAllLeafFeatureSetChanged, vAllLeafFeatureSet,vAllLeafResponseSet, LeafResponseMid);

	BotherMPSet = calMPwithMidPoint(BotherFeatureMid, vFeatures, vAllLeafFeatureSetChanged, vAllLeafFeatureSet,vAllLeafResponseSet, BotherResponseMid);
	
	AllMPSet.push_back({ leafMPSet[0].first,leafMPSet[0].second });
	AllMPSet.push_back({ BotherMPSet[0].first,BotherMPSet[0].second });
	return AllMPSet;


}

float  CTrainingSet::calLeafWeightForTreeByLp(std::vector<std::vector<float>> NodeDataFeature, const std::vector<float>& vFeatures)
{
	//Ҷ��ά�����ĵ�
	float LPParam = 2.0f;
	float LPValueSum = 0.0f;
	float LPValue = 0.0f;
	std::vector<float> NodeFeatureMid(NodeDataFeature[0].size(), 0);
	for (int j = 0;j < NodeDataFeature[0].size();j++)//��
	{
		for (int i = 0; i < NodeDataFeature.size(); i++)//��
		{
			NodeFeatureMid[j] += NodeDataFeature[i][j];
		}
		NodeFeatureMid[j] = NodeFeatureMid[j] / NodeDataFeature.size();//ÿ��/ά�ȵľ�ֵ
	}

	for (int i = 0; i < NodeFeatureMid.size(); ++i)
	{
		LPValueSum += pow(NodeFeatureMid[i] - vFeatures[i], LPParam);
	}

	LPValue= pow(LPValueSum, 1 / LPParam);
	return LPValue;
}

//KMeans����ɭ�ֵ�Ҷ�Ӿ���(LP)��ͨ�����ĵ��ȶ��������������������صľ�ֵ����ΪԤ����-----------------
float CTrainingSet::calKMeansForest(std::vector<std::vector<float>>  treeNodeFeatureMid, std::vector<float> treeNodeDataResponse,
	                               std::vector<std::vector<float>> TotalNodeDataFeature, const std::vector<float>& TotalNodeDataResponseSet, const std::vector<float>& vFeatures)
{
	std::vector<std::vector<float>> allNodeAndFeature; 
	float predictValueByKmeans = 0.0f;
	for (int i = 0;i < TotalNodeDataFeature.size();i++)
	{ 
		  
	  allNodeAndFeature.push_back({ TotalNodeDataFeature[i] }); 
	  
	}
	
	//allNodeAndFeature.erase(unique(allNodeAndFeature.begin(), allNodeAndFeature.end()), allNodeAndFeature.end());
	allNodeAndFeature.push_back({ vFeatures });
	for (int i = 0;i < treeNodeFeatureMid.size();i++)
	{
		allNodeAndFeature.push_back({ treeNodeFeatureMid[i] });
	}
	//��һ��-��������Ҷ�������������������������ĵ�
	//standard(allNodeAndFeature);
	//Kmeans
	int definedClusterNum = treeNodeFeatureMid.size();
	int clusterNumByNodeNum = allNodeAndFeature.size();
	int clusterK = definedClusterNum>clusterNumByNodeNum?clusterNumByNodeNum: definedClusterNum;
	int iterTime = 30;
	std::vector<int> clusters(allNodeAndFeature.size() - definedClusterNum-1, 0);//��¼�������ڵĴ��±�
	//std::vector<std::pair<int, float>> testClusters=kMeans(allNodeAndFeature, treeNodeFeatureMid.size(), clusterK,iterTime, clusters);
	std::vector<std::pair<int, float>> testClusters = kMeansByDyCls(allNodeAndFeature, treeNodeFeatureMid.size(), clusterK, iterTime, clusters);
	
	//����KMmeans��Ԥ��ֵ---- ����1������test����Ĵ��е��������ֵ----------------------------------------------------
	std::vector<std::pair<float, float>> predValueOfKmeans;//first--LP,second--value;
	
	for (int j=0;j<testClusters.size();j++)
	{ 
		int countSample = 0;
		float sumOfCluster = 0.0f;
		for (int i = 0;i < clusters.size();i++)
		{
			int sampleClusterIndex = clusters[i];//�������ڵĴص��±� 

			if (sampleClusterIndex == testClusters[j].first)
			{
				countSample++; 
				sumOfCluster += TotalNodeDataResponseSet[i];
			}

		}
		if (countSample != 0)
		{
			predValueOfKmeans.push_back({ testClusters[j].second,sumOfCluster/countSample }) ;
		}
			 
   }
	//����1:���������ƵĽ��-------------------------------------------------------------------------------------
	//return predValueOfKmeans[0].second;
   
//����2����Ȩ����------------------------------------------------------------------------------------------------
	std::vector<float> distance(predValueOfKmeans.size(),0.0f);
	std::vector<float> wi(predValueOfKmeans.size(), 0.0f);
	float predict = 0.0f;
	for (int i = 0;i < predValueOfKmeans.size();i++)
	{
		distance[i] = predValueOfKmeans[i].first;
	}
	wi = antiDistanceWeight(distance, -2);
	for (int i = 0;i < predValueOfKmeans.size();i++)
	{
		predict+= predValueOfKmeans[i].second * wi[i];
	}

	return predict;
}



//KMeans����ɭ�ֵ�Ҷ�Ӿ���(LP)��ͨ��Y��bais�ʴﵽ��ֵ�� �����������������صľ�ֵ����ΪԤ����-----------------
float CTrainingSet::calKMeansForestByResponseVarRatio(std::vector<float>& treeResponseVar, float maxResponseOfTree,float maxVarRatio,std::vector<std::vector<float>>  treeNodeFeatureMid, std::vector<float> treeNodeDataResponse,
	std::vector<std::vector<float>> TotalNodeDataFeature, const std::vector<float>& TotalNodeDataResponseSet, const std::vector<float>& vFeatures)
{
	std::vector<std::vector<float>> allNodeAndFeature;
	float predictValueByKmeans = 0.0f;
	for (int i = 0;i < TotalNodeDataFeature.size();i++)
	{

		allNodeAndFeature.push_back({ TotalNodeDataFeature[i] });

	}

	//allNodeAndFeature.erase(unique(allNodeAndFeature.begin(), allNodeAndFeature.end()), allNodeAndFeature.end());
	allNodeAndFeature.push_back({ vFeatures });
	for (int i = 0;i < treeNodeFeatureMid.size();i++)
	{
		allNodeAndFeature.push_back({ treeNodeFeatureMid[i] });
	}
	//��һ��-��������Ҷ�������������������������ĵ�
	//standard(allNodeAndFeature);
	//Kmeans
	int definedClusterNum = treeNodeFeatureMid.size();
	int clusterNumByNodeNum = allNodeAndFeature.size();
	int clusterK = definedClusterNum>clusterNumByNodeNum ? clusterNumByNodeNum : definedClusterNum;
	int iterTime = 30;
	std::vector<int> clusters(allNodeAndFeature.size() - definedClusterNum - 1, 0);//��¼�������ڵĴ��±�
																				   //std::vector<std::pair<int, float>> testClusters=kMeans(allNodeAndFeature, treeNodeFeatureMid.size(), clusterK,iterTime, clusters);
	std::vector<std::pair<int, float>> testClusters = kMeansByResponseVarRatio(treeResponseVar, maxResponseOfTree, maxVarRatio, TotalNodeDataResponseSet, allNodeAndFeature, treeNodeFeatureMid.size(), clusterK, iterTime, clusters);
	                                                 
	//����KMmeans��Ԥ��ֵ---- ����1������test����Ĵ��е��������ֵ----------------------------------------------------
	std::vector<std::pair<float, float>> predValueOfKmeans;//first--LP,second--value;
	std::ofstream clusterInfo;
	clusterInfo.open("clusterInfo.csv", std::ios::app);

	for (int i = 0;i < clusters.size();i++)
	{
		clusterInfo << i << ",";
		for (int j = 0;j < vFeatures.size();j++)
		{
			clusterInfo << vFeatures[j] << ",";
		}
		clusterInfo << clusters[i] << ",";//�غ�
		for (int j = 0;j < TotalNodeDataFeature[0].size();j++)
		{
			clusterInfo << TotalNodeDataFeature[i][j] << ",";
		}
		clusterInfo << TotalNodeDataResponseSet[i]<<std::endl;
	}


	for (int j = 0;j<testClusters.size();j++)
	{
	 
			int countSample = 0;
			float sumOfCluster = 0.0f;
			for (int i = 0;i < clusters.size();i++)
			{
				int sampleClusterIndex = clusters[i];//�������ڵĴص��±� 

				if (sampleClusterIndex == testClusters[j].first)
				{
					countSample++;
					sumOfCluster += TotalNodeDataResponseSet[i];
				}

			}
			if (countSample != 0)
			{
				predValueOfKmeans.push_back({ testClusters[j].second,sumOfCluster / countSample });
			}


	}
	//����1:���������ƵĽ��-------------------------------------------------------------------------------------
	//return predValueOfKmeans[0].second;

	//����2����Ȩ����------------------------------------------------------------------------------------------------
	std::vector<float> distance(predValueOfKmeans.size(), 0.0f);
	std::vector<float> wi(predValueOfKmeans.size(), 0.0f);
	float predict = 0.0f;
	for (int i = 0;i < predValueOfKmeans.size();i++)
	{
		distance[i] = predValueOfKmeans[i].first;
	}
	wi = antiDistanceWeight(distance, -2);
	for (int i = 0;i < predValueOfKmeans.size();i++)
	{
		predict += predValueOfKmeans[i].second * wi[i];
	}

	return predict;
}


//���㵱ǰҶ�ӽڵ���Features�ķ���
std::vector<float> CTrainingSet::calTreeFeaturesVar(std::vector<std::vector<float>>& NodeDataFeature)
{
	std::vector<float> meanOfFeats(NodeDataFeature[0].size(), 0.0f);
	std::vector<float> meanOfFeatsVar(NodeDataFeature[0].size(), 0.0f);
	for (int j = 0;j < NodeDataFeature[0].size();j++)//��
	{
		float sumOfFet = 0.0f;
		for (int i = 0;i< NodeDataFeature.size();i++)//��
		{
			sumOfFet += NodeDataFeature[i][j];
		}
		meanOfFeats[j] = sumOfFet / NodeDataFeature.size();//ÿ�������ľ�ֵ
	}

	for (int j = 0;j < NodeDataFeature[0].size();j++)//��
	{
		float sumOfFetVar = 0.0f;
		//�󷽲�-------------------------
		for (int i = 0;i< NodeDataFeature.size();i++)//��
		{
			sumOfFetVar += pow((NodeDataFeature[i][j] - meanOfFeats[j]), 2);
		}

		if (sumOfFetVar != 0)
		{
			meanOfFeatsVar[j] = sumOfFetVar / NodeDataFeature.size();
		}
		else
		{
			meanOfFeatsVar[j] = FLT_MIN;//����=0
		}
	}

	return meanOfFeatsVar;


}
//���㵱ǰҶ�ӽڵ���Features�ķ���-Ͷ����Ե㣬���㷽��仯��-�仯�ʴ󣬴������ƶȵͣ���֮�仯��С���������ƶȸߣ��仯�ʼ����¼���ķ�����֮ǰ����֮��
std::vector<float> CTrainingSet::calTreeFeaturesVar(const std::vector<float>& vFeatures, std::vector<std::vector<float>>& NodeDataFeature, std::vector<float>& FeatsVarChangedRatio)
{
	std::vector<float> meanOfFeats(NodeDataFeature[0].size(),0.0f);//�ڵ�ԭ�������ľ�ֵ
	std::vector<float> FeatsVar(NodeDataFeature[0].size(), 0.0f);//�ڵ�ԭ�������ķ���
	std::vector<float> meanOfAddTestFeats(NodeDataFeature[0].size(), 0.0f);//������Ե��������ľ�ֵ
	std::vector<float> AddTestFeatsVar(NodeDataFeature[0].size(), 0.0f);//������Ե��������ķ���
	 
	for (int j = 0;j < NodeDataFeature[0].size();j++)//��
	{
		float sumOfFet = 0.0f;
		for (int i= 0;i< NodeDataFeature.size();i++)//��
		{
			sumOfFet += NodeDataFeature[i][j];
		}
		
		meanOfFeats[j] = sumOfFet / NodeDataFeature.size();//ÿ�������ľ�ֵ
		sumOfFet += vFeatures[j];// Ͷ�� ���Ե������
		meanOfAddTestFeats[j] = sumOfFet / (NodeDataFeature.size() + 1);//������Ե��������ľ�ֵ
	}

	for (int j = 0;j < NodeDataFeature[0].size();j++)//��-����
	{
		float sumOfFetVar = 0.0f;
		float sumOfAddTestFetVar = 0.0f;
		//�󷽲�-------------------------
		for (int i = 0;i< NodeDataFeature.size();i++)//��
		{
			sumOfFetVar += pow((NodeDataFeature[i][j] - meanOfFeats[j]), 2);
			sumOfAddTestFetVar += pow((NodeDataFeature[i][j] - meanOfAddTestFeats[j]), 2);
		}
		//����Ե�
		sumOfAddTestFetVar += pow((vFeatures[j] - meanOfAddTestFeats[j]), 2);
		
		if (sumOfFetVar != 0)
		{
			FeatsVar[j] = sumOfFetVar / NodeDataFeature.size();
			 
		}
		else
		{
			FeatsVar[j] = FLT_MIN;//����=0
			 
		}

		if (sumOfAddTestFetVar != 0)
		{
			 
			AddTestFeatsVar[j] = sumOfAddTestFetVar / (NodeDataFeature.size() + 1);
		}
		else
		{
			AddTestFeatsVar[j] = FLT_MIN;//����=0

		}

		if (FeatsVar[j] == FLT_MIN && AddTestFeatsVar[j] == FLT_MIN)
		{
			FeatsVarChangedRatio[j] = FLT_MIN;
		}
		else if(FeatsVar[j]== AddTestFeatsVar[j])
		{
			FeatsVarChangedRatio[j] = FLT_MIN;
		}
		else
		{
			FeatsVarChangedRatio[j] = abs(FeatsVar[j] - AddTestFeatsVar[j]);
		}


	}

	 
	return FeatsVar;
 

}
 
//���㵱ǰҶ�ӽڵ���Y�ķ���
float CTrainingSet::calTreeResponseVar(std::vector<float> NodeDataResponse)
{

	float sumOfResponse=0.0f;
	float meanOfResponse = 0.0f;
	float sumOfVar = 0.0f;
	float meanOfVar = 0.0f;
	
		for (int i = 0;i < NodeDataResponse.size();i++)
		{
			sumOfResponse += NodeDataResponse[i];

		}
		meanOfResponse = sumOfResponse / NodeDataResponse.size();
		for (int i = 0;i < NodeDataResponse.size();i++)
		{
			sumOfVar += pow((NodeDataResponse[i] - meanOfResponse), 2);

		}
		if (sumOfVar != 0)
		{
			meanOfVar = sumOfVar / NodeDataResponse.size();
		}
		else
		{
			meanOfVar =FLT_MIN;//����Ϊ0
		}
	 
	 
		
	return meanOfVar;
}
//�������Ȩ���㷨---------------------------
std::vector<float> CTrainingSet::antiDistanceWeight(std::vector<float> distance,int p)
{
	float sumOfWi = 0.0f;
	std::vector<float> wi(distance.size(),0.0f); 
	for (int i = 0;i < distance.size();i++)
	{ 
		if (distance[i] == 0)//������ھ���Ϊ0��������Ȩ��Ϊ1������wi
		{
			wi[i] = 1;
			return wi;
		}
		else
		{
			sumOfWi += pow(distance[i], p);
		}
	   
		
	}
	for (int i = 0;i < distance.size();i++)
	{
		wi[i]=pow(distance[i], p)/sumOfWi;
	}
	return wi;
}

//���������Ȩ����
std::vector<std::vector<float>>  CTrainingSet::antiFeatsWeight(std::vector<std::vector<float>> treeNodeFeatureVar, int p)
{
	std::vector<std::vector<float>>  wi;//ÿ��Ҷ���е�ÿ������
	std::vector<float> sumOfFetWi(treeNodeFeatureVar[0].size());//ÿ������֮��
	std::vector<float> minOfFet(treeNodeFeatureVar[0].size(), FLT_MAX);//ÿ����������Сֵ 

 
	//����Сֵ
	for (int i = 0;i < treeNodeFeatureVar[0].size();i++)//��-����
	{
		for (int j = 0;j < treeNodeFeatureVar.size();j++)//��-��
		{
			if (treeNodeFeatureVar[j][i] != FLT_MIN&&treeNodeFeatureVar[j][i] !=0.0f)
			{
				if (minOfFet[i] > treeNodeFeatureVar[j][i])
				{
					minOfFet[i] = treeNodeFeatureVar[j][i];
				}

			}
		}
		minOfFet[i] = (minOfFet[i] /100000)*99999;
	}
	 
  //����Ȩ��
  for (int i = 0;i < treeNodeFeatureVar[0].size();i++)//ÿ������-��
  {
	  
	  for (int k = 0;k<treeNodeFeatureVar.size();k++)//ÿ��Ҷ��-��
	  {
		  if (treeNodeFeatureVar[k][i] == FLT_MIN)
		  {
			  treeNodeFeatureVar[k][i] = abs(pow(minOfFet[i], p));//�ø��е��������Сֵ����0����

		  }
		  else
		  {
			  treeNodeFeatureVar[k][i] = abs(pow(treeNodeFeatureVar[k][i], p));
		  }
		  sumOfFetWi[i] += treeNodeFeatureVar[k][i];//����ÿ�е�Ȩ��֮��
	  }
	 
  }
  
  for (int k = 0;k < treeNodeFeatureVar.size();k++)//ÿ��Ҷ��-��
  {
	  std::vector<float> wiOfFeat;
	  for (int i = 0;i < treeNodeFeatureVar[0].size();i++)
	  {
		  wiOfFeat.push_back(treeNodeFeatureVar[k][i] / sumOfFetWi[i]);//����ÿ�����ĳ������ֵ��ռ����Ȩ�ر���
	  }
	  wi.push_back(wiOfFeat);
  }
	return wi;
}
//�����Ȩ����--------------
std::vector<float> CTrainingSet::antiVarWeight(std::vector<float> responseVar, int p)
{
	float sumOfWi = 0.0f;
	std::vector<float> wi(responseVar.size(), 0.0f); 
	float minVar = FLT_MAX;
	//����С����-��0�������С����
	for (int i = 0;i < responseVar.size();i++)
	{ 
		if (responseVar[i]!= FLT_MIN)
		{ 
			if (minVar > responseVar[i])
			{
				minVar = responseVar[i];
			} 
			
		} 
	}
	minVar = (minVar / 100000) * 99999;//�趨һ����ʵ����Сֵ��Сһ�������ֵ����0����Ա����

	//�ۼ�
	for (int i = 0;i < responseVar.size();i++)
	{
		if (responseVar[i] ==FLT_MIN)
		{
			responseVar[i] = pow(minVar,p);
		}
		else
		{
			responseVar[i] = pow(responseVar[i], p);
		}
		sumOfWi += responseVar[i];
	}
	

	for (int i = 0;i < responseVar.size();i++)
	{
		wi[i] = responseVar[i] / sumOfWi;
	}
	return wi;
}
std::vector<std::pair<float, float>>   CTrainingSet::calLeafWeightForTreeByMp(std::vector<std::vector<float>>  treeNodeFeatureMid, std::vector<float> treeNodeDataResponse, 
	                                                                          std::vector<std::vector<float>> TotalNodeDataFeature, const std::vector<float>& TotalNodeDataResponseSet, const std::vector<float>& vFeatures)
{
	float MPParam = 2.0f;
	std::vector<std::pair<float, float>> MPSet;//first Ϊ���ĵ��Ԥ��Yֵ��second Ϊ���Ե㵽���ĵ��MPֵ   
	std::vector<std::vector<float>> vAllLeafFeatureSetChanged;//����

     //���л���
	for (int j = 0;j < TotalNodeDataFeature[0].size();j++)
	{
		std::vector<float> vAllLeafFeatureRow;
		for (int i = 0;i < TotalNodeDataFeature.size();i++)
		{
			vAllLeafFeatureRow.push_back(TotalNodeDataFeature[i][j]);
		}
		vAllLeafFeatureSetChanged.push_back(vAllLeafFeatureRow);
	}
	//��������-ά��-��
	for (int j = 0;j < vAllLeafFeatureSetChanged.size();j++)
	{
		std::sort(vAllLeafFeatureSetChanged[j].begin(), vAllLeafFeatureSetChanged[j].end());
	}

	for (int i = 0; i < treeNodeFeatureMid.size(); i++)//ÿ������Ҷ�����ĵ�
	{
		std::vector<float> TrainFeatureValue = treeNodeFeatureMid[i];
		std::vector<std::pair<float, float>> MaxMinValue;
		std::vector<int> EachFeatureIntervalCount(TotalNodeDataFeature[i].size(), 0);
		std::vector<std::pair<float, float>> ResponseRange(TotalNodeDataFeature[i].size(), { 0.0f, 0.0f });
		float MPValueSum = 0.f;

		//���Ե㵽Ҷ�����ĵ�ķ�Χ
		for (int j = 0; j < TrainFeatureValue.size(); j++)
		{
			MaxMinValue.push_back({ std::max(TrainFeatureValue[j], vFeatures[j]), std::min(TrainFeatureValue[j], vFeatures[j]) });//��ʱȥ����׼��,��Ϊ���ϱ�׼�Χ̫��
		}

	   //��������Ҷ�������ڸ÷�Χ�ڵ���������
		__calIntervalSampleByOrder(MaxMinValue, vAllLeafFeatureSetChanged, TotalNodeDataResponseSet, EachFeatureIntervalCount, ResponseRange);

		for (int k = 0; k < EachFeatureIntervalCount.size(); k++)//����ÿһά
		{
			MPValueSum += pow(((float)EachFeatureIntervalCount[k] / TotalNodeDataResponseSet.size())/**(((ResponseRange[k].first - ResponseRange[k].second) / AllREsponseRange))*/, MPParam);
		}


		MPSet.push_back({ treeNodeDataResponse[i], pow(MPValueSum, 1 / MPParam) });


	}
	std::sort(MPSet.begin(), MPSet.end(), comparePair);
 

	return MPSet;


}
float CTrainingSet::calLeafAndBotherKNNResponse(const std::vector<std::vector<float>>& vLeafFeatureSet, const std::vector<float>& vLeafResponseSet,
	const std::vector<std::vector<float>>& vBotherFeatureSet, const std::vector<float>& vBotherResponseSet,
	const std::vector<std::vector<float>>& vAllLeafFeatureSet, const std::vector<float>& vAllLeafResponseSet,
	const std::vector<float>& vFeatures, float vPredictResponse)
{
	float MPParam = 2.0f;
	std::vector<std::pair<float, float>> leafMPSet;//first Ϊѵ������Yֵ��second ΪMPֵ
	std::vector<std::pair<float, float>> leafLPSet;//first Ϊѵ������Yֵ��second ΪLPֵ
	 //�������ֵ���� 
	float LeafResponseMid = 0.0f;
 
	//first Ϊѵ������Yֵ��second ΪMPֵ
	std::vector<std::pair<float, float>> leafResponseMPSet = calReCombineDataMP(vLeafFeatureSet, vLeafResponseSet, vFeatures, 0);
	std::vector<std::pair<float, float>> botherResponseMPSet = calReCombineDataMP(vBotherFeatureSet, vBotherResponseSet, vFeatures, 0);
	std::vector<std::pair<float, float>> similarResponseMPSet;
	/*std::vector<std::vector<float>> vNormalBotherFeatureSet;
	for (int i = 0;i < vBotherFeatureSet.size();i++)
	{
		std::vector<float> row;
		for (int j = 0;j < vBotherFeatureSet[0].size();j++)
		{
			row.push_back({ vBotherFeatureSet[i][j] });
		}
		vNormalBotherFeatureSet.push_back({ row });
	}
	normalization(vNormalBotherFeatureSet);*/
	std::vector<std::pair<float, float>> leafResponseLPSet = calReCombineDataLP(vLeafFeatureSet, vLeafResponseSet, vFeatures, 0);
	std::vector<std::pair<float, float>> botherResponseLPSet = calReCombineDataLP(vBotherFeatureSet, vBotherResponseSet, vFeatures, 0);
	std::vector<std::pair<float, float>> similarResponseLPSet;

	//Ҷ�ӵ�y��ֵ
	for (int j = 0;j < vLeafResponseSet.size();j++)//��
	{
		LeafResponseMid += vLeafResponseSet[j];
	}
	//float brotherK = 0.2;
	//int number = (int)(brotherK*vBotherResponseSet.size());

	 std::sort(leafResponseLPSet.begin(), leafResponseLPSet.end(), comparePair);
	
	for (int j = 0;j < botherResponseLPSet.size();j++)
	{
		if (botherResponseLPSet[j].second <= leafResponseLPSet[(int)(leafResponseLPSet.size()/2)].second)
		{
			similarResponseLPSet.push_back({ botherResponseLPSet[j].first,botherResponseLPSet[j].second });
		}
	}
	for (int j = 0;j < similarResponseLPSet.size();j++)
	{
		LeafResponseMid += similarResponseLPSet[j].first;
	}
	LeafResponseMid = LeafResponseMid / (vLeafResponseSet.size()+ similarResponseLPSet.size()); 
 
	//mp
	/*std::sort(leafResponseMPSet.begin(), leafResponseMPSet.end(), comparePair);
	for (int j = 0;j < botherResponseMPSet.size();j++)
	{
		if (botherResponseMPSet[j].second <= leafResponseMPSet[leafResponseMPSet.size() - 1].second)
		{
			similarResponseMPSet.push_back({ botherResponseMPSet[j].first,botherResponseMPSet[j].second });
		}
	}
	for (int j = 0;j < similarResponseMPSet.size();j++)
	{
		LeafResponseMid += similarResponseMPSet[j].first;
	}
	LeafResponseMid = LeafResponseMid / (vLeafResponseSet.size() + similarResponseMPSet.size());*/
	return LeafResponseMid;


}


 
//******************************************************************************
//FUNCTION:�ֲ�
std::vector<std::pair<float, float>> CTrainingSet::calMPwithMidPoint(const std::vector<float> &FeatureMid, const std::vector<float>& vFeatures, const std::vector<std::vector<float>>& vAllLeafFeatureSetChanged, const std::vector<std::vector<float>>& vAllLeafFeatureSet, const std::vector<float>& vAllLeafResponseSet, float ResponseMid)
{
	float MPParam = 2.0f;
	std::vector<std::pair<float, float>> MPSet;//first Ϊѵ������Yֵ��second ΪMPֵ
	std::vector<float> TrainFeatureValue = FeatureMid;
	std::vector<std::pair<float, float>> MaxMinValue;
	std::vector<float> voEachDimStandard;
	//�����׼��
	__calStandardDeviation(vAllLeafFeatureSet, voEachDimStandard);
	for (int j = 0; j < TrainFeatureValue.size(); j++)
	{
		MaxMinValue.push_back({ std::max(TrainFeatureValue[j], vFeatures[j])+ voEachDimStandard[j], std::min(TrainFeatureValue[j], vFeatures[j])- voEachDimStandard[j] });//��ʱȥ����׼��,��Ϊ���ϱ�׼�Χ̫��
	}
	//����Test��Ҷ�Ӹ�ά�Ⱦ�ֵ(�е㣩��MP
	std::vector<int> EachFeatureIntervalCount(TrainFeatureValue.size(), 0);
	float MPValueSum = 0.f;
	__calIntervalSampleByMid(MaxMinValue, vAllLeafFeatureSetChanged, vAllLeafResponseSet, EachFeatureIntervalCount);

	for (int k = 0; k < EachFeatureIntervalCount.size(); k++)//����ÿһά
	{
		MPValueSum += pow(((float)EachFeatureIntervalCount[k] / vAllLeafResponseSet.size())/**(((ResponseRange[k].first - ResponseRange[k].second) / AllREsponseRange))*/, MPParam);
	}
   
	MPSet.push_back({ ResponseMid, pow(MPValueSum, 1 / MPParam) });
	return MPSet;
}
//******************************************************************************
//FUNCTION:�ֲ�
std::vector<std::pair<float, float>>  CTrainingSet::calmidMP(const std::vector<std::vector<float>>& vAllLeafFeatureSet, const std::vector<float>& vAllLeafResponseSet, const std::vector<float>& vFeatures, float vPredictResponse)
{
	float MPParam = 2.0f;
	std::vector<std::pair<float, float>> MPSet;//first Ϊѵ������Yֵ��second ΪMPֵ
	//�������ֵ����
	std::vector<float> LeafFeatureMid(vAllLeafFeatureSet[0].size(), 0);;
	float LeafResponseMid=0.0f;
	for (int j = 0;j < vAllLeafFeatureSet[0].size();j++)//��
	{
		for (int i = 0; i < vAllLeafFeatureSet.size(); i++)//��
		{
			LeafFeatureMid[j] += vAllLeafFeatureSet[i][j];
		}
		LeafFeatureMid[j] = LeafFeatureMid[j] / vAllLeafFeatureSet.size();//ÿ��/ά�ȵľ�ֵ
	}
	//Ҷ�ӵ�y��ֵ
	for (int j = 0;j < vAllLeafResponseSet.size();j++)//��
	{
		LeafResponseMid+= vAllLeafResponseSet[j];
	}
	LeafResponseMid = LeafResponseMid / vAllLeafResponseSet.size();

	std::vector<std::vector<float>> vAllLeafFeatureSetChanged;//����

	//���������л���
	for (int j = 0;j < vAllLeafFeatureSet[0].size();j++)
	{
		std::vector<float> vAllLeafFeatureRow;
		for (int i = 0;i < vAllLeafFeatureSet.size();i++)
		{
			vAllLeafFeatureRow.push_back(vAllLeafFeatureSet[i][j]);
		}
		vAllLeafFeatureSetChanged.push_back(vAllLeafFeatureRow);
	}
	//������������-ά��-��
	for (int j = 0;j < vAllLeafFeatureSetChanged.size();j++)
	{
		std::sort(vAllLeafFeatureSetChanged[j].begin(), vAllLeafFeatureSetChanged[j].end());
	}
	std::vector<float> TrainFeatureValue = LeafFeatureMid;//�ԱȾ�ֵ
	std::vector<std::pair<float, float>> MaxMinValue;
	for (int j = 0; j < TrainFeatureValue.size(); j++)
	{
		MaxMinValue.push_back({ std::max(TrainFeatureValue[j], vFeatures[j]), std::min(TrainFeatureValue[j], vFeatures[j]) });//��ʱȥ����׼��,��Ϊ���ϱ�׼�Χ̫��
	}
	//����Test����ά�Ⱦ�ֵ(�е㣩��MP
	std::vector<int> EachFeatureIntervalCount(TrainFeatureValue.size(), 0);
	float MPValueSum = 0.f;  
    __calIntervalSampleByMid(MaxMinValue, vAllLeafFeatureSetChanged, vAllLeafResponseSet, EachFeatureIntervalCount);
	 
	for (int k = 0; k < EachFeatureIntervalCount.size(); k++)//����ÿһά
	{
			MPValueSum += pow(((float)EachFeatureIntervalCount[k] / vAllLeafResponseSet.size())/**(((ResponseRange[k].first - ResponseRange[k].second) / AllREsponseRange))*/, MPParam);
	}


	MPSet.push_back({ LeafResponseMid, pow(MPValueSum, 1 / MPParam) });


	 
	return MPSet;


}
//******************************************************************************
//FUNCTION:�ֲ�
std::vector<std::pair<float, float>> CTrainingSet::calReCombineDataMP(const std::vector<std::vector<float>>& vAllLeafFeatureSet, const std::vector<float>& vAllLeafResponseSet, const std::vector<float>& vFeatures, float vPredictResponse)
{
	float MPParam = 2.0f;
	std::vector<std::pair<float, float>> MPSet;//first Ϊѵ������Yֵ��second ΪMPֵ
	
	float AllREsponseRange = *std::max_element(vAllLeafResponseSet.begin(), vAllLeafResponseSet.end()) - *std::min_element(vAllLeafResponseSet.begin(), vAllLeafResponseSet.end());
	int InstanceNoRepeat = CTrainingSetConfig::getInstance()->getAttribute<int>(KEY_WORDS::INSTANCE_NO_REPEAT);
	 
	std::vector<std::vector<float>> vAllLeafFeatureSetChanged;//����
	
	//���л���
	for (int j = 0;j < vAllLeafFeatureSet[0].size();j++)
	{
		std::vector<float> vAllLeafFeatureRow;
		for (int i = 0;i < vAllLeafFeatureSet.size();i++)
		{
			vAllLeafFeatureRow.push_back(vAllLeafFeatureSet[i][j]);
		}
		vAllLeafFeatureSetChanged.push_back(vAllLeafFeatureRow);
	}
	//��������-ά��-��
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
			MaxMinValue.push_back({ std::max(TrainFeatureValue[j], vFeatures[j]), std::min(TrainFeatureValue[j], vFeatures[j]) });//��ʱȥ����׼��,��Ϊ���ϱ�׼�Χ̫��
		}

		/*__calIntervalSample(MaxMinValue, vAllLeafFeatureSet, vAllLeafResponseSet, EachFeatureIntervalCount, ResponseRange);*/
		//__countIntervalNode(MaxMinValue, EachFeatureIntervalCount, ResponseRange);
		__calIntervalSampleByOrder(MaxMinValue, vAllLeafFeatureSetChanged, vAllLeafResponseSet, EachFeatureIntervalCount, ResponseRange);

		for (int k = 0; k < EachFeatureIntervalCount.size(); k++)//����ÿһά
		{
			MPValueSum += pow(((float)EachFeatureIntervalCount[k] / vAllLeafResponseSet.size())/**(((ResponseRange[k].first - ResponseRange[k].second) / AllREsponseRange))*/, MPParam);
		}
		
		 
	   MPSet.push_back({ vAllLeafResponseSet[i], pow(MPValueSum, 1 / MPParam) });
		 
		
	}
	std::sort(MPSet.begin(), MPSet.end(), comparePair);
	
	//ȥ���ظ�ֵ-----zy20190214
	if (InstanceNoRepeat == 1)
	{
		MPSet.erase(unique(MPSet.begin(), MPSet.end()), MPSet.end());

	}

	return MPSet;
}

//******************************************************************************
//FUNCTION:�ֲ�
std::vector<std::pair<float, float>> CTrainingSet::calReCombineDataLP(const std::vector<std::vector<float>>& vAllLeafFeatureSet, const std::vector<float>& vAllLeafResponseSet, const std::vector<float>& vFeatures, float vPredictResponse)
{
	float LPParam = 2.0f;
	std::vector<std::pair<float, float>> LPSet;//first Ϊѵ������Yֵ��second ΪLPֵ  
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

	//ȥ���ظ�ֵ-----zy20190214
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
	std::vector<std::pair<int, float>> MPSet;//first Ϊѵ������������second ΪMPֵ
	std::vector<std::pair<int, float>> MPSetByNumber;//first Ϊѵ������������second ΪMPֵ
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
	std::vector<std::pair<int, float>> MPSet;//first Ϊѵ������������second ΪMPֵ
	std::vector<std::pair<int, float>> MPSetByNumber;//first Ϊѵ������������second ΪMPֵ
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
	std::vector<std::pair<int, float>> LPSet;//first Ϊѵ������������second ΪMPֵ
	std::vector<std::pair<int, float>> LPSetByNumber;//first Ϊѵ������������second ΪMPֵ
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
//FUNCTION:����ĳ����Χ�ڵ�����ĵ���
void CTrainingSet::__countIntervalNode(const std::vector<std::pair<float, float>>& vMaxMinValue, std::vector<int>& voIntervalCount, std::vector<std::pair<float, float>>& voInterResponseRange)
{
	_ASSERTE(vMaxMinValue.size() == m_FeatureSet[0].size());
    #pragma omp parallel for
	for (int i = 0; i < m_FeatureSet[0].size(); i++)//ȷ��ĳһά
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
//FUNCTION:�ֲ���Χ������
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
////FUNCTION:�ֲ���Χ������-�ڰ�����ֵ���������������
void CTrainingSet::__calIntervalSampleByOrder(const std::vector<std::pair<float, float>>& vMaxMinValue, const std::vector<std::vector<float>>& vAllLeafFeatureSetOrdered, const std::vector<float>& vAllLeafResponseSet, std::vector<int>& voIntervalCount, std::vector<std::pair<float, float>>& voInterResponseRange)
{
	_ASSERTE(vMaxMinValue.size() == vAllLeafFeatureSetOrdered.size());
	_ASSERTE(!vAllLeafFeatureSetOrdered.empty() && !vAllLeafResponseSet.empty() && !voIntervalCount.empty() && !voInterResponseRange.empty());

	//����������ά��
//#pragma omp parallel for
	for (int i = 0; i < vAllLeafFeatureSetOrdered.size(); i++)
	{
		int count = 0;
		//ĳ���������������������ֵ
		int indexMin = 0;
	 
		int indexMax = 0;
		//��ȡ��Сֵ���±�
		for (int j = 0; j < vAllLeafFeatureSetOrdered[i].size(); j++) 
		{
			if (vAllLeafFeatureSetOrdered[i][j] >= vMaxMinValue[i].second)
			{
				indexMin = j; 
				indexMax = indexMin;
				break; 
			} 
		}
		//��ȡ���ֵ���±�
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


////******************************************************************************
////FUNCTION:�ֲ���Χ������-�ڰ�����ֵ���������������-��response
void CTrainingSet::__calIntervalSampleByMid(const std::vector<std::pair<float, float>>& vMaxMinValue, const std::vector<std::vector<float>>& vAllLeafFeatureSetOrdered, const std::vector<float>& vAllLeafResponseSet, std::vector<int>& voIntervalCount)
{
	_ASSERTE(vMaxMinValue.size() == vAllLeafFeatureSetOrdered.size());
	_ASSERTE(!vAllLeafFeatureSetOrdered.empty() && !vAllLeafResponseSet.empty() && !voIntervalCount.empty() && !voInterResponseRange.empty());

	//����������ά��
	//#pragma omp parallel for
	for (int i = 0; i < vAllLeafFeatureSetOrdered.size(); i++)
	{
		int count = 0;
		//ĳ���������������������ֵ
		int indexMin = 0; 
		int indexMax = 0;
		//��ȡ��Сֵ���±�
		for (int j = 0; j < vAllLeafFeatureSetOrdered[i].size(); j++)
		{
			if (vAllLeafFeatureSetOrdered[i][j] >= vMaxMinValue[i].second)
			{
				indexMin = j;
				indexMax = indexMin;
				break;
			}
		}
		//��ȡ���ֵ���±�
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


////FUNCTION:�ֲ���Χ������-�ڰ�����ֵ���������������-��response
void CTrainingSet::__calIntervalSampleByMid(const std::vector<std::pair<float, float>>& vMaxMinValue, const std::vector<std::vector<float>>& vAllLeafFeatureSetOrdered, std::vector<int>& voIntervalCount)
{
	_ASSERTE(vMaxMinValue.size() == vAllLeafFeatureSetOrdered.size());
	_ASSERTE(!vAllLeafFeatureSetOrdered.empty());

	//����������ά��
	//#pragma omp parallel for
	for (int i = 0; i < vAllLeafFeatureSetOrdered.size(); i++)
	{
		int count = 0;
		//ĳ���������������������ֵ
		int indexMin = 0;
		int indexMax = 0;
		//��ȡ��Сֵ���±�
		for (int j = 0; j < vAllLeafFeatureSetOrdered[i].size(); j++)
		{
			if (vAllLeafFeatureSetOrdered[i][j] >= vMaxMinValue[i].second)
			{
				indexMin = j;
				indexMax = indexMin;
				break;
			}
		}
		//��ȡ���ֵ���±�
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
	 
	//ȥ���ظ���
	sort(TotalDataIndex.begin(), TotalDataIndex.end()); 
	TotalDataIndex.erase(unique(TotalDataIndex.begin(), TotalDataIndex.end()), TotalDataIndex.end());
 
	std::vector<std::pair<int, int>>  DataIndexCount;
	std::vector<std::pair<int, float>>  DataIndexMp;//first Ϊѵ������indexֵ��second ΪMPֵ
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

	 //ͳ��ÿ�������������֮��
	
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
	////��ȡ�����Ƶĵ��y��count���ĵ�
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
