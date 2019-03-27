#include "FeatureWeightPearsonMethod.h"
#include "RegressionForestCommon.h"
#include "TrainingSet.h"
#include "common/ProductFactory.h"
#include "math/PearsonCorrelation.h"

#define  PEARSON_MIN -1
#define  PEARSON_MAX 1

using namespace hiveRegressionForest;

hiveOO::CProductFactory<CPearsonFeatureWeightMethod> theCreator(KEY_WORDS::PEARSON_METHOD);

CPearsonFeatureWeightMethod::CPearsonFeatureWeightMethod()
{
}

CPearsonFeatureWeightMethod::~CPearsonFeatureWeightMethod()
{
}

//****************************************************************************************************
//FUNCTION:
void CPearsonFeatureWeightMethod::__calculateFeatureWeightV(const std::vector<std::vector<float>>& vInstanceSet, const std::vector<float>& vResponseSet, std::vector<std::pair<unsigned int, float>>& voFeatureWeightSet)
{
	_ASSERTE(vInstanceSet.size() && vResponseSet.size() && vInstanceSet.size() == vResponseSet.size());
	unsigned int FeautureSize = vInstanceSet[0].size();
	unsigned int InstanceSize = vInstanceSet.size();
	std::vector<std::vector<float>> FeautureSet(FeautureSize);
	std::vector<float> PearsonCorrelationSet(FeautureSize);
	
	for (unsigned int i = 0; i < FeautureSize; i++)
	{
		for (unsigned int k = 0; k < InstanceSize; k++)
		{
			FeautureSet[i].push_back(vInstanceSet[k][i]);
		}
	}

	CPearsonCorrelation<float>::getInstance()->calPearsonCorrelation(FeautureSet, vResponseSet, PearsonCorrelationSet);
	voFeatureWeightSet.clear();
	for (auto i = 0; i < PearsonCorrelationSet.size(); ++i)
	{
		if (PearsonCorrelationSet[i] == PEARSON_INVALID || (PearsonCorrelationSet[i] > -FLT_MIN && PearsonCorrelationSet[i] <FLT_MIN))		//LWJ:���ｫȨ��ֵnormalize,1.���������-1��1֮��ʱ��2.Ȩ��Ϊ0ʱ����Ϊ��Ȩ�ض�Ϊ0ʱ����Ȩ�ظ������ѡ�����ַ��������
		{
			PearsonCorrelationSet[i] = FLT_MIN;
		}
		voFeatureWeightSet.push_back(std::make_pair(i, abs(PearsonCorrelationSet[i])));
	}
	std::sort(voFeatureWeightSet.begin(), voFeatureWeightSet.end(),
		[](const std::pair<unsigned int, float>& vLeft, const std::pair<unsigned int, float>& vRight) { return vLeft.second > vRight.second; });
}