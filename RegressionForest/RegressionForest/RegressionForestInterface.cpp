#include "RegressionForestInterface.h"
#include <fstream>
#include <stack>
#include "common/HiveCommonMicro.h"
#include "common/ProductFactoryData.h"
#include "math/RegressionAnalysisInterface.h"
#include "RegressionForestCommon.h"
#include "TrainingSet.h"
#include "RegressionForestConfig.h"
#include "math/LeastSquaresRegression.h"
#include "math/AverageOutputRegression.h"
#include "math/ForwardStagewiseRegression.h"
#include "math/ForwardStepwiseRgression.h"
#include "SingleResponseNode.h"
#include "MultiResponseNode.h"
#include "RegressionForestPool.h"

using namespace hiveRegressionAnalysis;

//****************************************************************************************************
//FUNCTION:
unsigned int hiveRegressionForest::hiveBuildRegressionForest(const std::string& vConfig)
{
	_ASSERTE(!vConfig.empty());

	CRegressionForest* pRegressionForest = new CRegressionForest();
	pRegressionForest->buildForest(vConfig);

	unsigned int ForestId = CRegressionForestPool::getInstance()->putForest(pRegressionForest);
	return ForestId;
}

//****************************************************************************************************
//FUNCTION:
unsigned int hiveRegressionForest::hiveRebuildRegressionForest(const std::string & vConfig, const std::string & vFilePath)
{
	_ASSERTE(!vConfig.empty());

	CRegressionForest* pRegressionForest = new CRegressionForest();
	pRegressionForest->reParseConfig(vConfig);

	return hiveLoadForestFromFile(vFilePath);
}

//******************************************************************************
//FUNCTION:  for single response
void hiveRegressionForest::hivePredict(const std::vector<std::vector<float>>& vTestFeatureSet, const std::vector<float>& vTestResponseSet, std::vector<float>& voPredictSet, unsigned int vForestId)
{
	_ASSERT(!vTestFeatureSet.empty() && !vTestResponseSet.empty());

	const CRegressionForest* pRegressionForest = CRegressionForestPool::getInstance()->fetchForest(vForestId);
	_ASSERTE(pRegressionForest);

	pRegressionForest->predict(vTestFeatureSet, vTestResponseSet, voPredictSet);
}

//****************************************************************************************************
//FUNCTION:  for multiple response
void hiveRegressionForest::hivePredict(const std::vector<float>& vTestInstance, unsigned int vForestId, unsigned int vNumResponse, std::vector<float>& voPredictValue, bool vIsWeightedPrediction /*= true*/)
{
	_ASSERT(!vTestInstance.empty());

	const CRegressionForest* pRegressionForest = CRegressionForestPool::getInstance()->fetchForest(vForestId);
	_ASSERTE(pRegressionForest);

	hivePredict(vTestInstance, vForestId, pRegressionForest->getNumOfTrees(), vNumResponse, voPredictValue, vIsWeightedPrediction);
}

//****************************************************************************************************
//FUNCTION:  for multiple response
void hiveRegressionForest::hivePredict(const std::vector<float>& vTestInstance, unsigned int vForestId, int vNumOfUsingTrees, unsigned int vNumResponse, std::vector<float>& voPredictValue, bool vIsWeightedPrediction /*= true*/)
{
	_ASSERT(!vTestInstance.empty());

	const CRegressionForest* pRegressionForest = CRegressionForestPool::getInstance()->fetchForest(vForestId);
	_ASSERTE(pRegressionForest);

	pRegressionForest->predict(vTestInstance, vNumOfUsingTrees, vIsWeightedPrediction, vNumResponse, voPredictValue);
}

//****************************************************************************************************
//FUNCTION:
float hiveRegressionForest::hivePredictFromTreeAt(const std::vector<float>& vTestInstance, int vTreeIndex, unsigned int vForestId)
{
	const CRegressionForest* pRegressionForest = CRegressionForestPool::getInstance()->fetchForest(vForestId);

	_ASSERTE(!vTestInstance.empty() && vTreeIndex < pRegressionForest->getNumOfTrees());

	const CTree* CurTree = pRegressionForest->getTreeAt(vTreeIndex);
	const CNode* CurLeafNode = CurTree->locateLeafNode(vTestInstance);
	
	float Weight = 0.0f;
	return CurTree->predict(*CurLeafNode, vTestInstance, Weight);
}

//****************************************************************************************************
//FUNCTION:
const std::vector<int>& hiveRegressionForest::hiveGetOOBIndexSet(int vTreeIndex, unsigned int vForestId)
{
	const CRegressionForest* pRegressionForest = CRegressionForestPool::getInstance()->fetchForest(vForestId);

	_ASSERTE(vTreeIndex < pRegressionForest->getNumOfTrees());

	return pRegressionForest->getTreeAt(vTreeIndex)->getOOBIndexSet();
}

//****************************************************************************************************
//FUNCTION:
void hiveRegressionForest::hiveOutputForestInfo(const std::string& vOutputFileName, unsigned int vForestId)
{
	const CRegressionForest* pRegressionForest = CRegressionForestPool::getInstance()->fetchForest(vForestId);
	pRegressionForest->outputForestInfo(vOutputFileName);
}

//****************************************************************************************************
//FUNCTION:
void hiveRegressionForest::hiveOutputForestAndOOBInfo(const std::string& vOutputFileName, unsigned int vForestId)
{
	const CRegressionForest* pRegressionForest = CRegressionForestPool::getInstance()->fetchForest(vForestId);
	pRegressionForest->outputOOBInfo(vOutputFileName);
}

//****************************************************************************************************
//FUNCTION:
void hiveRegressionForest::hiveSaveForestToFile(const std::string& vFilePath, unsigned int vForestId)
{
	const CRegressionForest* pRegressionForest = CRegressionForestPool::getInstance()->fetchForest(vForestId);

	std::ofstream OutputSerizlizedFile(vFilePath);
	boost::archive::text_oarchive Oarchive(OutputSerizlizedFile);
	Oarchive.register_type<CLeastSquaresRegression>();
	Oarchive.register_type<CAverageOutputRegression>();
	Oarchive.register_type<CForwardStagewiseRegression>();
	Oarchive.register_type<CForwardStepwiseRegression>();
	Oarchive.register_type<CSingleResponseNode>();
	Oarchive.register_type<CMultiResponseNode>();

	Oarchive << pRegressionForest;
	OutputSerizlizedFile.close();
}

//****************************************************************************************************
//FUNCTION:
unsigned int hiveRegressionForest::hiveLoadForestFromFile(const std::string & vFilePath)
{
	_ASSERTE(!vFilePath.empty());

	CRegressionForest* pRegressionForest = new CRegressionForest();

	std::ifstream InputSerizlizedFile(vFilePath);
	boost::archive::text_iarchive Iarchive(InputSerizlizedFile);
	Iarchive.register_type<CLeastSquaresRegression>();
	Iarchive.register_type<CAverageOutputRegression>();
	Iarchive.register_type<CForwardStagewiseRegression>();
	Iarchive.register_type<CForwardStepwiseRegression>();
	Iarchive.register_type<CSingleResponseNode>();
	Iarchive.register_type<CMultiResponseNode>();

	Iarchive >> pRegressionForest;
	InputSerizlizedFile.close();

	unsigned int ForestId = CRegressionForestPool::getInstance()->putForest(pRegressionForest);
	return ForestId;
}

//****************************************************************************************************
//FUNCTION:
bool hiveRegressionForest::hiveCompareForests(unsigned int vForestId1, unsigned int vForestId2)
{
	const CRegressionForest* pRegressionForest1 = CRegressionForestPool::getInstance()->fetchForest(vForestId1);
	const CRegressionForest* pRegressionForest2 = CRegressionForestPool::getInstance()->fetchForest(vForestId2);

	return pRegressionForest1->operator==(*pRegressionForest2);
}

//****************************************************************************************************
//FUNCTION:
bool hiveRegressionForest::hiveParseTestSet(const std::string& vTestFilePath, std::vector<std::vector<float>>& voTestFeatureSet, std::vector<float>& voTestResponseSet, unsigned int vNumResponse, bool vHeader)
{
	_ASSERTE(!vTestFilePath.empty());

	int NumOfFeature = CTrainingSet::getInstance()->getNumOfFeatures();

	std::ifstream DataFile(vTestFilePath);
	if (DataFile.is_open())
	{
		std::string Line;

		if (vHeader)
		{
			getline(DataFile, Line);
			_LOG_("Ignore header of [" + vTestFilePath + "] by default.\n");
		}

		std::vector<std::string> InstanceString;
		while (getline(DataFile, Line))
		{
			InstanceString.clear();
			hiveCommon::hiveSplitLine(Line, ", ", false, 0, InstanceString);
			_ASSERTE(!InstanceString.empty());

			if (!voTestFeatureSet.empty() && !voTestResponseSet.empty() && (InstanceString.size() != (NumOfFeature + vNumResponse)))
			{
				DataFile.close();
				voTestFeatureSet.clear();
				voTestResponseSet.clear();

				return false;
			}
			std::vector<float> Instance(InstanceString.size());
			int i = 0;
			std::for_each(Instance.begin(), Instance.end(), [&InstanceString, &i](float& vData) { vData = std::atof(InstanceString[i++].c_str()); });
			
			voTestFeatureSet.push_back(std::vector<float>(Instance.begin(), Instance.end() - vNumResponse));
			for (int k = 0; k < vNumResponse; ++k)
				voTestResponseSet.push_back(Instance[NumOfFeature + k]);
		}
		_ASSERTE((voTestFeatureSet.size()*vNumResponse) == voTestResponseSet.size());

		DataFile.close();

		return true;
	}
	else
	{
		std::cout << "Failed to open file [" + vTestFilePath + "], check file path.\n";

		return false;
	}
}