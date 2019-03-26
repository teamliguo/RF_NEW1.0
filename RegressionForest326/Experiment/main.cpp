#include <vector>
#include <fstream>
#include <numeric>
#include "common/ConfigInterface.h"
#include "common/HiveCommonMicro.h"
#include "math/RegressionAnalysisInterface.h"
#include "../RegressionForest/RegressionForestInterface.h"
#include "../RegressionForest/RegressionForestCommon.h"
#include "../RegressionForest/TrainingSet.h"
#include "../RegressionForest/TrainingSetConfig.h"
#include "../RegressionForest/TrainingSetCommon.h"
#include "ExtraConfig.h"
#include "ExtraCommon.h"

using namespace hiveRegressionForest;
using namespace hiveRegressionForestExtra;

//FUNCTION: detect the memory leak in DEBUG mode
void installMemoryLeakDetector()
{
	// Enable run-time memory check for debug builds.
#if defined(DEBUG) | defined(_DEBUG)
	//_CRTDBG_LEAK_CHECK_DF: Perform automatic leak checking at program exit through a call to _CrtDumpMemoryLeaks and generate an error 
	//report if the application failed to free all the memory it allocated. OFF: Do not automatically perform leak checking at program exit.
	_CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);

	//the following statement is used to trigger a breakpoint when memory leak happens
	//comment it out if there is no memory leak report;
	//_crtBreakAlloc = 150484;
#endif
}

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

void main()
{
	installMemoryLeakDetector();

	try
	{
		_LOG_("Parsing extra config file...");
		bool IsConfigParsedExtra = hiveConfig::hiveParseConfig("BatchRunConfig.xml", hiveConfig::EConfigType::XML, CExtraConfig::getInstance());
		_ASSERTE(IsConfigParsedExtra);

		_LOG_("Loading Training Set...");
		CTrainingSet* pTrainingSet = CTrainingSet::getInstance();
		bool IsDataLoaded = pTrainingSet->loadTrainingSet("TrainingSetConfig.xml");
		_ASSERTE(IsDataLoaded);
		
		_LOG_("Parsing Test Set...");
		std::vector<std::vector<float>> TestFeatureSet;
		std::vector<float> TestResponseSet;
		const std::string TestSetPath = CTrainingSetConfig::getInstance()->getAttribute<std::string>(hiveRegressionForest::KEY_WORDS::TESTSET_PATH);
		hiveParseTestSet(TestSetPath, TestFeatureSet, TestResponseSet);

		bool IsNormalize = CTrainingSetConfig::getInstance()->getAttribute<bool>(hiveRegressionForest::KEY_WORDS::IS_NORMALIZE);
		if (IsNormalize)
			pTrainingSet->normalization(TestFeatureSet);

		_LOG_("Building Forests...");
		int ForestId = hiveRegressionForest::hiveBuildRegressionForest("Config.xml");
		_LOG_("Building Finished.");

		//hiveOutputForestInfo("OutputForestInfo.csv", ForestId);

		_LOG_("Predict...");
		std::vector<float> PredictSet(TestFeatureSet.size(), 0.0f);
		clock_t PredictStart = clock();

		for (auto Index = 0; Index < TestFeatureSet.size(); ++Index)
		{
			std::cout << "Actual value: " << TestResponseSet[Index] << std::endl;
			PredictSet[Index] = hiveRegressionForest::hivePredict(TestFeatureSet[Index], ForestId, false);
		}

		clock_t PredictEnd = clock();
		_LOG_("Predict Finished in " + std::to_string(PredictEnd - PredictStart) + " milliseconds.");

		_LOG_("Output predict result to file...");
		int NumResponse = pTrainingSet->getNumOfResponse();
		std::ofstream PredictResultFile(CExtraConfig::getInstance()->getAttribute<std::string>(hiveRegressionForestExtra::KEY_WORDS::PREDICT_RESULT_PATH));
		for (auto i = 0; i < TestFeatureSet.size(); i++)
		{
			PredictResultFile << TestResponseSet[i] << "," << PredictSet[i] << "," << abs(PredictSet[i] - TestResponseSet[i]) << std::endl;

		}
		PredictResultFile.close();
		_LOG_("Output predict result to file finished");

		std::vector<float> BiasSet(TestFeatureSet.size(), 0.0f);
		std::vector<float> BiasRateSet(TestFeatureSet.size(), 0.0f);

		for (int i = 0; i < TestFeatureSet.size(); ++i)
		{
			BiasSet[i] = std::abs(PredictSet[i] - TestResponseSet[i]);
			BiasRateSet[i] = BiasSet[i] / TestResponseSet[i];
		}

		float MaxBias = *std::max_element(BiasSet.begin(), BiasSet.end());
		float SumBias = std::accumulate(BiasSet.begin(), BiasSet.end(), 0.0f);

		std::cout << "max bias is " << MaxBias << std::endl;
		std::cout << "sum bias is " << SumBias << std::endl;

		float MaxBiasRate = *std::max_element(BiasRateSet.begin(), BiasRateSet.end());
		float SumBiasRate = std::accumulate(BiasRateSet.begin(), BiasRateSet.end(), 0.0f);

		std::vector<float> ThresholdVec = { 0.5f, 1.0f, 1.5f, 2.0f, 2.5f, 3.0f, 3.5f, 4.0f, 4.5f, 5.0f };
		std::vector<float> BiasRateVec = { 0.01f, 0.02f, 0.03f, 0.04f, 0.05f, 0.06f, 0.07f, 0.08f, 0.09f, 0.10f };
		std::vector<float> AccuracyVecByThreshold, AccuracyVecByBiasRate;
		calAccuracyByThreshold(BiasSet, ThresholdVec, AccuracyVecByThreshold);
		calAccuracyByBiasRate(BiasRateSet, BiasRateVec, AccuracyVecByBiasRate);
		float MSE = calMSE(BiasSet);
		float R2Score = calR2Score(TestResponseSet, PredictSet);

		std::cout << "MSE " << MSE << std::endl;
		std::cout << "R2Score " << R2Score << std::endl;

		std::ofstream StatisticalResultFile(CExtraConfig::getInstance()->getAttribute<std::string>(hiveRegressionForestExtra::KEY_WORDS::STATISTICAL_RESULT_PATH));
		_ASSERTE(StatisticalResultFile.is_open());
		StatisticalResultFile << "Max bias rate: " << MaxBiasRate << std::endl;
		StatisticalResultFile << "Sum bias rate: " << SumBiasRate << std::endl;
		StatisticalResultFile << "MSE:" << MSE << std::endl;
		StatisticalResultFile << "R2Score:" << R2Score << std::endl;
		for (auto i = 0; i < AccuracyVecByThreshold.size(); i++)
		{
			StatisticalResultFile << "Accuracy of bias < " << ThresholdVec[i] << " : " << AccuracyVecByThreshold[i] << std::endl;
		}
		for (auto i = 0; i < AccuracyVecByBiasRate.size(); i++)
		{
			StatisticalResultFile << "Accuracy of bias rate < " << BiasRateVec[i] << " : " << AccuracyVecByBiasRate[i] << std::endl;
		}
		StatisticalResultFile.close();

		if (CExtraConfig::getInstance()->getAttribute<bool>(hiveRegressionForestExtra::KEY_WORDS::IS_SERIALIZE_MODEL))
		{
			_LOG_("SerializeRF...");
			hiveSaveForestToFile(CExtraConfig::getInstance()->getAttribute<std::string>(hiveRegressionForestExtra::KEY_WORDS::SERIALIZATION_MODEL_PATH), ForestId);
			_LOG_("SerializeRF Finished.");
		}
	}
	catch (const std::exception&)
	{
		hiveCommon::hiveOutputWarning(__EXCEPTION_SITE__, "The program is terminated due to unexpected error.");
	}
}