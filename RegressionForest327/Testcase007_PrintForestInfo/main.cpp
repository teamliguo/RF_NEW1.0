#include <vector>
#include <fstream>
#include <numeric>
#include "common/ConfigInterface.h"
#include "common/HiveCommonMicro.h"
#include "math/RegressionAnalysisInterface.h"
#include "../RegressionForest_PrintForestInfo/RegressionForestInterface.h"
#include "../RegressionForest_PrintForestInfo/RegressionForestCommon.h"
#include "../RegressionForest_PrintForestInfo/TrainingSet.h"
#include "../RegressionForest_PrintForestInfo/TrainingSetConfig.h"
#include "../RegressionForest_PrintForestInfo/TrainingSetCommon.h"
#include "../RegressionForest_PrintForestInfo/Utility.h"
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
		
		int ForestId;
		bool IsModelExist = CExtraConfig::getInstance()->getAttribute<bool>(hiveRegressionForestExtra::KEY_WORDS::IS_MODEL_EXIST);
		bool IsSerializeModel = CExtraConfig::getInstance()->getAttribute<bool>(hiveRegressionForestExtra::KEY_WORDS::IS_SERIALIZE_MODEL);
		if (IsModelExist)
		{
			_LOG_("ReBuilding Forests...");
			ForestId = hiveLoadForestFromFile(CExtraConfig::getInstance()->getAttribute<std::string>(hiveRegressionForestExtra::KEY_WORDS::SERIALIZATION_MODEL_PATH));
			_LOG_("ReBuilding Finished.");
		}
		else
		{
			_LOG_("Building Forests...");
			ForestId = hiveRegressionForest::hiveBuildRegressionForest("Config.xml");
			_LOG_("Building Finished.");
			if(IsSerializeModel)
			{
				_LOG_("SerializeRF...");
				hiveSaveForestToFile(CExtraConfig::getInstance()->getAttribute<std::string>(hiveRegressionForestExtra::KEY_WORDS::SERIALIZATION_MODEL_PATH), ForestId);
				_LOG_("SerializeRF Finished.");
			}
		}	

		//hiveOutputForestInfo("OutputForestInfo.csv", ForestId);
		
		_LOG_("Predict...");
		std::vector<float> PredictSet(TestFeatureSet.size(), 0.0f);
		clock_t PredictStart = clock();
		bool IsPrint = CTrainingSetConfig::getInstance()->getAttribute<bool>(hiveRegressionForest::KEY_WORDS::IS_PRINT_LEAF_NODE);
		for (auto Index = 0; Index < TestFeatureSet.size(); ++Index)
		{
			if (IsPrint)
			{
				std::ofstream BestTreeFile(CTrainingSetConfig::getInstance()->getAttribute<std::string>(hiveRegressionForest::KEY_WORDS::BEST_TREE_PATH), std::ios::app);
				std::ofstream BadTreeFile(CTrainingSetConfig::getInstance()->getAttribute<std::string>(hiveRegressionForest::KEY_WORDS::BAD_TREE_PATH), std::ios::app);
				BestTreeFile << "TEST " << Index << ",";
				BadTreeFile << "TEST " << Index << ",";
				for (int i = 0; i < TestFeatureSet[Index].size(); i++)
				{
					BestTreeFile << TestFeatureSet[Index][i] << ",";
					BadTreeFile << TestFeatureSet[Index][i] << ",";
				}
				BestTreeFile << TestResponseSet[Index] << std::endl;
				BadTreeFile << TestResponseSet[Index] << std::endl;
				BestTreeFile.close();
				BadTreeFile.close();
			}
			PredictSet[Index] = hiveRegressionForest::hivePredict(TestFeatureSet[Index], TestResponseSet[Index], ForestId, false);
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
		std::vector<std::pair<int, float>> BiasRateIndex;

		for (int i = 0; i < TestFeatureSet.size(); ++i)
		{
			BiasSet[i] = std::abs(PredictSet[i] - TestResponseSet[i]);
			BiasRateSet[i] = BiasSet[i] / TestResponseSet[i];
			BiasRateIndex.push_back(std::make_pair(i, BiasRateSet[i]));
		}

		bool IsDevideTestFile = CTrainingSetConfig::getInstance()->getAttribute<bool>(hiveRegressionForest::KEY_WORDS::IS_DIVIDE_FILE);
		if (IsDevideTestFile)
		{
			sort(BiasRateIndex.begin(), BiasRateIndex.end(), [](std::pair<int, float>& vFirst, std::pair<int, float>& vSecond) {return vFirst.second < vSecond.second; });
			std::ofstream GoodTestFile(CTrainingSetConfig::getInstance()->getAttribute<std::string>(hiveRegressionForest::KEY_WORDS::GOOD_TEST_FILE));
			std::ofstream BadTestFile(CTrainingSetConfig::getInstance()->getAttribute<std::string>(hiveRegressionForest::KEY_WORDS::BAD_TEST_FILE));
			int PrintDataSize = CTrainingSetConfig::getInstance()->getAttribute<int>(hiveRegressionForest::KEY_WORDS::NEW_FILE_DATA_SIZE);
			_ASSERTE(PrintDataSize <= BiasRateIndex.size());

			for (int i = 0; i < PrintDataSize; i++)
			{
				int FrontIndex = BiasRateIndex[i].first;
				int LastIndex = BiasRateIndex[BiasRateIndex.size() - 1 - i].first;
				for (int k = 0; k < TestFeatureSet[0].size(); k++)
				{
					GoodTestFile << TestFeatureSet[FrontIndex][k] << ",";
					BadTestFile << TestFeatureSet[LastIndex][k] << ",";
				}
				GoodTestFile << TestResponseSet[FrontIndex] << std::endl;
				BadTestFile << TestResponseSet[LastIndex] << std::endl;
			}
			GoodTestFile.close();
			BadTestFile.close();
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

	}
	catch (const std::exception&)
	{
		hiveCommon::hiveOutputWarning(__EXCEPTION_SITE__, "The program is terminated due to unexpected error.");
	}
}