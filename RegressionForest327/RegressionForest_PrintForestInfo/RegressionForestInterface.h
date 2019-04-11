#pragma once
#include <vector>
#include "RegressionForest.h"
#include "RegressionForest_EXPORTS.h"

namespace hiveRegressionForest
{
	REGRESSION_FOREST_EXPORTS unsigned int				hiveBuildRegressionForest(const std::string& vConfig);
	
	//for single response
	REGRESSION_FOREST_EXPORTS float						hivePredict(const std::vector<float>& vTestInstance, unsigned int vForestId, bool vIsWeightedPrediction = true);
	REGRESSION_FOREST_EXPORTS float 					hivePredict(const std::vector<float>& vTestInstance, unsigned int vForestId, unsigned int vNumOfUsingTrees, bool vIsWeightedPrediction = true);
	
	REGRESSION_FOREST_EXPORTS float						hivePredict(const std::vector<float>& vTestInstance, float vTestResponse, unsigned int vForestId, float& voMPPredictSet, bool vIsWeightedPrediction = true);
	REGRESSION_FOREST_EXPORTS float 					hivePredict(const std::vector<float>& vTestInstance, float vTestResponse, unsigned int vForestId, unsigned int vNumOfUsingTrees, float& voMPPredictSet, bool vIsWeightedPrediction = true);
	
	//for multiple response
	REGRESSION_FOREST_EXPORTS void						hivePredict(const std::vector<float>& vTestInstance, unsigned int vForestId, unsigned int vNumResponse, std::vector<float>& voPredictValue, bool vIsWeightedPrediction = false);
	REGRESSION_FOREST_EXPORTS void						hivePredict(const std::vector<float>& vTestInstance, unsigned int vForestId, int vNumOfUsingTrees, unsigned int vNumResponse, std::vector<float>& voPredictValue, bool vIsWeightedPrediction = false);
	
	REGRESSION_FOREST_EXPORTS float						hivePredictFromTreeAt(const std::vector<float>& vTestInstance, int vTreeIndex, unsigned int vForestId);

	REGRESSION_FOREST_EXPORTS const std::vector<int>&	hiveGetOOBIndexSet(int vTreeIndex, unsigned int vForestId);
	REGRESSION_FOREST_EXPORTS void						hiveOutputForestInfo(const std::string& vOutputFileName, unsigned int vForestId);
	REGRESSION_FOREST_EXPORTS void						hiveOutputForestAndOOBInfo(const std::string& vOutputFileName, unsigned int vForestId);

	REGRESSION_FOREST_EXPORTS void						hiveSaveForestToFile(const std::string& vFilePath, unsigned int vForestId);
	REGRESSION_FOREST_EXPORTS unsigned int				hiveLoadForestFromFile(const std::string& vFilePath);
	
	REGRESSION_FOREST_EXPORTS bool						hiveCompareForests(unsigned int vForestId1, unsigned int vForestId2);

	REGRESSION_FOREST_EXPORTS bool						hiveParseTestSet(const std::string& vTestFilePath, std::vector<std::vector<float>>& voTestFeatureSet, std::vector<float>& voTestResponseSet, unsigned int vNumResponse = 1, bool vHeader = false);
}