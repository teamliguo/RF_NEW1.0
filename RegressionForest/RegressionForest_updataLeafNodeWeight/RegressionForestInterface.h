#pragma once
#include <vector>
#include "RegressionForest.h"
#include "RegressionForest_EXPORTS.h"

namespace hiveRegressionForest
{
	REGRESSION_FOREST_EXPORTS unsigned int				hiveBuildRegressionForest(const std::string& vConfig);
	REGRESSION_FOREST_EXPORTS unsigned int              hiveRebuildRegressionForest(const std::string& vConfig, const std::string& vFilePath);
	
	//for single response
	REGRESSION_FOREST_EXPORTS void				 		hivePredict(const std::vector<std::vector<float>>& vTestFeatureSet, const std::vector<float>& vTestResponseSet, std::vector<float>& voPredictSet, unsigned int vForestId);
	REGRESSION_FOREST_EXPORTS void				 		hivePrePredict(const std::vector<std::vector<float>>& vOOBFeatureSet, const std::vector<float>& vOOBResponseSet, unsigned int vForestId);

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