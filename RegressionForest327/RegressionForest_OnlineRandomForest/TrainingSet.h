#pragma once
#include <vector>
#include <string>
#include "RegressionForest_EXPORTS.h"
#include "common/Singleton.h"
#include "RegressionForestConfig.h"
#include "BaseBootstrapSelector.h"

namespace hiveRegressionForest
{
	class REGRESSION_FOREST_EXPORTS CTrainingSet : public hiveOO::CSingleton<CTrainingSet>
	{
	public:
		~CTrainingSet();
		  
		static bool comparePair(const std::pair<float, float>& vFirst, const std::pair<float, float>& vSecond);//Add_ljy_12/20
		static bool comparePair2(const std::pair<float, float>& vFirst, const std::pair<float, float>& vSecond);
		static bool cmpByIndexDesc(const std::vector<float> &a, const std::vector<float> &b, int index);//AddZY315
		bool loadTrainingSet(const std::string& vConfig, bool vHeader = false);//Fix_ljy_12/20

		//NOTES : vBootstrapIndexRange用于表明取出vBootstrapIndexSet中哪些范围的数据，first是第一个元素，second保存最后一个元素的下一个位置;如果全部取出，则是{0，vBootStrapIndexSet.size()}
		void recombineBootstrapDataset(const std::vector<int>& vBootstrapIndexSet, const std::pair<int, int>& vBootstrapIndexRange, std::pair<std::vector<std::vector<float>>, std::vector<float>>& voFeatureResponseSet);
		void dumpFeatureValueSetAt(const std::vector<int>& vInstanceIndexSet, unsigned int vFeatureIndex, std::vector<float>& voValueSet);
		void recombineBootstrapDataset(const std::vector<int>& vBootstrapIndexSet, const std::pair<int, int>& vBootstrapIndexRange, std::vector<int>& voRangeIndex);//Add_ljy_12/20
		
		int							getNumOfInstances() const { return m_FeatureSet.size(); }
		int							getNumOfFeatures() const { return m_FeatureSet[0].size(); }
		int							getNumOfResponse() const { return m_NumResponse; }
		float						getFeatureValueAt(unsigned int vInstanceIndex, unsigned int vFeatureIndex) const { return m_FeatureSet[vInstanceIndex][vFeatureIndex]; }
		float						getResponseValueAt(unsigned int vInstanceIndex, unsigned int vResponseIndex = 0) const { return m_pResponseSet[vInstanceIndex * m_NumResponse + vResponseIndex]; }
		const std::vector<float>&	getFeatureInstanceAt(unsigned int vInstanceIndex) const { return m_FeatureSet[vInstanceIndex]; }
		void                        normalization( std::vector<std::vector<float>>& voFeatureSet);//11.30-gss
		void                        standard(std::vector<std::vector<float>>& voFeatureSet);//zy327
		std::vector<std::pair<int, float>>   kMeans(std::vector<std::vector<float>>& voFeatureSet, int TreeNum, int clusterK, int iterTime, std::vector<int>&clusters);//zy327
		std::vector<std::pair<int, float>>   kMeansByDyCls(std::vector<std::vector<float>>& voFeatureSet, int TreeNum, int clusterK, int iterTime, std::vector<int>&clusters);//zy328
		std::vector<std::pair<int, float>>   kMeansByResponseVarRatio(std::vector<float>& oldClustersResponseBias, float maxResponseOfTree, float maxBiasRatio, const std::vector<float>& TotalNodeDataResponseSet, std::vector<std::vector<float>>& voFeatureSet, int TreeNum, int clusterK, int iterTime, std::vector<int>& Clusters);//zy401
		std::vector<float>                   calMeanOfClusters(int currentClusterNum, std::vector<int>&  Clusters, const std::vector<float>& TotalNodeDataResponseSet);//zy401
		std::vector<float>                   calResponseVar(int currentClusterNum, std::vector<int>&  Clusters, const std::vector<float>& TotalNodeDataResponseSet, float& maxResponseBias);//zy401
		void                        setCenterFLTMAX(std::vector<float>& FeatureCenterMax);//zy401
		float                       calLp(std::vector<float> voFeature, std::vector<float> objFeature);//zy327
		float                       calMp(std::vector<float> voFeature, std::vector<float> objFeature, std::vector<std::vector<float>>& vAllLeafFeatureSet, std::vector<std::vector<float>> vAllLeafFeatureSetChanged, std::vector<float> voEachDimStandard);//ZY401
		std::vector<bool>           isSameVandV(std::vector<std::vector<float>> currentMeanCenter, std::vector<std::vector<float>> oldMeanCenter);//zy327
		bool                        isSameVandV(std::vector<float> currentMeanCenter, std::vector<float> oldMeanCenter);//zy327
		const std::vector<float>&   getEachDimStandard() const { return m_EachDimStandard; }////Add_ljy_12/20
		float                       calMPDissimilarityGlobal(const std::vector<int>& vLeafIndex, const std::vector<float>& vFeature, float vPredictResponse);////Add_ljy_12/20
		void                        calDimFeatures(const std::vector<std::vector<float>>& vFeatureDataSet, std::vector<std::vector<float>>& voDimFeatureDataSet);////Add_ljy_12/20
		float                       calEuclideanDistance(const std::vector<int>& vLeafIndex, const std::vector<float>& vFeature); //add-gss-1.3
		float                       calEuclideanDistance(int vLeafIndex, const std::vector<float>& vFeature);//add-ZY-2.19
		std::pair<float, float>		getResponseRange() { return m_ResponseRange; }//add-gss-1.5
		std::pair<std::vector<float>, std::vector<float>> getFeatureRange() { return m_FeatureRange; }//add-gss-1.5
		std::pair<int, float>                   calMinMPAndIndex(const std::vector<int>& vDataIndex, const std::vector<float>& vFeature);
		std::vector<std::pair<int, float>>      calMinMPAndIndex(const std::vector<int>& vDataIndex, const std::vector<float>& vFeature, int vInstanceMethod, float vInstanceNumber, float vInstanceNumberRation);//add-gss-1.16
		std::vector<std::pair<int, float>>      calMinMPAndIndex(const std::vector<std::vector<float>>& vAllLeafFeatureSet, const std::vector<float>& vAllLeafResponseSet, const std::vector<int>& vDataIndex, const std::vector<float>& vFeature, int vInstanceMethod, float vInstanceNumber, float vInstanceNumberRation);//add-zy0219
		
		std::vector<std::pair<int, float>>       calMinLPAndIndex(const std::vector<int>& vDataIndex, const std::vector<float>& vFeature, int vInstanceMethod, float vInstanceNumber, float vInstanceNumberRation);//add-ZY-2.18
		std::vector<std::pair<float, float>> calReCombineDataMP(const std::vector<std::vector<float>>& vAllLeafFeatureSet, const std::vector<float>& vAllLeafResponseSet, const std::vector<float>& vFeatures, float vPredictResponse);
		std::vector<std::pair<float, float>> calReCombineDataLP(const std::vector<std::vector<float>>& vAllLeafFeatureSet, const std::vector<float>& vAllLeafResponseSet, const std::vector<float>& vFeatures, float vPredictResponse);
		std::vector<std::pair<int, float>>     calIfMp(unsigned int vNumOfUsingTrees, std::vector<int> TotalDataIndexByOrder, std::vector<std::vector<int>> NodeDataIndex);
		std::vector<std::pair<float, float>>   calmidMP(const std::vector<std::vector<float>>& vAllLeafFeatureSet, const std::vector<float>& vAllLeafResponseSet, const std::vector<float>& vFeatures, float vPredictResponse);//zy326
		std::vector<std::pair<float, float>>   calLeafAndBotherMP(const std::vector<std::vector<float>>& vLeafFeatureSet, const std::vector<float>& vLeafResponseSet,
			                                                      const std::vector<std::vector<float>>& vBotherFeatureSet, const std::vector<float>& vBotherResponseSet,
			                                                     const std::vector<std::vector<float>>& vAllLeafFeatureSet, const std::vector<float>& vAllLeafResponseSet,
			                                                     const std::vector<float>& vFeatures, float vPredictResponse);//zy326
		float   calLeafWeightForTreeByLp(std::vector<std::vector<float>> NodeDataFeature, const std::vector<float>& vFeatures);
		std::vector<std::pair<float, float>>  calLeafWeightForTreeByMp(std::vector<std::vector<float>>  treeNodeFeatureMid, std::vector<float> treeNodeDataResponse,
			                                                  std::vector<std::vector<float>> TotalNodeDataFeature, const std::vector<float>& TotalNodeDataResponseSet, const std::vector<float>& vFeatures);
		std::vector<float>                    antiDistanceWeight(std::vector<float> distance, int p);
		std::vector<float>                    antiVarWeight(std::vector<float> responseVar, int p);//zy402
		std::vector<std::vector<float>>                  antiFeatsWeight(std::vector<std::vector<float>> treeNodeFeatureVar, int p);//zy403
		float                                  calKMeansForest(std::vector<std::vector<float>>  treeNodeFeatureMid, std::vector<float> treeNodeDataResponse,
			                                                          std::vector<std::vector<float>> TotalNodeDataFeature, const std::vector<float>& TotalNodeDataResponseSet, const std::vector<float>& vFeatures);
		
	    float                                  calKMeansForestByResponseVarRatio(std::vector<float>& treeResponseBias, float maxResponseOfTree,float maxBiasRatio, std::vector<std::vector<float>>  treeNodeFeatureMid, std::vector<float> treeNodeDataResponse,
				                                                  std::vector<std::vector<float>> TotalNodeDataFeature, const std::vector<float>& TotalNodeDataResponseSet, const std::vector<float>& vFeatures);
		
		float                                  calTreeResponseVar(std::vector<float> NodeDataResponse);//zy401
		std::vector<float>                     calTreeFeaturesVar(const std::vector<float>& vFeatures, std::vector<std::vector<float>>& NodeDataFeature, std::vector<float>& FeatsVarChangedRatio);//zy409
		std::vector<float>                     calTreeFeaturesVar(std::vector<std::vector<float>>& NodeDataFeature);//zy403
		float                                  calLeafAndBotherKNNResponse(const std::vector<std::vector<float>>& vLeafFeatureSet, const std::vector<float>& vLeafResponseSet,
			const std::vector<std::vector<float>>& vBotherFeatureSet, const std::vector<float>& vBotherResponseSet,
			const std::vector<std::vector<float>>& vAllLeafFeatureSet, const std::vector<float>& vAllLeafResponseSet,
			const std::vector<float>& vFeatures, float vPredictResponse);
		std::vector<std::pair<float, float>> calMPwithMidPoint(const std::vector<float> &FeatureMid, const std::vector<float>& vFeatures, const std::vector<std::vector<float>>& vAllLeafFeatureSetChanged, 
			                                                 const std::vector<std::vector<float>>& vAllLeafFeatureSet, const std::vector<float>& vAllLeafResponseSet, float ResponseMid);
	private:
		CTrainingSet();
		void __countIntervalNode(const std::vector<std::pair<float, float>>& vMaxMinValue, std::vector<int>& voIntervalCount, std::vector<std::pair<float, float>>& voInterResponseRange);////Add_ljy_12/20
		void __calIntervalSample(const std::vector<std::pair<float, float>>& vMaxMinValue, const std::vector<std::vector<float>>& vAllLeafFeatureSet, const std::vector<float>& vAllLeafResponseSet, std::vector<int>& voIntervalCount, std::vector<std::pair<float, float>>& voInterResponseRange);
		void __calIntervalSampleByOrder(const std::vector<std::pair<float, float>>& vMaxMinValue, const std::vector<std::vector<float>>& vAllLeafFeatureSetOrdered, const std::vector<float>& vAllLeafResponseSet, std::vector<int>& voIntervalCount, std::vector<std::pair<float, float>>& voInterResponseRange);
		void __calIntervalSampleByMid(const std::vector<std::pair<float, float>>& vMaxMinValue, const std::vector<std::vector<float>>& vAllLeafFeatureSetOrdered, const std::vector<float>& vAllLeafResponseSet, std::vector<int>& voIntervalCount);
		void __calIntervalSampleByMid(const std::vector<std::pair<float, float>>& vMaxMinValue, const std::vector<std::vector<float>>& vAllLeafFeatureSetOrdered, std::vector<int>& voIntervalCount);//ZY401
		void __calStandardDeviation(const std::vector<std::vector<float>>& vFeatureDataSet, std::vector<float>& voEachDimStandard);//Add_ljy_12/20
		void __initTrainingSetConfig(const std::string& vConfig);
		bool __loadSetFromBinaryFile(const std::string& vBinaryFile);
		bool __loadSetFromCSVFile(const std::string& vCSVFile, bool vHeader);//Fix_ljy_12/20
		void __calResponseRange();//add-gss-1.4
		void __calFeatureRange();//add-gss-1.4
		float __calMPValue(const std::vector<float>& vLeafDate, const std::vector<float>& vTestData, float vPredictResponse = 0.f);
		float __calMPValue(const std::vector<std::vector<float>>& vAllLeafFeatureSet, const std::vector<float>& vAllLeafResponseSet, const std::vector<float>& vLeafDate, const std::vector<float>& vTestData, float vPredictResponse = 0.f);
		
		//NOTES : m_FeatureSet用二维vector存储，能尽量避免拷贝。m_ResponseSet用指针，应对单响应和多响应的情况, 且效率更高
		std::vector<std::vector<float>>          m_FeatureSet;
		std::vector<float>                       m_EachDimStandard;//Add_ljy_12/20

		std::pair<float, float>					 m_ResponseRange;//Add_ljy_12/20
		std::pair<std::vector<float>,std::vector<float>>     m_FeatureRange;//add-gss-1.4
		float*									 m_pResponseSet	 = nullptr;
		int										 m_NumResponse    = 0;

		std::vector<float>						 m_FeatureMean, m_FeatureStd;

		friend class hiveOO::CSingleton<CTrainingSet>;
	};
}