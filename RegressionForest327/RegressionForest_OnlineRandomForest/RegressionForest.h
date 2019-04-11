#pragma once
#include <string>
#include <vector>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/scoped_ptr.hpp>
#include "RegressionForest_EXPORTS.h"
#include "Tree.h"
#include "BaseFeatureWeightMethod.h"
#include "common/BaseProduct.h"

namespace hiveRegressionForest
{
	class CRegressionForest : public hiveOO::CBaseProduct
	{
	public:
		CRegressionForest();
		~CRegressionForest();

		void buildForest(const std::string& vConfigFile);
		
		bool    operator==(const CRegressionForest& vRegressionForest) const;
		bool    isForestBuilt() const { return m_Trees.size() != 0; }
		int     getNumOfTrees() const { return m_Trees.size(); }
		void	outputForestInfo(const std::string& vOutputFileName) const;
		void	outputOOBInfo(const std::string& vOutputFileName) const;
		float   predict(const std::vector<float>& vFeatures, unsigned int vNumOfUsingTrees, float& voKnnPredictSet, float& voVarWiPerdict, bool vIsWeightedPrediction) const;
		float   predict(const std::vector<float>& vFeatures, unsigned int vNumOfUsingTrees, float& voLPPredictSet, float& voMPPredictSet, float& voLLPerdict, float& voLLPSet, float& voMPDissimilarity, float& voIFMPPerdict, bool vIsWeightedPrediction) const;
		void	predict(const std::vector<float>& vFeatures, unsigned int vNumOfUsingTrees, bool vIsWeightedPrediction, unsigned int vNumResponse, std::vector<float>& voPredictValue) const;
		
		const CTree* getTreeAt(int vTreeIndex) const { return m_Trees[vTreeIndex]; }
	
	private:
		//根据测试点加入叶子后引起的方差变化率计算相似度计算权重
		std::vector<float> __calWeightOfTree(int vNumOfUsingTrees, std::vector<std::vector<float>>&  wiFeatsChangedRatioOfTree)const;//add by zy410
		//根据Y的方差 、测试点加入叶子后引起的方差变化率计算相似度计算权重
		std::vector<float> __calWeightOfTree(int vNumOfUsingTrees, std::vector<float>& wiVarOfTree, std::vector<std::vector<float>>&  wiFeatsChangedRatioOfTree, int numOfWi)const;//add by zy410
		std::vector<float> __calWeightOfTree(int vNumOfUsingTrees, std::vector<std::vector<float>>&  wiFeatsOfTree, std::vector<float>& wiVarOfTree, std::vector<float> wiDisOfTree)const;//add by zy409
		std::vector<float> __calWeightOfTree(int vNumOfUsingTrees, std::vector<std::vector<float>>&  wiFeatsOfTree, std::vector<float>& wiVarOfTree, int numOfWi)const;//add by zy409
		std::vector<float> __calWeightOfTree(int vNumOfUsingTrees, std::vector<std::vector<float>>&  wiFeatsOfTree, std::vector<float>& wiVarOfTree, std::vector<float> wiDisOfTree, int numOfWi)const;//add by zy409
		std::vector<float> __calWeightOfTree(int vNumOfUsingTrees, std::vector<std::vector<float>>&  wiFeatsOfTree, std::vector<float>& wiVarOfTree)const;//add by zy 409
		std::vector<float> __calWeightOfTree(int vNumOfUsingTrees, std::vector<std::vector<float>>&  wiFeatsOfTree, std::vector<float>& wiVarOfTree, std::vector<std::vector<float>>&  wiFeatsChangedRatioOfTree, int numOfWi)const;//add by zy 409
		float __predictCertainResponse(const std::vector<float>& vFeatures, unsigned int vNumOfUsingTrees, bool vIsWeightedPrediction, float& voLPPredictSet, float& voMPPredictSet, float& voLLPerdict, float& voLLPSet, float& voMPDissimilarity, float& voIFMPPerdict, unsigned int vResponseIndex = 0) const;
		float __predictByMpWeightTree(const std::vector<float>& vFeatures, unsigned int vNumOfUsingTrees, float& voMpWeightTreePredict, unsigned int vResponseIndex) const;//add by zy 2019.3.29
		float __predictByVarWeightTree(const std::vector<float>& vFeatures, unsigned int vNumOfUsingTrees, float& voLpWeightTreePredict, unsigned int vResponseIndex) const;//add by zy 2019.3.29
		float __predictByKMeansForest(const std::vector<float>& vFeatures, unsigned int vNumOfUsingTrees, float& voLpWeightTreePredict, unsigned int vResponseIndex) const;//add by zy 2019.3.29
		void __initForest();
		void __initForestParameters(IBootstrapSelector*& voBootstrapSelector, IFeatureSelector*& voFeatureSelector, INodeSpliter*& voNodeSpliter, IBaseTerminateCondition*& voTerminateCondition, IFeatureWeightGenerator*& voFeatureWeightMethod);
 		void __clearForest();
	
		std::vector<CTree*> m_Trees;
		float m_OOBError = -1.0f;
		
		template <typename Archive>
		void serialize(Archive & ar, const unsigned int version)
		{
			ar & m_Trees;
		}

		friend class boost::serialization::access;
	};
}