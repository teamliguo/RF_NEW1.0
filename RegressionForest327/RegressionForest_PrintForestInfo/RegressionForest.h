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
		void    outputPathNodeInfo(const std::string& vOutputFileName, const std::vector<SPathNodeInfo>& vPathNodeInfo) const;
		void    printAllInfo(const std::vector<float>& vFeatures, const std::vector<std::pair<int, float>>& voPredictBias, const std::vector<const CNode*>& vLeafNodeSet, const std::vector<std::vector<SPathNodeInfo>>& vAllTreePath, const std::vector<std::vector<float>>& vOutRange, const std::vector<std::vector<float>>& vOutSplitRange) const;
		void    outputOutFeatureRange(const std::string& vOutputFileName, const std::vector<float>& vOutRange) const;

		float   predict(const std::vector<float>& vFeatures, unsigned int vNumOfUsingTrees, bool vIsWeightedPrediction) const;
		float   predict(const std::vector<float>& vFeatures, float vResponse, unsigned int vNumOfUsingTrees, bool vIsWeightedPrediction, float& voMPPredictSet) const;
		void	predict(const std::vector<float>& vFeatures, unsigned int vNumOfUsingTrees, bool vIsWeightedPrediction, unsigned int vNumResponse, std::vector<float>& voPredictValue) const;
		
		const CTree* getTreeAt(int vTreeIndex) const { return m_Trees[vTreeIndex]; }
		
	private:
		float __predictCertainResponse(const std::vector<float>& vFeatures, unsigned int vNumOfUsingTrees, bool vIsWeightedPrediction, unsigned int vResponseIndex = 0) const;
		float __predictCertainResponse(const std::vector<float>& vFeatures, float vResponse, unsigned int vNumOfUsingTrees, bool vIsWeightedPrediction, float& voMPPredictSet, unsigned int vResponseIndex = 0) const;

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