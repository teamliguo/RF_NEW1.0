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

		void  buildForest(const std::string& vConfigFile);
		void  reParseConfig(const std::string& vConfigFile);
		void  outputForestInfo(const std::string& vOutputFileName) const;
		void  outputOOBInfo(const std::string& vOutputFileName) const;
		bool  isForestBuilt() const { return m_Trees.size() != 0; }
		bool  operator==(const CRegressionForest& vRegressionForest) const;

		int                        getNumOfTrees() const { return m_Trees.size(); }
		const CTree*               getTreeAt(int vTreeIndex) const { return m_Trees[vTreeIndex]; }
		const std::vector<CTree*>& getTreeSet() const { return m_Trees; }

		void  predict(const std::vector<float>& vFeatures, unsigned int vNumOfUsingTrees, bool vIsWeightedPrediction, unsigned int vNumResponse, std::vector<float>& voPredictValue) const;
		void  predict(const std::vector<std::vector<float>>& vTestFeatureSet, const std::vector<float>& vTestResponseSet, std::vector<float>& voPredictSet) const;
		void  prePredict(const std::vector<std::vector<float>>& vOOBFeatureSet, const std::vector<float>& vOOBResponseSet) const;

	private:
		float __predictCertainResponse(const std::vector<float>& vFeatures, unsigned int vNumOfUsingTrees, bool vIsWeightedPrediction, unsigned int vResponseIndex = 0) const;
		void  __initForest();
		void  __initForestParameters(IBootstrapSelector*& voBootstrapSelector, IFeatureSelector*& voFeatureSelector, INodeSpliter*& voNodeSpliter, IBaseTerminateCondition*& voTerminateCondition, IFeatureWeightGenerator*& voFeatureWeightMethod);
 		void  __clearForest();
		
		std::vector<CTree*> m_Trees;
		float               m_OOBError = -1.0f;
		
		template <typename Archive>
		void serialize(Archive & ar, const unsigned int version)
		{
			ar & m_Trees;
		}

		friend class boost::serialization::access;
	};
}