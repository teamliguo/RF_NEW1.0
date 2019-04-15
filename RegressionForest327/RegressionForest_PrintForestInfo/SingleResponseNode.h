#pragma once
#include "Node.h"
#include "math/RegressionAnalysisInterface.h"
#include <boost/serialization/base_object.hpp>

namespace hiveRegressionForest
{
	class CSingleResponseNode : public CNode
	{
	public:
		CSingleResponseNode();
		CSingleResponseNode(unsigned int vLevel) { m_Level = vLevel; }
		~CSingleResponseNode();

		virtual void createAsLeafNodeV(const std::pair<std::vector<std::vector<float>>, std::vector<float>>& vBootstrapDataset) override;
		virtual void createAsLeafNodeV(const std::pair<std::vector<std::vector<float>>, std::vector<float>>& vBootstrapDataset, const std::vector<int>& vDataSetIndex, const std::pair<int, int>& vIndexRange) override;
		virtual float predictV(const std::vector<float>& vFeatureInstance, unsigned int vResponseIndex) const override;
		virtual float getNodeMeanV(unsigned int vResponseIndex = 0) const override;
		virtual void calStatisticsV(const std::pair<std::vector<std::vector<float>>, std::vector<float>>& vBootstrapDataset) override;
		virtual std::vector<int> getNodeDataIndexV() const override;

	protected:
		virtual float getNodeVarianceV(unsigned int vResponseIndex = 0) const override;

	private:
		float m_NodeVariance = 0.0f;
		float m_NodeMean = 0.0f; 
		std::vector<int> m_DataSetIndex;
		hiveRegressionAnalysis::IBaseRegression* m_pRegressionModel = nullptr;

	private:
		template <typename Archive>
		void serialize(Archive & ar, const unsigned int version)
		{
			ar & boost::serialization::base_object<CNode>(*this);
			ar & m_NodeMean;
			ar & m_NodeVariance;
			ar & m_pRegressionModel;
		}

		friend class boost::serialization::access;
	};
}
