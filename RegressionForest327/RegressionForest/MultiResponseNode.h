#pragma once
#include "Node.h"
#include <boost/serialization/base_object.hpp>

namespace hiveRegressionForest
{
	class CMultiResponseNode : public CNode
	{
	public:
		CMultiResponseNode();
		CMultiResponseNode(unsigned int vLevel) { m_Level = vLevel; }
		~CMultiResponseNode();

		virtual void createAsLeafNodeV(const std::pair<std::vector<std::vector<float>>, std::vector<float>>& vBootstrapDataset) override;
		virtual float predictV(const std::vector<float>& vFeatureInstance, unsigned int vResponseIndex) const override;
		
	protected:
		virtual float _getNodeVarianceV(unsigned int vResponseIndex = 0) const override;

	private:
		float* m_NodeAvgValuePtr = new float(0.0f);
		float* m_NodeVariancePtr = new float(0.0f);
		unsigned int m_NumResponse = 1;

	private:
		template <typename Archive>
		void serialize(Archive & ar, const unsigned int version)
		{
			ar & boost::serialization::base_object<CNode>(*this);

			ar & m_NumResponse;
			for (int i = 0; i < m_NumResponse; ++i)
				ar & m_NodeAvgValuePtr[i]; 

			for (int k = 0; k < m_NumResponse; ++k)
				ar & m_NodeVariancePtr[k];
		}

		friend class boost::serialization::access;
	};
}