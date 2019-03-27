#pragma once
#include <boost/format.hpp>
#include <boost/function.hpp>
#include "common/BaseProduct.h"
#include "Node.h"

namespace hiveRegressionForest
{
	struct SSplitHyperplane
	{
		SSplitHyperplane() = default;
		SSplitHyperplane(const std::pair<int, float>& vAxisAlignedSplitHyperplane) : m_AxisAlignedSplitHyperplane(vAxisAlignedSplitHyperplane) {}
		~SSplitHyperplane() = default;

		bool IsInstanceInLeftSpace(const std::vector<float>& vFeature)
		{
			return vFeature[m_AxisAlignedSplitHyperplane.first] < m_AxisAlignedSplitHyperplane.second;
		}

		std::pair<int, float> m_AxisAlignedSplitHyperplane = std::pair<int, float>(-1, FLT_MAX);
	};

	class REGRESSION_FOREST_EXPORTS INodeSpliter : public hiveOO::CBaseProduct
	{
	public:
		INodeSpliter();
		virtual ~INodeSpliter();

		bool splitNode(CNode* vCurrentNode, const std::pair<int, int>& vBootstrapRange, std::vector<int>& vBootstrapIndex, const std::vector<int>& vCurrentFeatureIndexSubSet, int& voRangeSplitPos);

	protected:
		virtual void _generateSortedFeatureResponsePairSetV(std::vector<int>& vBootstrapIndex, const std::pair<int, int>& vBootstrapRange, unsigned int vFeatureIndex, std::vector<std::pair<float, float>>& voSortedFeatureResponseSet);
		
	private:
		virtual void __findLocalBestSplitHyperplaneV(const std::vector<std::pair<float, float>>& vFeatureResponseSet, float vSum, float& voCurrentFeatureMaxObjVal, float& voCurBestGap) = 0;
	
		void __findBestSplitHyperplane(std::vector<int>& vBootstrapIndex, const std::pair<int, int>& vBootstrapRange, const std::vector<int>& vFeatureIndexSubset, SSplitHyperplane& voSplitHyperplane);
		int __processBootstrapRange(std::vector<int>& vBootstrapIndex, const std::pair<int, int>& vBootstrapRange, SSplitHyperplane& vSplitHyperplane);
	};
}