#pragma once
#include "BaseSplitMethod.h"

namespace hiveRegressionForest
{
	class CRelativeVarianceMethod : public INodeSpliter
	{
	public:
		CRelativeVarianceMethod() = default;
		~CRelativeVarianceMethod() = default;

	private:
		void __findBestSplitHyperplaneV(std::vector<int>& vBootstrapIndex, const std::pair<int, int>& vBootstrapRange, const std::vector<int>& vFeatureIndexSubset, SSplitHyperplane& voSplitHyperplane) override;
		
		void __calculateVarianceAndMeanVal(const std::vector<float>& vInput, float& voVariance, float& voMeanVal);
	};
}