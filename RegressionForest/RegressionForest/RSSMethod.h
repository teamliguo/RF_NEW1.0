#pragma once
#include "BaseSplitMethod.h"

namespace hiveRegressionForest
{
	class CResidualSumOfSquaresMethod : public INodeSpliter
	{
	public:
		CResidualSumOfSquaresMethod() = default;
		~CResidualSumOfSquaresMethod() = default;

	private:
		virtual void __findLocalBestSplitHyperplaneV(const std::vector<std::pair<float, float>>& vFeatureResponseSet, float vSum, float& voCurrentFeatureMaxObjVal, float& voCurBestGap) override;
	};
}