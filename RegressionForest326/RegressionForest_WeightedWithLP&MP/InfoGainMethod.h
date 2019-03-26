#pragma once
#include "BaseSplitMethod.h"
#include <vector>

namespace hiveRegressionForest
{
	class CInfoGainMethod : public INodeSpliter
	{
	public:
		CInfoGainMethod() = default;
		~CInfoGainMethod() = default;

	private:
		void __findLocalBestSplitHyperplaneV(const std::vector<std::pair<float, float>>& vFeatureResponseSet, float vSum, float& voCurrentFeatureMaxObjVal, float& voCurBestGap) override;
	};
}