#pragma once
#include "BaseFeatureWeightMethod.h"

namespace hiveRegressionForest
{
	class CPearsonFeatureWeightMethod : public IFeatureWeightGenerator
	{
	public:
		CPearsonFeatureWeightMethod();
		~CPearsonFeatureWeightMethod();

	private:
		virtual void __calculateFeatureWeightV(const std::vector<std::vector<float>>& vInstanceSet, const std::vector<float>& vResponseSet, std::vector<std::pair<unsigned int, float>>& voFeatureWeightSet) override;
		virtual void __calculateFeatureWeightV(const std::vector<int>& vFeaturesInvokingNum, std::vector<float>& voFeatureWeightNormalized) override {}
	};
}