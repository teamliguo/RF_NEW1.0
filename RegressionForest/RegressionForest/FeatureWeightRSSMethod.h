#pragma once
#include "BaseFeatureWeightMethod.h"

namespace hiveRegressionForest
{
	class CRSSFeatureWeightMethod : public IFeatureWeightGenerator
	{
	public:
		CRSSFeatureWeightMethod();
		~CRSSFeatureWeightMethod();

	private:
		virtual void __calculateFeatureWeightV(const std::vector<std::vector<float>>& vInstanceSet, const std::vector<float>& vResponseSet, std::vector<std::pair<unsigned int, float>>& voFeatureWeightSet) override;
		virtual void __calculateFeatureWeightV(const std::vector<int>& vFeaturesInvokingNum, std::vector<float>& voFeatureWeightNormalized) override {}
		void __getSortedFeatureResponsePairSet(unsigned int vFeatureIndex, const std::vector<std::vector<float>>& vInstanceSet, const std::vector<float>& vResponseSet, std::vector<std::pair<float, float>>& voFeatureResponseSet);
		float __calculateMaxObjectFuncValue(const std::vector<std::pair<float, float>>& vFeatureResponseSet, float vSumResponse);
	};
}