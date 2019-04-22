#pragma once
#include <vector>
#include "common/BaseProduct.h"

namespace hiveRegressionForest
{
	class IFeatureWeightGenerator : public hiveOO::CBaseProduct
	{
	public:
		IFeatureWeightGenerator() {}
		virtual ~IFeatureWeightGenerator() {}
		
		void generateFeatureWeight(const std::vector<std::vector<float>>& vInstanceSet, const std::vector<float>& vResponseSet, std::vector<std::pair<unsigned int, float>>& voFeatureWeightSet)
		{
			__calculateFeatureWeightV(vInstanceSet, vResponseSet, voFeatureWeightSet);
		}

		void generateFeatureWeight(const std::vector<int>& vFeaturesInvokingNum, std::vector<float>& voFeatureWeightNormalized)
		{
			__calculateFeatureWeightV(vFeaturesInvokingNum, voFeatureWeightNormalized);
		}

	private:
		virtual void __calculateFeatureWeightV(const std::vector<std::vector<float>>& vInstanceSet, const std::vector<float>& vResponseSet, std::vector<std::pair<unsigned int, float>>& voFeatureWeightSet) = 0;
		virtual void __calculateFeatureWeightV(const std::vector<int>& vFeaturesInvokingNum, std::vector<float>& voFeatureWeightNormalized) = 0;
	};
}