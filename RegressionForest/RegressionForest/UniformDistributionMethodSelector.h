#pragma once
#include "BaseSelector.h"

namespace hiveRegressionForest
{
	class CUniformDistributionMethod : public IBaseRandomSelector
	{
	public:
		CUniformDistributionMethod();
		virtual ~CUniformDistributionMethod();

		virtual void generateFeatureIndexSubsetV(unsigned int vFeatureNum, std::vector<int>& voFeatureIndexSubset, const std::vector<float>& vWeightSet = std::vector<float>()) override;
		virtual void generateBootstrapIndexSetV(unsigned int vInstanceNum, std::vector<int>& voBootstrapSet, const std::vector<float>& vWeightSet = std::vector<float>()) override;
	};
}