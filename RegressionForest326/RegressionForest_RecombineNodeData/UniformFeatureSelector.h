#pragma once
#include "BaseFeatureSelector.h"

namespace hiveRegressionForest
{
	class CUniformFeatureSelector : public IFeatureSelector
	{
	public:
		CUniformFeatureSelector();
		~CUniformFeatureSelector();

		void generateFeatureIndexSetV(unsigned int vFeatureNum, std::vector<int>& voFeatureIndexSubset, const std::vector<float>& vWeightSet) override;
	};
}