#pragma once
#include "BaseFeatureSelector.h"

namespace hiveRegressionForest
{
	class CWeightedFeatureSelector : public IFeatureSelector
	{
	public:
		CWeightedFeatureSelector();
		~CWeightedFeatureSelector();

		void generateFeatureIndexSetV(unsigned int vFeatureNum, std::vector<int>& voFeatureIndexSubset, const std::vector<float>& vWeightSet) override;
	};
}