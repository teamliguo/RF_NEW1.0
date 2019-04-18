#pragma once
#include "BaseBootstrapSelector.h"

namespace hiveRegressionForest
{
	class CWeightedBootstrapSelector : public IBootstrapSelector
	{
	public:
		CWeightedBootstrapSelector();
		~CWeightedBootstrapSelector();

		void generateBootstrapIndexSetV(unsigned int vInstanceNum, std::vector<int>& voBootstrapIndexSet, const std::vector<float>& vWeightSet) override;
	};
}