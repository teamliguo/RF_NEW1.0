#pragma once
#include "BaseBootstrapSelector.h"

namespace hiveRegressionForest
{
	class CUniformBootstrapSelector : public IBootstrapSelector
	{
	public:
		CUniformBootstrapSelector();
		~CUniformBootstrapSelector();

		void generateBootstrapIndexSetV(unsigned int vInstanceNum, std::vector<int>& voBootstrapIndexSetconst, const std::vector<float>& vWeightSet) override;
	};
}