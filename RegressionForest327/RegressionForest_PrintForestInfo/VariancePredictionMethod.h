#pragma once
#include "BasePredictionMethod.h"

namespace hiveRegressionForest
{
	class CVariancePredictionMethod : public IBasePredictionMethod
	{
	public:
		CVariancePredictionMethod() {};
		~CVariancePredictionMethod() {};

		virtual float predictCertainResponseV(const std::vector<float>& vTestFeatureInstance, float vTestResponse, const std::vector<CTree*>& vTreeSet) override;
	};
}