#pragma once
#include "BasePredictionMethod.h"

namespace hiveRegressionForest
{
	class CVariancePredictionMethod : public IPredictionMethod
	{
	public:
		CVariancePredictionMethod() {};
		~CVariancePredictionMethod() {};

		virtual float predictCertainResponseV(const std::vector<float>& vTestFeatureInstance, float vTestResponse, unsigned int vTreeNumber) override;
	};
}