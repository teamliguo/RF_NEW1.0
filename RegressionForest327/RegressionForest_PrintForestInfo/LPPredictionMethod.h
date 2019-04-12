#pragma once
#include "BasePredictionMethod.h"

namespace hiveRegressionForest
{
	class CLPPredictionMethod : public IPredictionMethod
	{
	public:
		CLPPredictionMethod() {};
		~CLPPredictionMethod() {};

		virtual float predictCertainResponseV(const std::vector<float>& vTestFeatureInstance, float vTestResponse, unsigned int vTreeNumber) override;
	};
}