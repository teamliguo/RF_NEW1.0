#pragma once
#include "BasePredictionMethod.h"

namespace hiveRegressionForest
{
	class CMPPredictionMethod : public IPredictionMethod
	{
	public:
		CMPPredictionMethod() {};
		~CMPPredictionMethod() {};

		virtual float predictCertainResponseV(const std::vector<float>& vTestFeatureInstance, float vTestResponse, unsigned int vTreeNumber) override;
	};
}