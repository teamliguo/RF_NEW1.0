#pragma once
#include "BasePredictionMethod.h"

namespace hiveRegressionForest
{
	class CMeanPredictionMethod : public IPredictionMethod
	{
	public:
		CMeanPredictionMethod() {};
		~CMeanPredictionMethod() {};

		virtual float predictCertainResponseV(const std::vector<float>& vTestFeatureInstance, float vTestResponse, unsigned int vTreeNumber) override;
	};
}