#pragma once
#include "BasePredictionMethod.h"

namespace hiveRegressionForest
{
	class CInternalNodePredictionMethod : public IPredictionMethod
	{
	public:
		CInternalNodePredictionMethod() {};
		~CInternalNodePredictionMethod() {};

		virtual float predictCertainResponseV(const std::vector<float>& vTestFeatureInstance, float vTestResponse, unsigned int vTreeNumber) override;
	};
}