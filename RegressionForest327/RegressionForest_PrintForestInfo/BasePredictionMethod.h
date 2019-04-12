#pragma once
#include "common/BaseProduct.h"

namespace hiveRegressionForest
{
	class IPredictionMethod : public hiveOO::CBaseProduct
	{
	public:
		IPredictionMethod() {}
		virtual ~IPredictionMethod() {}
		virtual float predictCertainResponseV(const std::vector<float>& vTestFeatureInstance, float vTestResponse, unsigned int vTreeNumber) = 0;
	};
}