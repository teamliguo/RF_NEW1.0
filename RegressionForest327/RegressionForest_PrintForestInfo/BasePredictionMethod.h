#pragma once
#include "common/BaseProduct.h"
#include "Tree.h"

namespace hiveRegressionForest
{
	class IBasePredictionMethod : public hiveOO::CBaseProduct
	{
	public:
		IBasePredictionMethod() {}
		virtual ~IBasePredictionMethod() {}
		virtual float predictCertainResponseV(const std::vector<float>& vTestFeatureInstance, float vTestResponse, const std::vector<CTree*>& vTreeSet) = 0;
	};
}