#pragma once
#include "BasePredictionMethod.h"

namespace hiveRegressionForest
{
	class CMPPredictionMethod : public IBasePredictionMethod
	{
	public:
		CMPPredictionMethod() {};
		~CMPPredictionMethod() {};

		virtual float predictCertainResponseV(const std::vector<float>& vTestFeatureInstance, float vTestResponse, const std::vector<CTree*>& vTreeSet) override;
	};
}