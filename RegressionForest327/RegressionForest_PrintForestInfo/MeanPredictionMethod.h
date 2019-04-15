#pragma once
#include "BasePredictionMethod.h"

namespace hiveRegressionForest
{
	class CMeanPredictionMethod : public IBasePredictionMethod
	{
	public:
		CMeanPredictionMethod() {};
		~CMeanPredictionMethod() {};

		virtual float predictCertainResponseV(const std::vector<float>& vTestFeatureInstance, float vTestResponse, const std::vector<CTree*>& vTreeSet) override;
	};
}