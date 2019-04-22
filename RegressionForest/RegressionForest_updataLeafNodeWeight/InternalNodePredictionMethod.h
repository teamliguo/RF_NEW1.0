#pragma once
#include "BasePredictionMethod.h"

namespace hiveRegressionForest
{
	class CInternalNodePredictionMethod : public IBasePredictionMethod
	{
	public:
		CInternalNodePredictionMethod() {};
		~CInternalNodePredictionMethod() {};

		virtual float predictCertainTestV(const std::vector<float>& vTestFeatureInstance, float vTestResponse, const std::vector<CTree*>& vTreeSet) override;
	};
}