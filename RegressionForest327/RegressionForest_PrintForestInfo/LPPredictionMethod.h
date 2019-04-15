#pragma once
#include "BasePredictionMethod.h"

namespace hiveRegressionForest
{
	class CLPPredictionMethod : public IBasePredictionMethod
	{
	public:
		CLPPredictionMethod() {};
		~CLPPredictionMethod() {};

		virtual float predictCertainResponseV(const std::vector<float>& vTestFeatureInstance, float vTestResponse, const std::vector<CTree*>& vTreeSet) override;
	
	private:
		float __calEuclideanDistance(std::pair<std::vector<float>, std::vector<float>>& vFeatureRange, const std::vector<float>& vTestFeatureInstance);
	};
}