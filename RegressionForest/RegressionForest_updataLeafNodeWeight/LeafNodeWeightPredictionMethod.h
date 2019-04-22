#pragma once
#include "BasePredictionMethod.h"

namespace hiveRegressionForest
{
	class CLeafNodeWeightPredictionMethod : public IBasePredictionMethod
	{
	public:
		CLeafNodeWeightPredictionMethod() {};
		~CLeafNodeWeightPredictionMethod() {};

		virtual float predictCertainTestV(const std::vector<float>& vTestFeatureInstance, float vTestResponse, const std::vector<CTree*>& vTreeSet) override;
	private:
		void __updataNodeWeigth(std::vector<const CNode*>& vioLeafNodeSet, const std::vector<float>& vLeafNodeBias, float vTestResponse);
	};
}

