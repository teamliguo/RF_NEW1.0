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
		virtual float predictCertainTestV(const std::vector<float>& vTestFeatureInstance, float vTestResponse, const std::vector<CTree*>& vTreeSet) = 0;
		virtual void  prePredictOOBDataV(const std::vector<std::vector<float>>& vOOBFeatureSet, const std::vector<float>& vOOBResponseSet, const std::vector<CTree*>& vTreeSet) {}
		virtual float predictCertainTestBlockV(const std::vector<float>& vTestFeatureInstance, float vTestResponse, const std::vector<CTree*>& vTreeSet) { return 0.f; }
	};
}