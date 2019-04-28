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
		virtual void prePredictOOBDataV(const std::vector<std::vector<float>>& vOOBFeatureSet, const std::vector<float>& vOOBResponseSet, const std::vector<CTree*>& vTreeSet) override;

	private:
		void __updataNodeWeigth(std::vector<const CNode*>& vioLeafNodeSet, const std::vector<float>& vLeafNodeBias, float vTestResponse);
		void __updataNodeWeigthByChangeThreshold(std::vector<const CNode*>& vioLeafNodeSet, const std::vector<float>& vLeafNodeBias, float vTestResponse, float vPredictValue);

		int   m_TestInstanceCount = 0;
		float m_SumBiasRatio = 0.f;
	};
}

