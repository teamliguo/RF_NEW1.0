#pragma once
#include "BasePredictionMethod.h"

namespace hiveRegressionForest
{
	class CVariancePredictionMethod : public IBasePredictionMethod
	{
	public:
		CVariancePredictionMethod() {};
		~CVariancePredictionMethod() {};

		virtual float predictCertainTestV(const std::vector<float>& vTestFeatureInstance, float vTestResponse, const std::vector<CTree*>& vTreeSet) override;

	private:
		void __calVarChangedRatio(const std::vector<std::vector<float>>& vData, const std::vector<float>& vAddValue, std::vector<float>& voNativeVar, std::vector<float>& voChangedRatio);
		void __calWeightByComponentVar(const std::vector<float>& vTreeVar, std::vector<float>& voWeight);
		void __calTreeWeightByFeatureVar(const std::vector<std::vector<float>>& vFeatureVar, std::vector<std::vector<float>>& voWeight);
		void __calFinalWeight(const std::vector<std::vector<float>>& vWeigthByFeature, const std::vector<std::vector<float>>& vWeightByFeatureWithTest, const std::vector<float>& vWeightByResponse, std::vector<float>& voWeight);
	};
}