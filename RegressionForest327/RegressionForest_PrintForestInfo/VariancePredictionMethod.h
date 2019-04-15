#pragma once
#include "BasePredictionMethod.h"

namespace hiveRegressionForest
{
	class CVariancePredictionMethod : public IBasePredictionMethod
	{
	public:
		CVariancePredictionMethod() {};
		~CVariancePredictionMethod() {};

		virtual float predictCertainResponseV(const std::vector<float>& vTestFeatureInstance, float vTestResponse, const std::vector<CTree*>& vTreeSet) override;

	private:
		void  __calVarChangedRatio(const std::vector<std::vector<float>>& vData, const std::vector<float>& vAddValue, std::vector<float>& voNativeVar, std::vector<float>& voChangedRatio);
		std::vector<float> __calWeightByResponseVar(const std::vector<float>& vTreeVar, int vParadigmValue);
		std::vector<std::vector<float>> __calTreeWeightByFeatureVar(const std::vector<std::vector<float>>& vFeatureVar, int vParadigmValue);
		std::vector<float> __calFinalWeight(const std::vector<std::vector<float>>& vWeigthByFeature, const std::vector<std::vector<float>>& vWeightByFeatureWithTest, const std::vector<float>& vWeightByResponse);
	};
}