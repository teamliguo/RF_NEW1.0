#pragma once
#include "BasePredictionMethod.h"
#include <bitset>

namespace hiveRegressionForest
{
	class CLeafNodeWeightPredictionMethod : public IBasePredictionMethod
	{
	public:
		CLeafNodeWeightPredictionMethod() {};
		~CLeafNodeWeightPredictionMethod() {};

		virtual float predictCertainTestV(const std::vector<float>& vTestFeatureInstance, float vTestResponse, const std::vector<CTree*>& vTreeSet) override;
		virtual float predictCertainTestBlockV(const std::vector<float>& vTestFeatureInstance, float vTestResponse, const std::vector<CTree*>& vTreeSet) override;
		virtual void  prePredictOOBDataV(const std::vector<std::vector<float>>& vOOBFeatureSet, const std::vector<float>& vOOBResponseSet, const std::vector<CTree*>& vTreeSet) override;
 
	private:
		void __updataNodeWeigthByChangeThreshold(std::vector<const CNode*>& vioLeafNodeSet, const std::vector<float>& vLeafNodeBias, float vTestResponse, float vPredictValue);
		void __updataBlockWeigthByChangeThreshold(std::vector<const CNode*>& vioLeafNodeSet, const std::vector<float>& vLeafNodeBiasRatio, const std::vector<std::bitset<16>>& vBlockIdSet, float vTestResponse, float vPredictValue);
		void __updataNodeWeigthByConstantThreshold(std::vector<const CNode*>& vioLeafNodeSet, const std::vector<float>& vLeafNodeBiasRatio);
		void __calBlockId(const CNode* vLeafNode, const std::vector<float>& vTestFeatureInstance, std::bitset<16>& voBlockId);
		void __resetMember() { m_TestInstanceCount = 0; m_SumBiasRatio = 0.f; }

		int   m_TestInstanceCount = 0;
		float m_SumBiasRatio = 0.f;
	
	};
}

