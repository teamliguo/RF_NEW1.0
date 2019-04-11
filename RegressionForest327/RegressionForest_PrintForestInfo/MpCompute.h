#pragma once
#include <string>
#include <vector>
#include "Tree.h"
#include "Node.h"
#include "common/BaseProduct.h"

namespace hiveRegressionForest
{
	class CMpCompute : public hiveOO::CBaseProduct
	{
	public:
		CMpCompute();
		~CMpCompute();

		float					computeMpOfTwoFeatures(const CTree* vTree, const std::vector<float>& vLeafDate, const std::vector<float>& vTestData, float vPredictResponse = 0.f);
		float					calMPDissimilarityGlobal(const CTree* vTree, const std::vector<int>& vLeafIndex, const std::vector<float>& vFeature, float vPredictResponse);
		float					calMPOutOfFeatureAABB(const CTree* vTree, const CNode* vNode, const std::vector<float>& vFeature);
		void					generateSortedFeatureResponsePairSet(const std::vector<std::vector<float>>& vFeatureSet, const std::vector<float>& vResponseSet, unsigned int vFeatureIndex, std::vector<std::pair<float, float>>& voSortedFeatureResponseSet);
		std::pair<int, float>	calMinMPAndIndex(const CTree* vTree, const std::vector<int>& vDataIndex, const std::vector<float>& vFeature);

	private:
		void					__countIntervalNode(const CTree* vTree, const std::vector<std::pair<float, float>>& vMaxMinValue, std::vector<int>& voIntervalCount, std::vector<std::pair<float, float>>& voInterResponseRange);
		float					__calMPValue(const CTree* vTree, const std::vector<std::pair<float, float>>& vMaxMinValue);
	};
}