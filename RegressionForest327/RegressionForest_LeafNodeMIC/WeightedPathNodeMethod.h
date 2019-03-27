#pragma once
#include <vector>
#include "Tree.h"
#include "Node.h"

namespace hiveRegressionForest
{
	class CWeightedPathNodeMethod : public hiveOO::CSingleton<CWeightedPathNodeMethod>
	{
	public:
		float calOutNodeBound(const std::vector<float>& vFeature, const std::pair<std::vector<float>, std::vector<float>>& vFeatureRange);
		float calEuclideanDistanceFromNodeBound(const std::vector<float>& vFeature, const std::pair<std::vector<float>, std::vector<float>>& vFeatureRange);
		float traversalPathPrediction(const CTree* vTree, const std::vector<float>& vFeatures);
		float traverWithDistanceFromFeatureRange(const CTree* vTree, const std::vector<float>& vFeatures);
		float traversePathWithFeatureCentre(const CTree* vTree, const std::vector<float>& vFeatures);
		float traverWithDistanceFromFeaturesCentre(const CTree* vTree, const std::vector<float>& vFeatures);
		float predictWithMonteCarlo(const CNode& vCurLeafNode, const std::vector<float>& vFeatures);
		float predictWithMinMPOnWholeDimension(const CTree* vTree, const std::vector<float>& vFeatures);//add-gss-1.16

	private:
		float __computeCDF(float vFirst, float vSecond);
		bool __isTotalInBoundBox(const std::vector<float>& vTestPoint, const std::vector<float>& vLow, const std::vector<float>& vHeigh);//add-gss-1.16
		bool __isOneMoreInBoundBox(const std::vector<float>& vTestPoint, const std::vector<float>& vLow, const std::vector<float>& vHeigh);//add-gss-1.17
		std::vector<int> __calInterNodeDataIndex(const CNode* vNode);//add-gss-1.16
	};
}
