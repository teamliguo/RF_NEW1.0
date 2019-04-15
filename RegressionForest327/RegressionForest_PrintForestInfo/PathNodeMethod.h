#pragma once
#include <vector>
#include "Tree.h"
#include "Node.h"

namespace hiveRegressionForest
{
	class CPathNodeMethod : public hiveOO::CSingleton<CPathNodeMethod>
	{
	public:
		float calOutNodeBound(const std::vector<float>& vFeature, const std::pair<std::vector<float>, std::vector<float>>& vFeatureRange);
		float calEuclideanDistanceFromNodeBound(const std::vector<float>& vFeature, const std::pair<std::vector<float>, std::vector<float>>& vFeatureRange);
		float traversalPathPrediction(const CTree* vTree, const std::vector<float>& vFeature);
		float traverWithDistanceFromFeatureRange(const CTree* vTree, const std::vector<float>& vFeature);
		float traversePathWithFeatureCentre(const CTree* vTree, const std::vector<float>& vFeature);
		float traverWithDistanceFromFeaturesCentre(const CTree* vTree, const std::vector<float>& vFeature);
		float predictWithMonteCarlo(const CNode& vCurLeafNode, const std::vector<float>& vFeature);
		float predictWithMinMPOnWholeDimension(const CTree* vTree, const std::vector<float>& vFeature);
		float prediceWithInternalNode(const CTree* vTree, const std::vector<float>& vFeature);

	private:
		float __computeCDF(float vFirst, float vSecond);
		float __calInternalNodeMeanValue(const std::vector<int>& vDataSetIndexSet);
		bool  __isTotalInBoundBox(const std::vector<float>& vTestPoint, const std::vector<float>& vLow, const std::vector<float>& vHeigh);
		bool  __isOneMoreOutBoundRange(const std::vector<float>& vTestPoint, const std::vector<float>& vLow, const std::vector<float>& vHeigh, int vOutDimension);
		std::vector<int> __calInternalNodeDataIndex(const CNode* vNode);
	};
}
