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
		float predictWithMinMPOnWholeDimension(const CTree* vTree, const std::vector<float>& vFeatures, std::vector<int>& voNodeDataIndex);//add-gss-1.16
		std::pair<float, float> predictWithMinMPAndLPOnWholeDimension(const CTree* vTree, const std::vector<float>& vFeatures, std::vector<int>& voNodeDataIndex, int vInstanceMethod, float vInstanceNumber, float vInstanceNumberRation);//add-ZY-2.18
		std::pair<float, float> predictWithMinMPAndLPAndIForestWay(const CTree* vTree, const std::vector<float>& vFeatures, std::vector<int>& voNodeDataIndex, int vInstanceMethod, float vInstanceNumber, float vInstanceNumberRation);//add-ZY-3.13
		
	private:
		float __computeCDF(float vFirst, float vSecond);
		bool __isOneMoreOutBoundRange(const std::vector<float>& vTestPoint, const std::vector<float>& vLow, const std::vector<float>& vHeigh, int vOutDimension);//add-gss-1.17
		std::vector<int> __calInterNodeDataIndex(const CNode* vNode);//add-gss-1.16
	};
}
