#pragma once
#include <vector>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/serialization/vector.hpp>
#include "Node.h"
#include "BaseTerminateCondition.h"
#include "BaseSplitMethod.h"
#include "BaseFeatureWeightMethod.h"
#include "RegressionForest_EXPORTS.h"

namespace hiveRegressionForest
{
 	class IBootstrapSelector;

	struct STreeInfo
	{
		int m_NumOfNodes{ 0 };
		int m_NumOfUnfittedLeafNodes{ 0 };
		int m_NumOfLeafNodes{ 0 };
		
		std::vector<int> m_FeatureSplitTimes;
		std::vector<int> m_InstanceOOBTimes;
	};

	class REGRESSION_FOREST_EXPORTS CTree : public hiveOO::CBaseProduct
	{
	public:
		CTree();
		virtual ~CTree();

		void buildTree(IBootstrapSelector* vBootstrapSelector, IFeatureSelector* vFeatureSelector, INodeSpliter* vNodeSpliter, IBaseTerminateCondition* vTerminateCondition, IFeatureWeightGenerator* vFeatureWeightMethod);
		
		const CNode* locateLeafNode(const std::vector<float>& vFeatures) const;
		float predict(const CNode& vCurLeafNode, const std::vector<float>& vFeatures, float& voWeight, unsigned int vResponseIndex = 0) const;
		
		void fetchTreeInfo(STreeInfo& voTreeInfo) const;
		const CNode& getRoot() const { return *m_pRoot; }
		const std::vector<int>& getOOBIndexSet() const { _ASSERTE(!m_OOBIndexSet.empty()); return m_OOBIndexSet; }
		
		bool operator==(const CTree& vTree) const;

		float traversalPathPrediction(const std::vector<float>& vFeatures);
		float calculateOutNodeBound(const std::vector<float>& vFeature, const std::pair<std::vector<float>, std::vector<float>>& vFeatureRange);
		float traverWithDistanceFromFeatureRange(const std::vector<float>& vFeatures);
		float traversePathWithFeatureCentre(const std::vector<float>& vFeatures);
		float traverWithDistanceFromFeaturesCentre(const std::vector<float>& vFeatures);
		float predictWithMonteCarlo(const CNode& vCurLeafNode, const std::vector<float>& vFeatures);
		float computeCDF(float vFirst, float vSecond);
		float predictWithMinMPOnWholeDimension(const std::vector<float>& vFeatures);

	protected:
		virtual void _selectCandidateFeaturesV(IFeatureSelector* vFeatureSelector, IFeatureWeightGenerator* vFeatureWeightMethod, bool vIsUpdatingFeaturesWeight, const std::pair<std::vector<std::vector<float>>, std::vector<float>>& vBootstrapDataset, std::vector<int>& voCandidateFeaturesIndex);
		
		CNode*				m_pRoot = nullptr;
		std::vector<int>	m_OOBIndexSet;

	private:
		void __createLeafNode(CNode* vCurNode, const std::vector<int>& vDataSetIndex, const std::pair<int, int>& vRange, const std::pair<std::vector<std::vector<float>>, std::vector<float>>& vBootstrapDataset);
		void __initTreeParameters(IBootstrapSelector* vBootstrapSelector, std::vector<int>& voBootstrapIndex);
		void __obtainOOBIndex(std::vector<int>& vBootStrapIndexSet);
		void __dumpAllTreeNodes(std::vector<const CNode*>& voAllTreeNodes) const;
		void __updateFeaturesWeight(IFeatureWeightGenerator* vFeatureWeightMethod, bool vIsLiveUpdating, const std::pair<std::vector<std::vector<float>>, std::vector<float>>& vBootstrapDataset, std::vector<float>& voFeaturesWeight);
		
		CNode*       __createNode(unsigned int vLevel);
		boost::any   __getTerminateConditionExtraParameter(const CNode* vNode);

		template <typename Archive>
		void serialize(Archive & ar, const unsigned int version)
		{
			ar & m_pRoot;
		}

		friend class boost::serialization::access;
	};
}
