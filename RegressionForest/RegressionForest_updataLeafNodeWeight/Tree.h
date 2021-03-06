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

		void					buildTree(IBootstrapSelector* vBootstrapSelector, IFeatureSelector* vFeatureSelector, INodeSpliter* vNodeSpliter, IBaseTerminateCondition* vTerminateCondition, IFeatureWeightGenerator* vFeatureWeightMethod);		
		void					fetchTreeInfo(STreeInfo& voTreeInfo) const;
		void					setBootstrapIndex(std::vector<int>& vBootstrapIndex) { m_BootstrapIndex = vBootstrapIndex; }
		float					predict(const CNode& vCurLeafNode, const std::vector<float>& vFeatures, float& voWeight, unsigned int vResponseIndex = 0) const;
		const CNode*			locateLeafNode(const std::vector<float>& vFeatures) const;
		void					updateUnusedNodeWeight();

		const CNode&			getRoot() const { return *m_pRoot; }
		const std::vector<int>& getOOBIndexSet() const { _ASSERTE(!m_OOBIndexSet.empty()); return m_OOBIndexSet; }
		const std::vector<int>&	getBootstrapIndex() const { return m_BootstrapIndex; }
		const std::vector<std::vector<std::pair<float, float>>>& getSortedFeatureResponsePairSet() const { return m_SortedFeatureResponsePairSet; }

		bool                    operator==(const CTree& vTree) const;
	protected:
		virtual void _selectCandidateFeaturesV(IFeatureSelector* vFeatureSelector, IFeatureWeightGenerator* vFeatureWeightMethod, bool vIsUpdatingFeaturesWeight, const std::pair<std::vector<std::vector<float>>, std::vector<float>>& vBootstrapDataset, std::vector<int>& voCandidateFeaturesIndex);
		
		CNode*				m_pRoot = nullptr;
		std::vector<int>	m_OOBIndexSet;

	private:
		void __createLeafNode(CNode* vCurNode, const std::vector<int>& vDataSetIndex, const std::pair<int, int>& vRange, const std::pair<std::vector<std::vector<float>>, std::vector<float>>& vBootstrapDataset);
		void __initTreeParameters(IBootstrapSelector* vBootstrapSelector, std::vector<int>& voBootstrapIndex);
		void __obtainOOBIndex(std::vector<int>& vBootStrapIndexSet);
		void __dumpAllTreeNodes(std::vector<const CNode*>& voAllTreeNodes) const;
		void __dumpAllLeafNodes(std::vector<const CNode*>& voAllLeafNodes) const;
		void __updateFeaturesWeight(IFeatureWeightGenerator* vFeatureWeightMethod, bool vIsLiveUpdating, const std::pair<std::vector<std::vector<float>>, std::vector<float>>& vBootstrapDataset, std::vector<float>& voFeaturesWeight);
		void __sortFeatureResponsePairSet();
		void __generateSortedFeatureResponsePairSet(const std::vector<std::vector<float>>& vFeatureSet, const std::vector<float>& vResponseSet, unsigned int vFeatureIndex, std::vector<std::pair<float, float>>& voSortedFeatureResponseSet);

		CNode*       __createNode(unsigned int vLevel);
		boost::any   __getTerminateConditionExtraParameter(const CNode* vNode);

		std::vector<int> m_BootstrapIndex;
		std::vector<std::vector<std::pair<float, float>>> m_SortedFeatureResponsePairSet;
		template <typename Archive>
		void serialize(Archive & ar, const unsigned int version)
		{
			ar & m_pRoot;
			ar & m_OOBIndexSet;
			ar & m_BootstrapIndex;
			ar & m_SortedFeatureResponsePairSet;
		}

		friend class boost::serialization::access;
	};
}
