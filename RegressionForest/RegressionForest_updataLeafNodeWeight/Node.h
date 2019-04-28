#pragma once
#include "BaseFeatureSelector.h"
#include "RegressionForest_EXPORTS.h"
#include "TrainingSet.h"
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/serialization/vector.hpp>

namespace hiveRegressionForest
{
	class REGRESSION_FOREST_EXPORTS CNode : public hiveOO::CBaseProduct
	{
	public:
		CNode();
		CNode(unsigned int vLevel);
		~CNode();

		virtual void  createAsLeafNodeV(const std::pair<std::vector<std::vector<float>>, std::vector<float>>& vBootstrapDataset) {}
		virtual void  createAsLeafNodeV(const std::pair<std::vector<std::vector<float>>, std::vector<float>>& vBootstrapDataset, const std::vector<int>& vDataSetIndex, const std::pair<int, int>& vIndexRange) {}
		virtual void  calStatisticsV(const std::pair<std::vector<std::vector<float>>, const std::vector<float>>& vBootstrapDataset) {}
		virtual void  outputLeafNodeInfoV(const std::string& vFilePath) const {}
		virtual float predictV(const std::vector<float>& vFeatureInstance, unsigned int vResponseIndex = 0) const { return FLT_MAX; }//NOTES : 此处使用默认参数vResponseIndex以满足单响应和多响应的需求
		virtual float getNodeMeanV(unsigned int vResponseIndex = 0) const { return FLT_MAX; }
		virtual float getNodeVarianceV(unsigned int vResponseIndex = 0) const { return FLT_MAX; }

		float           calculateNodeWeight(unsigned int vResponseIndex = 0) const;
		bool			operator==(const CNode& vNode) const;
		bool			isLeafNode() const { return m_IsLeafNode; }
		bool			isUsed() const { return m_IsUsed; };
		bool			isUnfitted() const;
		void            calFeatureRange(const std::vector<std::vector<float>>& vFeatureSet, std::pair<std::vector<float>, std::vector<float>>& voFeatureRange);

		bool			setLeftChild(CNode* vNode);
		bool			setRightChild(CNode* vNode);
		void			setLevel(unsigned int vLevel) { m_Level = vLevel; }
		void			setNodeSize(unsigned int vNodeSize) { m_NodeSize = vNodeSize; }
		void			setBestSplitFeatureAndGap(unsigned int vFeatureIndex, float vGap) { m_BestSplitFeatureIndex = vFeatureIndex; m_BestGap = vGap; }
		void            setLeafNodeWeight(float vNodeWeight) { m_NodeWeight = vNodeWeight; }
		void			setIsUsed(bool vIsUsed) { m_IsUsed = vIsUsed; }

		unsigned int	getLevel()					const { return m_Level; }
		unsigned int	getNodeSize()				const { return m_NodeSize; }
		float           getNodeWeight()             const { return m_NodeWeight; }
		float			getBestGap()				const { return m_BestGap; }
		unsigned int	getBestSplitFeatureIndex()	const { return m_BestSplitFeatureIndex; }
		const CNode&	getLeftChild()				const { return *m_pLeftChild; }
		const CNode&	getRightChild()				const { return *m_pRightChild; }
		const std::pair<std::vector<float>, std::vector<float>>&   getFeatureRange()  const { return m_FeatureRange; }
		const std::vector<int>&                                    getNodeDataIndex() const { return m_DataSetIndex; }

	protected:
		bool             m_IsLeafNode = false;
		bool             m_IsUsed = false;
		float            m_NodeWeight = 0.f;
		float            m_BestGap = FLT_MAX;
		unsigned int     m_Level = 0;
		unsigned int     m_BestSplitFeatureIndex = 0;
		unsigned int     m_NodeSize = 0;
		CNode*           m_pLeftChild = nullptr;
		CNode*           m_pRightChild = nullptr;
		std::vector<int> m_DataSetIndex;
		std::pair<std::vector<float>, std::vector<float>> m_FeatureRange;

	private:
		template <typename Archive>
		void serialize(Archive & ar, const unsigned int version)
		{
			ar & m_IsLeafNode;
			ar & m_IsUsed;
			ar & m_NodeWeight;
			ar & m_Level;
			ar & m_BestSplitFeatureIndex;
			ar & m_BestGap;
			ar & m_NodeSize;
			ar & m_pLeftChild;
			ar & m_pRightChild;
			ar & m_FeatureRange;
			ar & m_DataSetIndex;
		}

		friend class boost::serialization::access;
	};
}