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

		virtual void createAsLeafNodeV(const std::pair<std::vector<std::vector<float>>, std::vector<float>>& vBootstrapDataset) {}
		
		//NOTES : �˴�ʹ��Ĭ�ϲ���vResponseIndex�����㵥��Ӧ�Ͷ���Ӧ������
		virtual float predictV(const std::vector<float>& vFeatureInstance, unsigned int vResponseIndex = 0) const { return FLT_MAX; }
		virtual float getNodeMeanV(unsigned int vResponseIndex = 0) const { return FLT_MAX; } //11.27-gss
		virtual void calStatisticsV(const std::pair<std::vector<std::vector<float>>, std::vector<float>>& vBootstrapDataset) {}; //11.28-gss
		float calculateNodeWeight(unsigned int vResponseIndex = 0) const;

		std::pair<std::vector<float>, std::vector<float>> calFeatureRange(const std::vector<std::vector<float>>& vFeatureSet); //11.27-gss

		bool			operator==(const CNode& vNode) const;
		bool			isLeafNode() const { return m_IsLeafNode; }
		bool			isUnfitted() const;
		
		bool			setLeftChild(CNode* vNode);
		bool			setRightChild(CNode* vNode);
		void			setLevel(unsigned int vLevel) { m_Level = vLevel; }
		void			setNodeSize(unsigned int vNodeSize) { m_NodeSize = vNodeSize; }
		void			setBestSplitFeatureAndGap(unsigned int vFeatureIndex, float vGap) { m_BestSplitFeatureIndex = vFeatureIndex; m_BestGap = vGap; }
		
		const CNode&	getLeftChild()				const { return *m_pLeftChild; }
		const CNode&	getRightChild()				const { return *m_pRightChild; }
		unsigned int	getLevel()					const { return m_Level; }
		unsigned int	getNodeSize()				const { return m_NodeSize; }		
		float			getBestGap()				const { return m_BestGap; }		
		unsigned int	getBestSplitFeatureIndex()	const { return m_BestSplitFeatureIndex; }

		std::pair<std::vector<float>, std::vector<float>> getFeatureRange() const { return m_FeatureRange; } //11.27-gss
		
	protected:
		float _calculateVariance(const std::vector<float>& vNumVec);
		float _calculateMean(const std::vector<float>& vNumVec);
		virtual float _getNodeVarianceV(unsigned int vResponseIndex = 0) const { return FLT_MAX; }

		bool m_IsLeafNode = false;
		unsigned int m_Level = 0;
		unsigned int m_BestSplitFeatureIndex = 0;
		float m_BestGap = FLT_MAX;
		unsigned int m_NodeSize = 0;

		std::pair<std::vector<float>, std::vector<float>> m_FeatureRange; //11.27-gss

		CNode* m_pLeftChild = nullptr;
		CNode* m_pRightChild = nullptr;

	private:
		template <typename Archive>
		void serialize(Archive & ar, const unsigned int version)
		{			
			ar & m_IsLeafNode;
			ar & m_Level;
			ar & m_BestSplitFeatureIndex;
			ar & m_BestGap;
			ar & m_NodeSize;			
			ar & m_pLeftChild;
			ar & m_pRightChild;
		}

		friend class boost::serialization::access;
	};
}