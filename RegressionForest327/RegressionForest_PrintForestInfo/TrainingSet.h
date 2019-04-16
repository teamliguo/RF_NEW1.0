#pragma once
#include <vector>
#include <string>
#include "RegressionForest_EXPORTS.h"
#include "common/Singleton.h"
#include "RegressionForestConfig.h"
#include "BaseBootstrapSelector.h"

namespace hiveRegressionForest
{
	class REGRESSION_FOREST_EXPORTS CTrainingSet : public hiveOO::CSingleton<CTrainingSet>
	{
	public:
		~CTrainingSet();

		bool loadTrainingSet(const std::string& vConfig, bool vHeader = false);

		//NOTES : vBootstrapIndexRange���ڱ���ȡ��vBootstrapIndexSet����Щ��Χ�����ݣ�first�ǵ�һ��Ԫ�أ�second�������һ��Ԫ�ص���һ��λ��;���ȫ��ȡ��������{0��vBootStrapIndexSet.size()}
		void recombineBootstrapDataset(const std::vector<int>& vBootstrapIndexSet, const std::pair<int, int>& vBootstrapIndexRange, std::pair<std::vector<std::vector<float>>, std::vector<float>>& voFeatureResponseSet);
		void recombineBootstrapDataset(const std::vector<int>& vBootstrapIndexSet, const std::pair<int, int>& vBootstrapIndexRange, std::vector<int>& voRangeIndex); 
		void dumpFeatureValueSetAt(const std::vector<int>& vInstanceIndexSet, unsigned int vFeatureIndex, std::vector<float>& voValueSet);

		int							getNumOfInstances() const { return m_FeatureSet.size(); }
		int							getNumOfFeatures() const { return m_FeatureSet[0].size(); }
		int							getNumOfResponse() const { return m_NumResponse; }
		float						getFeatureValueAt(unsigned int vInstanceIndex, unsigned int vFeatureIndex) const { return m_FeatureSet[vInstanceIndex][vFeatureIndex]; }
		float						getResponseValueAt(unsigned int vInstanceIndex, unsigned int vResponseIndex = 0) const { return m_pResponseSet[vInstanceIndex * m_NumResponse + vResponseIndex]; }
		const std::vector<float>&	getFeatureInstanceAt(unsigned int vInstanceIndex) const { return m_FeatureSet[vInstanceIndex]; }
		void						normalization(std::vector<std::vector<float>>& voFeatureSet);

	private:
		CTrainingSet();

		void __initTrainingSetConfig(const std::string& vConfig);
		bool __loadSetFromBinaryFile(const std::string& vBinaryFile);
		bool __loadSetFromCSVFile(const std::string& vCSVFile, bool vHeader);

		//NOTES : m_FeatureSet�ö�άvector�洢���ܾ������⿽����m_ResponseSet��ָ�룬Ӧ�Ե���Ӧ�Ͷ���Ӧ�����, ��Ч�ʸ���
		std::vector<std::vector<float>>							m_FeatureSet;
		float*  m_pResponseSet	 = nullptr;
		int		m_NumResponse    = 0;

		std::vector<float> m_FeatureMean, m_FeatureStd;

		friend class hiveOO::CSingleton<CTrainingSet>;
	};
}