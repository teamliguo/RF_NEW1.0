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

		//NOTES : vBootstrapIndexRange用于表明取出vBootstrapIndexSet中哪些范围的数据，first是第一个元素，second保存最后一个元素的下一个位置;如果全部取出，则是{0，vBootStrapIndexSet.size()}
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

		//NOTES : m_FeatureSet用二维vector存储，能尽量避免拷贝。m_ResponseSet用指针，应对单响应和多响应的情况, 且效率更高
		std::vector<std::vector<float>>							m_FeatureSet;
		float*  m_pResponseSet	 = nullptr;
		int		m_NumResponse    = 0;

		std::vector<float> m_FeatureMean, m_FeatureStd;

		friend class hiveOO::CSingleton<CTrainingSet>;
	};
}