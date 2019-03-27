#pragma once
#include "NodeSpliter.h"
#include <tuple>
#include <vector>

namespace hiveRegressionForest 
{
	class CNodeSplitterOfInfoGain : public IBaseNodeSplitter
	{
	public:
		CNodeSplitterOfInfoGain();
		~CNodeSplitterOfInfoGain();

		virtual std::tuple<double, int> splitNodeV(const std::vector<std::vector<double>>& vFeatureSet, const std::vector<double>& vResponseInstance, const std::vector<int>& vFeatureIndexSubset, const std::vector<int>& Bootstrap, unsigned int vNumInstancesHold) override;

	private:
		void __generateFeatureResponsePairSet(const std::vector<std::vector<double>>& vFeatureSet, const std::vector<double>& vResponseInstance, unsigned int vFeatureIndex, const std::vector<int>& vBootstrap, std::vector<std::pair<double, double>>& voFeatureResponseSet, double& voSumResponse);
	};
}