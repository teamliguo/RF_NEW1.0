#include "NodeSplitterOfInfoGain.h"
#include "common/ProductFactory.h"

using namespace hiveRegressionForest;

hiveOO::CProductFactory<CNodeSplitterOfInfoGain> SplitSig("INFORMATION_GAIN_FUNCTION");

CNodeSplitterOfInfoGain::CNodeSplitterOfInfoGain()
{
}

CNodeSplitterOfInfoGain::~CNodeSplitterOfInfoGain()
{
}

//********************************************************************************************************
//FUNCTION:
std::tuple<double, int> CNodeSplitterOfInfoGain::splitNodeV(const std::vector<std::vector<double>>& vFeatureSet, const std::vector<double>& vResponseInstance, const std::vector<int>& vFeatureIndexSubset, const std::vector<int>& vBootstrap, unsigned int vNumInstancesHold)
{
	_ASSERTE(!vFeatureSet.empty() && !vResponseInstance.empty() && !vFeatureIndexSubset.empty() && !vBootstrap.empty());
	_ASSERTE(vNumInstancesHold != 0);

	int NumL = 0, NumR = 0, BestSplitFeatureIndex = 0;
	double SumL = 0.0, SumR = 0.0, SumResponse = 0.0, BestGap = 0.0;
	double MaxCurrentFeatureObjVal = -DBL_MAX, CurrentFeatureObjVal = 0.0;
	std::vector<std::pair<double, double>> FeatureResponseSet;
	for (auto FeatureIndex : vFeatureIndexSubset)
	{
		__generateFeatureResponsePairSet(vFeatureSet, vResponseInstance, FeatureIndex, vBootstrap, FeatureResponseSet, SumResponse);
		_ASSERTE(!FeatureResponseSet.empty() && SumResponse != 0);
		SumL = 0.0, SumR = SumResponse;
		NumL = 0, NumR = vNumInstancesHold;
		for (auto k = 0; k < FeatureResponseSet.size() - 1; ++k)
		{
			SumL += FeatureResponseSet[k].second;
			SumR -= FeatureResponseSet[k].second;
			CurrentFeatureObjVal = (std::pow(SumL, 2.0) / ++NumL) + (std::pow(SumR, 2.0) / --NumR) - std::pow(SumResponse, 2.0) / vNumInstancesHold;
			if (CurrentFeatureObjVal > MaxCurrentFeatureObjVal)
			{
				MaxCurrentFeatureObjVal = CurrentFeatureObjVal;
				BestGap = (FeatureResponseSet[k].first + FeatureResponseSet[k + 1].first) / 2.0;
				BestSplitFeatureIndex = FeatureIndex;
			}
		}
	}
	std::tuple<double, int> SplitFeatureAndGap = std::make_tuple(BestGap, BestSplitFeatureIndex);
	return SplitFeatureAndGap;
}

//****************************************************************************************************
//FUNCTION:
void CNodeSplitterOfInfoGain::__generateFeatureResponsePairSet(const std::vector<std::vector<double>>& vFeatureSet, const std::vector<double>& vResponseInstance, unsigned int vFeatureIndex, const std::vector<int>& vBootstrap, std::vector<std::pair<double, double>>& voFeatureResponseSet, double& voSumResponse)
{
	_ASSERTE(!vBootstrap.empty());
	voSumResponse = 0.0;
	voFeatureResponseSet.clear();
	for (auto Itr : vBootstrap)
	{
		voFeatureResponseSet.push_back(std::make_pair(vFeatureSet[Itr][vFeatureIndex], vResponseInstance[Itr]));
		voSumResponse += vResponseInstance[Itr];
	}
	std::sort(voFeatureResponseSet.begin(), voFeatureResponseSet.end(), [](const std::pair<double, double>& P1, const std::pair<double, double>& P2) {return P1.first < P2.first; });
}