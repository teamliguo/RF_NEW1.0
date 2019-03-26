#include "FeatureWeightInvokingTreeMethod.h"
#include "RegressionForestCommon.h"
#include "common/ProductFactory.h"
#include "common/CommonInterface.h"

using namespace hiveRegressionForest;

hiveOO::CProductFactory<CFeatureWeightInvokingTreeMethod> theCreator(KEY_WORDS::INVOKE_FEATURES_METHOD);

CFeatureWeightInvokingTreeMethod::CFeatureWeightInvokingTreeMethod()
{
}


CFeatureWeightInvokingTreeMethod::~CFeatureWeightInvokingTreeMethod()
{
}

//***********************************************************************
//FUNCTION: 
void CFeatureWeightInvokingTreeMethod::__calculateFeatureWeightV(const std::vector<int>& vFeaturesInvokingNum, std::vector<float>& voFeatureWeightNormalized)
{
	//voFeatureWeightNormalized.resize(vFeaturesInvokingNum.size());
	int MinInvokingNum = INT_MAX;
	for (auto Iter : vFeaturesInvokingNum)
	{
		if (Iter < MinInvokingNum)
			MinInvokingNum = Iter;
	}
	if (!MinInvokingNum) MinInvokingNum = 1;

	for (unsigned int i = 0; i < vFeaturesInvokingNum.size(); ++i)
		voFeatureWeightNormalized[i] = __piecewiseRectifyFunction((vFeaturesInvokingNum[i] + 1.0f) / MinInvokingNum, 10.0f);	//magic number: This value is decided by actual experiment.
	//Fix-me: Is needed to normalize the output value ?

	// NOTE : print feature weight
	std::string Weight = "Feature Weight";
	for (unsigned int i = 0; i < vFeaturesInvokingNum.size(); ++i)
		Weight += "\nFeature Index " + std::to_string(i) + " : " + "[Invoke Num] " + std::to_string(vFeaturesInvokingNum[i]) + "[Weight] " + std::to_string(voFeatureWeightNormalized[i]);
	hiveCommon::hiveOutputEvent(Weight);
}

//***********************************************************************
//FUNCTION: 
float CFeatureWeightInvokingTreeMethod::__piecewiseRectifyFunction(float vValue, float vRectifyPosition)
{
	if (vValue < 1) return 0.0;
	if (vValue >= 1.0 && vValue < vRectifyPosition) return (log(vValue) + 1);
	else return (log10(vValue) + log(10));
}