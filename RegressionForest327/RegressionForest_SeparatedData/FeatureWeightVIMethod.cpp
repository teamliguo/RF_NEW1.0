#include "FeatureWeightVIMethod.h"
#include "common/ProductFactory.h"
#include "RegressionForestCommon.h"
#include <fstream>
#include <sstream>

#define FEATURESIZE 16

using namespace hiveRegressionForest;

hiveOO::CProductFactory<CVIFeatureWeightMethod> theCreator(KEY_WORDS::VI_FEATURES_METHOD);

CVIFeatureWeightMethod::CVIFeatureWeightMethod()
{}

CVIFeatureWeightMethod::~CVIFeatureWeightMethod()
{}

void hiveRegressionForest::CVIFeatureWeightMethod::__calculateFeatureWeightV(const std::vector<std::vector<float>>& vInstanceSet, const std::vector<float>& vResponseSet, std::vector<std::pair<unsigned int, float>>& voFeatureWeightSet)
{
	_ASSERTE(!vInstanceSet.empty() && !vResponseSet.empty() && vInstanceSet.size() == vResponseSet.size());
	_ASSERTE(vInstanceSet[0].size() == FEATURESIZE);
	
	float VIArray[FEATURESIZE] = { 0.0f };

	std::ifstream VIFile;
	VIFile.open("C:\\Users\\lily\\Desktop\\RF_Experiments\\0731_addSSAO\\3SUB\\TrainingTest\\VI_EXP_2.csv", std::ios::in);

	if (VIFile.is_open())
	{
		std::string VIString;
		getline(VIFile, VIString);
		std::stringstream VIData(VIString);
		std::string Filed;
		for (unsigned int i = 0; i < 16; i++)
		{
			getline(VIData, Filed, ',');
			VIArray[i] = stof(Filed);
		}
		VIFile.close();
	}

	for (unsigned int i = 0; i < 16; i++)
	{
		voFeatureWeightSet.push_back(std::make_pair(i, VIArray[i]));
	}
}

