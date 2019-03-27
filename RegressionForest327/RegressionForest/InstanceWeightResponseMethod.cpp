#include "InstanceWeightResponsetMethod.h"
#include <string>
#include "common/ProductFactory.h"
#include "math/PearsonCorrelation.h"
#include "TrainingSet.h"
#include "BaseInstanceWeightMethod.h"
#include "RegressionForestCommon.h"

using namespace hiveRegressionForest;

hiveOO::CProductFactory<CInstanceWeightResponsetMethod> TheCreator(KEY_WORDS::RESPONSE_METHOD);

CInstanceWeightResponsetMethod::CInstanceWeightResponsetMethod()
{
}

CInstanceWeightResponsetMethod::~CInstanceWeightResponsetMethod()
{
}

//****************************************************************************************************
//FUNCTION:
void CInstanceWeightResponsetMethod::generateInstancesWeightV(unsigned int vInstanceNum, std::vector<float>& voInstanceWeightSet)
{
	_ASSERTE(vInstanceNum > 0);
	const CTrainingSet* pTrainingSet = CTrainingSet::getInstance();
	float TempAvgResponse = 0.0f;

	voInstanceWeightSet.reserve(vInstanceNum);
	for (auto i = 0; i < vInstanceNum; ++i)
	{
		for (auto k = 0; k < pTrainingSet->getNumOfResponse(); ++k)
			TempAvgResponse += pTrainingSet->getResponseValueAt(i, k);

		voInstanceWeightSet.push_back(TempAvgResponse / pTrainingSet->getNumOfResponse());
	}
	
	float Max = *std::max_element(voInstanceWeightSet.begin(), voInstanceWeightSet.end());
	float Min = *std::min_element(voInstanceWeightSet.begin(), voInstanceWeightSet.end());
	_ASSERTE(Min != Max);
	std::for_each(voInstanceWeightSet.begin(), voInstanceWeightSet.end(), [&Min, &Max](float& vElem) {vElem = (vElem - Min) / (Max - Min) + 1; });
}