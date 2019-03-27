#pragma once
#include "common/BaseProduct.h"

namespace hiveRegressionForest
{
	class IBaseInstanceWeight : public hiveOO::CBaseProduct
	{
	public:
		IBaseInstanceWeight() {}
		virtual ~IBaseInstanceWeight() {}

		virtual void generateInstancesWeightV(unsigned int vInstanceNum, std::vector<float>& voInstanceWeightSet) = 0;
	};
}