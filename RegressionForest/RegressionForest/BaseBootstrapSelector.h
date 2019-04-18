#pragma once
#include "common/BaseProduct.h"

#define DEFAULT_GROUP_NUM 10

namespace hiveRegressionForest
{
	class IBootstrapSelector : public hiveOO::CBaseProduct
	{
	public:
		IBootstrapSelector() {}
		virtual ~IBootstrapSelector() {}

		virtual void generateBootstrapIndexSetV(unsigned int vInstanceNum, std::vector<int>& voBootstrapIndexSet, const std::vector<float>& vWeightSet = std::vector<float>()) = 0;
	};
}