#pragma once
#include "BaseInstanceWeightMethod.h"

namespace hiveRegressionForest
{
	class CInstanceWeightResponsetMethod : public IBaseInstanceWeight
	{
	public:
		CInstanceWeightResponsetMethod();
		~CInstanceWeightResponsetMethod();

		virtual void generateInstancesWeightV(unsigned int vInstanceNum, std::vector<float>& voInstanceWeightSet) override;
	};
}