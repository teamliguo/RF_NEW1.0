#pragma once
#include "BaseTerminateCondition.h"

namespace hiveRegressionForest
{
	class CPearsonPercentageCondition : public IBaseTerminateCondition
	{
	public:
		CPearsonPercentageCondition();
		virtual ~CPearsonPercentageCondition();

		virtual bool isMeetTerminateConditionV(const std::vector<std::vector<float>>& vFeatureSet, const std::vector<float>& vResponseSet, boost::any vExtra) override;
	};
}