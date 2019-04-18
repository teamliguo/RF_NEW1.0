#pragma once
#include "BaseTerminateCondition.h"

namespace hiveRegressionForest
{
	class CEarlyFittingCondition : public IBaseTerminateCondition
	{
	public:
		CEarlyFittingCondition();
		virtual ~CEarlyFittingCondition();

		virtual bool isMeetTerminateConditionV(const std::vector<std::vector<float>>& vFeatureSet, const std::vector<float>& vResponseSet, boost::any vExtra) override;

	private:
		float __calculateMeanSquaredError(const std::vector<std::vector<float>>& vFeatureSet, const std::vector<float>& vResponseSet);
	};
}