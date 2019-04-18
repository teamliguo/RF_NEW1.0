#pragma once
#include <vector>
#include <boost/any.hpp>
#include "common/BaseProduct.h"

namespace hiveRegressionForest
{
	class IBaseTerminateCondition : public hiveOO::CBaseProduct
	{
	public:
		IBaseTerminateCondition() {}
		virtual ~IBaseTerminateCondition() {}

		virtual bool isMeetTerminateConditionV(const std::vector<std::vector<float>>& vFeatureSet, const std::vector<float>& vResponseSet, boost::any vExtra) = 0;
		virtual bool isMeetTerminateConditionV(boost::any vExtra) { return true; }
	};
}