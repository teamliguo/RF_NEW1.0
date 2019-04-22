#pragma once
#include "BaseTerminateCondition.h"

namespace hiveRegressionForest
{
	class CBasicCondition : public IBaseTerminateCondition
	{
	public:
		CBasicCondition();
		virtual ~CBasicCondition();

		virtual bool isMeetTerminateConditionV(const std::vector<std::vector<float>>& vFeatureSet, const std::vector<float>& vResponseSet, boost::any vExtra) override;

	private:
		unsigned int m_MaxTreeDepth = 0;
		unsigned int m_MaxLeftNodeSize = 0;
	};
}