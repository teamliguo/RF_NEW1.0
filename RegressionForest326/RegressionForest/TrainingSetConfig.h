#pragma once
#include "common/HiveConfig.h"
#include "common/Singleton.h"
#include "RegressionForest_EXPORTS.h"

namespace hiveRegressionForest
{
	class REGRESSION_FOREST_EXPORTS CTrainingSetConfig : public hiveConfig::CHiveConfig, public hiveOO::CSingleton<CTrainingSetConfig>
	{
	public:
		~CTrainingSetConfig();

	private:
		CTrainingSetConfig();

		void __defineAcceptableAttributes();

		friend class hiveOO::CSingleton<CTrainingSetConfig>;
	};
}