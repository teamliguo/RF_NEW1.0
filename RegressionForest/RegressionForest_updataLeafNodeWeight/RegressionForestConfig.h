#pragma once
#include "common/HiveConfig.h"
#include "common/Singleton.h"
#include "RegressionForest_EXPORTS.h"

namespace hiveRegressionForest
{
	class REGRESSION_FOREST_EXPORTS CRegressionForestConfig : public hiveConfig::CHiveConfig, public hiveOO::CSingleton<CRegressionForestConfig>
	{
	public:
		~CRegressionForestConfig();

		static bool isConfigParsed();
		
	private:
		CRegressionForestConfig();

		void __defineAcceptableAttributes();

	friend class hiveOO::CSingleton<CRegressionForestConfig>;
	};
}