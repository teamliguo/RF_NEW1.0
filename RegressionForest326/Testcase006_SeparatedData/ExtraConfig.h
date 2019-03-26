#pragma once
#include "common/HiveConfig.h"
#include "common/Singleton.h"

namespace hiveRegressionForestExtra
{
	class CExtraConfig : public hiveConfig::CHiveConfig, public hiveOO::CSingleton<CExtraConfig>
	{
	public:
		~CExtraConfig();

	private:
		CExtraConfig();

		void __defineAcceptableAttributes();

		friend class hiveOO::CSingleton<CExtraConfig>;
	};
}