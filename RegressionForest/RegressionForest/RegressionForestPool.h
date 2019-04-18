#pragma once
#include <vector>
#include "common/Singleton.h"
#include "RegressionForest.h"

namespace hiveRegressionForest
{
	class CRegressionForestPool : public hiveOO::CSingleton<CRegressionForestPool>
	{
	public:
		~CRegressionForestPool();

		const CRegressionForest* fetchForest(const unsigned int& vForestId) const;
		unsigned int putForest(const CRegressionForest* vForest);

	private:
		CRegressionForestPool();

		std::vector<const hiveRegressionForest::CRegressionForest*> m_ForestSet;
		friend class hiveOO::CSingleton<CRegressionForestPool>;
	};
}
