#include "RegressionForestPool.h"
#include "common/HiveCommonMicro.h"

using namespace hiveRegressionForest;

CRegressionForestPool::CRegressionForestPool()
{

}

CRegressionForestPool::~CRegressionForestPool()
{
	for (int i = 0; i < m_ForestSet.size(); ++i)
	{
		if (m_ForestSet[i]) _SAFE_DELETE(m_ForestSet[i]);
	}
}

//****************************************************************************************************
//FUNCTION:
const CRegressionForest* CRegressionForestPool::fetchForest(const unsigned int& vForestId) const
{
	_ASSERT(vForestId < m_ForestSet.size());

	return m_ForestSet[vForestId];
}

//****************************************************************************************************
//FUNCTION:
unsigned int CRegressionForestPool::putForest(const CRegressionForest* vForest) 
{
	_ASSERT(vForest);

	m_ForestSet.push_back(vForest);
	return m_ForestSet.size() - 1;
}
