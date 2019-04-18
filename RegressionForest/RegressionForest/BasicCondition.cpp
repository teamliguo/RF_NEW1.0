#include "BasicCondition.h"
#include "RegressionForestConfig.h"
#include "RegressionForestCommon.h"
#include "common/ProductFactory.h"

using namespace hiveRegressionForest;

hiveOO::CProductFactory<CBasicCondition> theCreator(KEY_WORDS::BASIC_CONDITION);

CBasicCondition::CBasicCondition()
{
	m_MaxTreeDepth = CRegressionForestConfig::getInstance()->getAttribute<int>(KEY_WORDS::MAX_TREE_DEPTH);
	m_MaxLeftNodeSize = CRegressionForestConfig::getInstance()->getAttribute<int>(KEY_WORDS::MAX_LEAF_NODE_INSTANCE_SIZE);
}

CBasicCondition::~CBasicCondition()
{
}

//****************************************************************************************************
//FUNCTION:
bool CBasicCondition::isMeetTerminateConditionV(const std::vector<std::vector<float>>& vFeatureSet, const std::vector<float>& vResponseSet, boost::any vExtra)
{
	unsigned int NodeLevel = boost::any_cast<unsigned int>(vExtra);

	return (NodeLevel >= m_MaxTreeDepth || vFeatureSet.size() <= m_MaxLeftNodeSize);
}