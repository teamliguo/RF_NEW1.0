#pragma once
#include "common/Singleton.h"
#include <mutex>
#include <thread>
#include <boost/pool/object_pool.hpp>
#include "Node.h"
#include "RegressionForest_EXPORTS.h"
#include "TrainingSet.h"
#include "RegressionForestConfig.h"
#include "RegressionForestCommon.h"
#include "common/BaseProduct.h"

namespace hiveRegressionForest
{
	template <typename T>
	class CRegressionForestObjectPool : public hiveOO::CSingleton<CRegressionForestObjectPool<T>>
	{
	public:
		~CRegressionForestObjectPool() {}

		T* allocateNode(unsigned int vLevel)
		{
			std::lock_guard<std::mutex> lock(m_Mutex);

			return m_NodePool.construct(vLevel);
		}


	private:
		CRegressionForestObjectPool()
		{
			__initObjectPool();
		}

		void __initObjectPool()
		{
			int LeafNodeInstances = CRegressionForestConfig::getInstance()->getAttribute<int>(KEY_WORDS::MAX_LEAF_NODE_INSTANCE_SIZE);
			int NumTree = CRegressionForestConfig::getInstance()->getAttribute<int>(KEY_WORDS::NUMBER_OF_TREE);
			int NumSamples = CTrainingSet::getInstance()->getNumOfInstances();
			int MaxNumNodes = (2 * (NumSamples / LeafNodeInstances) - 1) * NumTree;

			// NOTES: set_next_size 当第一次申请内存时，一次性申请大小为 MaxNumNodes * sizeof(T)，由于存在 node_size < 5 的很多情况，因此调整为 MaxNumNodes * 2
			m_NodePool.set_next_size(MaxNumNodes * 2);
		}

		std::mutex m_Mutex;
		boost::object_pool<T> m_NodePool;

		friend class hiveOO::CSingleton<CRegressionForestObjectPool<T>>;
	};
}