#pragma once
#include "BaseBootstrapSelector.h"
#include <unordered_map>

namespace hiveRegressionForest
{
	typedef std::pair<std::vector<unsigned int>, std::vector<unsigned int>> TRangePair;

	class CBalancedBootstrapSelector : public IBaseBootstrapSelector
	{
	public:
		CBalancedBootstrapSelector();
		~CBalancedBootstrapSelector();

		virtual void generateBootstrapIndexSetV(unsigned int vInstanceNum, std::vector<int>& voBootstrapIndexSet) override;
		
	private:
		unsigned int __divideResponseByRange(unsigned int vInstanceNum);

		unsigned int m_MaxGroupSize = 0;
		std::unordered_map<int, TRangePair> m_ResponseAndModIndexSet;
	};
}