#pragma once
#include "BaseBootstrapSelector.h"
#include <unordered_map>

namespace hiveRegressionForest
{
	class CAbsoluteBalancedBootstrapSelector : public IBaseBootstrapSelector
	{
	public:
		CAbsoluteBalancedBootstrapSelector();
		~CAbsoluteBalancedBootstrapSelector();

		virtual void generateBootstrapIndexSetV(unsigned int vInstanceNum, std::vector<int>& voBootstrapIndexSet) override;

	private:
		std::vector<std::vector<unsigned int>> m_GroupedResponseIndex;
	};
}