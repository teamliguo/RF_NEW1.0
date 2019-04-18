#pragma once
#include "common/BaseProduct.h"

namespace hiveRegressionForest
{
	class IFeatureSelector : public hiveOO::CBaseProduct
	{
	public:
		IFeatureSelector();
		virtual ~IFeatureSelector() {}

		virtual void generateFeatureIndexSetV(unsigned int vFeatureNum, std::vector<int>& voFeatureIndexSubset, const std::vector<float>& vWeightSet = std::vector<float>()) = 0;
	
	protected:
		int m_NumCandidataFeature = 0;

	private:
		void __init();
	};
}