#pragma once
#include <string>
#include <vector>
#include "Tree.h"
#include "common/BaseProduct.h"

namespace hiveRegressionForest
{
	class CMpCompute : public hiveOO::CBaseProduct
	{
	public:
		CMpCompute();
		~CMpCompute();

		float computeMPOfTwoFeatures(const CTree* vTree, const std::vector<float>& vLeafDate, const std::vector<float>& vTestData, float vPredictResponse = 0.f);
		float calMPOutOfFeatureAABB(const CTree* vTree, const CNode* vNode, const std::vector<float>& vFeature);

	private:
		void  __countIntervalNode(const CTree* vTree, const std::vector<std::pair<float, float>>& vMaxMinValue, std::vector<int>& voIntervalCount, std::vector<std::pair<float, float>>& voIntervalResponseRange);
		float __calMPValue(const CTree* vTree, const std::vector<std::pair<float, float>>& vMaxMinValue);

		template<typename Bound, typename Value>
		std::vector<Value> __obtainIntervalDataInRange(Bound vMin, Bound vMax, const std::vector<std::pair<Bound, Value>>& vData)
		{
			std::vector<Value> IntervalData;
			int MinIndex = 0, MaxIndex = 0;
			for (int k = 0; k < vData.size(); k++)
			{
				if (vData[k].first >= vMin)
				{
					MinIndex = k;
					MaxIndex = k;
					break;
				}
			}
			for (int k = MinIndex; k < vData.size(); k++)
			{
				if (vData[k].first <= vMax)
					MaxIndex = k;
				else break;
			}
			for (int k = MinIndex; k <= MaxIndex; k++)
				IntervalData.push_back(vData[k].second);

			return IntervalData;
		}
	};
}