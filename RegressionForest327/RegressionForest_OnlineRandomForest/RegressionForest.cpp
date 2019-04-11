#include "RegressionForest.h"
#include <omp.h>
#include <numeric>
#include <fstream>
#include <algorithm>
#include "common/CommonInterface.h"
#include "common/ConfigParser.h"
#include "common/HiveCommonMicro.h"
#include "common/ProductFactoryData.h"
#include "common/productfactory.h"
#include "Tree.h"
#include "TrainingSet.h"
#include "BaseInstanceWeightMethod.h"
#include "RegressionForestConfig.h"
#include "RegressionForestCommon.h"
#include "WeightedPathNodeMethod.h"

using namespace hiveRegressionForest;

CRegressionForest::CRegressionForest()
{
}

CRegressionForest::~CRegressionForest()
{
	__clearForest();
}

//****************************************************************************************************
//FUNCTION:
void CRegressionForest::buildForest(const std::string& vConfigFile)
{
	_ASSERTE(!vConfigFile.empty());

	const CRegressionForestConfig* pRegressionForestConfig = CRegressionForestConfig::getInstance();
	if (!CRegressionForestConfig::isConfigParsed())
	{
		bool IsConfigParsed = hiveConfig::hiveParseConfig(vConfigFile, hiveConfig::EConfigType::XML, CRegressionForestConfig::getInstance());
		_ASSERTE(IsConfigParsed);
	}

	clock_t Begin = clock();

	__initForest();

	IBootstrapSelector* pBootstrapSelector = nullptr;
	IFeatureSelector* pFeatureSelector = nullptr;
	INodeSpliter* pNodeSpliter = nullptr;
	IBaseTerminateCondition* pTerminateCondition = nullptr;
	IFeatureWeightGenerator* pFeatureWeightMethod = nullptr;
	__initForestParameters(pBootstrapSelector, pFeatureSelector, pNodeSpliter, pTerminateCondition, pFeatureWeightMethod);

	bool OmpParallelSig = pRegressionForestConfig->getAttribute<bool>(KEY_WORDS::OPENMP_PARALLEL_BUILD_TREE);

#pragma omp parallel for if (OmpParallelSig)
	for (auto i = 0; i < m_Trees.size(); ++i)
	{
		m_Trees[i]->buildTree(pBootstrapSelector, pFeatureSelector, pNodeSpliter, pTerminateCondition, pFeatureWeightMethod);
		std::cout << "Successfully built the " << i << " th tree." << std::endl;
	}
	clock_t End = clock();
		
	_SAFE_DELETE(pBootstrapSelector);
	_SAFE_DELETE(pFeatureSelector);
	_SAFE_DELETE(pNodeSpliter);
	_SAFE_DELETE(pTerminateCondition);
	_SAFE_DELETE(pFeatureWeightMethod);

	hiveCommon::hiveOutputEvent("Successfully built regression forests in " + std::to_string(End - Begin) + " milliseconds.");
	_LOG_("Successfully built regression forests in " + std::to_string(End - Begin) + " milliseconds.");
}

//****************************************************************************************************
//FUNCTION:
void CRegressionForest::outputForestInfo(const std::string& vOutputFileName) const
{
	int NumOfNodes = 0, NumOfLeafNodes = 0, NumOfUnfittedLeafNode = 0;
	std::vector<int> FeatureSplitTimes(CTrainingSet::getInstance()->getNumOfFeatures(), 0);
	std::vector<int> InstancesOOBTimes(CTrainingSet::getInstance()->getNumOfInstances(), 0);
	std::vector<int> OOBSize(m_Trees.size(), 0);
		
	for (auto i = 0; i < m_Trees.size(); ++i)
	{
		STreeInfo TreeInfo;
		m_Trees[i]->fetchTreeInfo(TreeInfo);

		NumOfNodes				+= TreeInfo.m_NumOfNodes;
		NumOfLeafNodes			+= TreeInfo.m_NumOfLeafNodes;
		NumOfUnfittedLeafNode	+= TreeInfo.m_NumOfUnfittedLeafNodes;

		for (int m = 0; m < TreeInfo.m_FeatureSplitTimes.size(); ++m)
			FeatureSplitTimes[m] += TreeInfo.m_FeatureSplitTimes[m];

		for (int k = 0; k < TreeInfo.m_InstanceOOBTimes.size(); ++k)
			InstancesOOBTimes[k] += TreeInfo.m_InstanceOOBTimes[k];

		OOBSize[i] = m_Trees[i]->getOOBIndexSet().size();
	}

	std::fstream ForestInfo(vOutputFileName, std::ios::out);
	if (ForestInfo.is_open())
	{
		ForestInfo << "Number of Nodes : " << "," << NumOfNodes << "\n";
		ForestInfo << "Number of Leaf Nodes : " << "," << NumOfLeafNodes << "\n";
		ForestInfo << "Number of Unfitted Leaf Nodes : " << "," << NumOfUnfittedLeafNode << "\n";

		ForestInfo << "Feature Split Times:\n";
		for (int i = 0; i < FeatureSplitTimes.size(); ++i)
		{
			ForestInfo << FeatureSplitTimes[i] << ",";
			if (i % 100 == 99  && i != 0) ForestInfo << "\n";			
		}
		ForestInfo << "\n";

		ForestInfo << "OOB Size in Each Tree : \n";
		for (auto Itr : OOBSize) ForestInfo << Itr << ",";
		ForestInfo << "\n";

		ForestInfo << "Instance OOB Times : \n";
		for (int k = 0; k < InstancesOOBTimes.size(); ++k)
		{
			ForestInfo << InstancesOOBTimes[k] << ",";
			if (k % 100 == 99 && k != 0) ForestInfo << "\n";			
		}
		ForestInfo << "\n";
	}
	else
	{
		std::cout << "Forest Information File failed to open..." << std::endl;
	}
	ForestInfo.close();
}

//****************************************************************************************************
//FUNCTION:
float CRegressionForest::predict(const std::vector<float>& vFeatures, unsigned int vNumOfUsingTrees, float& voKnnPredictSet,    float& voVarWiPerdict, bool vIsWeightedPrediction) const
{
	//return __predictByKMeansForest(vFeatures, vNumOfUsingTrees, voIFMPPerdict, false);//树叶子集合，做KMean聚类预测
	// return __predictByMpWeightTree(vFeatures, vNumOfUsingTrees, voIFMPPerdict, false);//MP给树加权
	return	__predictByVarWeightTree(vFeatures, vNumOfUsingTrees, voVarWiPerdict, false);//Var给树加权
	//return __predictCertainResponse(vFeatures, vNumOfUsingTrees, vIsWeightedPrediction, voLPPredictSet, voMPPredictSet,  voLLPerdict,  voLLPSet, voMPDissimilarity,  voIFMPPerdict);
}
//****************************************************************************************************
//FUNCTION:
float CRegressionForest::predict(const std::vector<float>& vFeatures, unsigned int vNumOfUsingTrees, float& voLPPredictSet, float& voMPPredictSet, float& voLLPerdict, float& voLLPSet, float& voMPDissimilarity, float& voIFMPPerdict, bool vIsWeightedPrediction) const
{ 
	 //return __predictByKMeansForest(vFeatures, vNumOfUsingTrees, voIFMPPerdict, false);//树叶子集合，做KMean聚类预测
   // return __predictByMpWeightTree(vFeatures, vNumOfUsingTrees, voIFMPPerdict, false);//MP给树加权
	 return	__predictByVarWeightTree(vFeatures, vNumOfUsingTrees, voIFMPPerdict, false);//LP给树加权
    //return __predictCertainResponse(vFeatures, vNumOfUsingTrees, vIsWeightedPrediction, voLPPredictSet, voMPPredictSet,  voLLPerdict,  voLLPSet, voMPDissimilarity,  voIFMPPerdict);
}

//****************************************************************************************************
//FUNCTION:
void CRegressionForest::predict(const std::vector<float>& vFeatures, unsigned int vNumOfUsingTrees, bool vIsWeightedPrediction, unsigned int vNumResponse, std::vector<float>& voPredictValue) const
{
	_ASSERTE(!vFeatures.empty());

	voPredictValue.resize(vNumResponse); 
	/*for (int i = 0; i < vNumResponse; ++i)
		voPredictValue[i] = __predictCertainResponse(vFeatures, vNumOfUsingTrees, vIsWeightedPrediction, 0, 0, i);*/
}

//****************************************************************************************************
//FUNCTION:
float CRegressionForest::__predictCertainResponse(const std::vector<float>& vFeatures, unsigned int vNumOfUsingTrees, bool vIsWeightedPrediction, float& voLPPredictSet, float& voMPPredictSet,float& voLLPerdict,float& voLLPSet, float& voMPDissimilarity, float& voIFMPPerdict, unsigned int vResponseIndex) const
{
	_ASSERTE(!vFeatures.empty() && vNumOfUsingTrees > 0);

	float PredictValue = 0.0f;
	std::vector<float> PredictValueOfTree(vNumOfUsingTrees, 0.0f);
	std::vector<float> PredictValueOfBotherTree(vNumOfUsingTrees, 0.0f);
	std::vector<float> PredictMPValueOfTree(vNumOfUsingTrees, 0.0f);
	std::vector<float> PredictMPValueOfBotherTree(vNumOfUsingTrees, 0.0f);
	std::vector<float> PredictMidMPValueOfTree(vNumOfUsingTrees, 0.0f);
	std::vector<float> PredictMidMPOfTree(vNumOfUsingTrees, 0.0f);
	std::vector<float> PredictMidMPValueOfBotherTree(vNumOfUsingTrees, 0.0f);
	std::vector<float> PredictMidMPOfBotherTree(vNumOfUsingTrees, 0.0f);
	std::vector<float> PredictKNNBotherTree(vNumOfUsingTrees, 0.0f);
	std::vector<float> PredictLPWeightTree(vNumOfUsingTrees, 0.0f);
	float PredictKmeansForest = 0.0f;//用kmeans将所有树的叶子样本聚类，计算离测试点最近的簇的均值
	std::vector<std::pair<float, float>>  PredictMPWeightTree;
	std::vector<float> NodeWeight(vNumOfUsingTrees, 0.0f);
	std::vector<int> TotalDataIndex;
	std::vector<int> TotalDataIndexByOrder;//zy313
	std::vector<std::vector<int>> NodeDataIndex(vNumOfUsingTrees, std::vector<int>());

	CTrainingSet* pTrainingSet = CTrainingSet::getInstance();
	CWeightedPathNodeMethod* pWeightedPathNode = CWeightedPathNodeMethod::getInstance();
	static std::vector<const CNode*> LeafNodeSet;
	static std::vector<const CNode*> LeafBotherNodeSet;
	LeafNodeSet.resize(this->getNumOfTrees());
	LeafBotherNodeSet.resize(this->getNumOfTrees());
	//modify by zy 20190218
	int callMethod = CRegressionForestConfig::getInstance()->getAttribute<int>(KEY_WORDS::CALL_METHOD);  
	int InstanceGetMethodInter = CRegressionForestConfig::getInstance()->getAttribute<int>(KEY_WORDS::INSTANCE_GET_METHOD_INTER);
	int InstanceNumberInter = CRegressionForestConfig::getInstance()->getAttribute<int>(KEY_WORDS::INSTANCE_NUMBER_INTER);
	float InstanceNumberRatioInter = CRegressionForestConfig::getInstance()->getAttribute<float>(KEY_WORDS::INSTANCE_NUMBER_RATIO_INTER);
	/*std::ofstream PredictNodeInfo;
	PredictNodeInfo.open("PredictNodeInfo.csv", std::ios::app);*/

 
	std::vector<std::vector<float>> TotalNodeDataFeature;
	std::vector<float> TotalNodeDataResponse;

	std::vector<std::vector<float>> treeTotalNodeDataFeature;
	std::vector<float> treeTotalNodeDataResponse;
	std::vector<std::vector<float>>  treeNodeFeatureMid;//每棵树的每个叶子的中心点 
 
	/*PredictNodeInfo << "Tree"<< "," << "叶子编号" << "," << "叶子预测值"<< "," << "叶子样本个数"
		<< "," <<"叶子兄弟编号" << "," << "兄弟预测值"<< "," << "兄弟样本个数"<< ","<<"叶子分裂维度" << "," << "叶子与兄弟每维度的范围：" << ",";*/
//#pragma omp parallel for-----------------计算中间节点的MP和LP--------------------------------
	for (int i = 0; i < vNumOfUsingTrees; ++i)
	{
		TotalNodeDataFeature.clear();
		TotalNodeDataResponse.clear();
		std::vector<std::vector<float>> vNodeDataFeature;
		std::vector<float> vNodeDataResponse;
		std::vector<std::vector<float>> vchildNodeDataFeature;
		std::vector<float> vchildNodeDataResponse;
		std::vector<float>  NodeFeatureMid;//每个叶子的中心点 
		//std::cout << "tree " << i << ": " << std::endl;
		if (vResponseIndex == 0)
		{
			LeafNodeSet[i] = m_Trees[i]->locateLeafNode(vFeatures);
			//叶子均值预测
			PredictValueOfTree[i] = m_Trees[i]->predict(*LeafNodeSet[i], vFeatures, NodeWeight[i], vResponseIndex);
			std::vector<int> nodeDataIndex;
			std::vector<std::pair<float, float>> nodeResponseMPSet;
			std::vector<std::pair<float, float>> nodeMidResponseMPSet;
			//遍历其节点信息
			nodeDataIndex = LeafNodeSet[i]->getNodeDataIndexV();//pWeightedPathNode->calNodeDataIndex(LeafBotherNodeSet[i]);
			std::vector<std::vector<float>> NodeDataFeature(nodeDataIndex.size(), std::vector<float>());
			std::vector<float> NodeDataResponse(nodeDataIndex.size(), 0.f);
			float sumy = 0.0f;

			for (int k = 0; k < nodeDataIndex.size(); ++k)
			{
				NodeDataFeature[k] = pTrainingSet->getFeatureInstanceAt(nodeDataIndex[k]);
				NodeDataResponse[k] = pTrainingSet->getResponseValueAt(nodeDataIndex[k]);
				sumy += NodeDataResponse[k];
				TotalNodeDataFeature.push_back(NodeDataFeature[k]);
				TotalNodeDataResponse.push_back(NodeDataResponse[k]);

				treeTotalNodeDataFeature.push_back(NodeDataFeature[k]);
				treeTotalNodeDataResponse.push_back(NodeDataResponse[k]);
			}

			for (int j = 0;j < NodeDataFeature[0].size();j++)//列
			{
				float featureMid = 0.0f;
				for (int i = 0; i < NodeDataFeature.size(); i++)//行
				{
					featureMid += NodeDataFeature[i][j];
				}
				NodeFeatureMid.push_back({ featureMid / NodeDataFeature.size() });//每列/维度的均值
			}

			treeNodeFeatureMid.push_back({ NodeFeatureMid });//每棵树的叶子中心点

			vNodeDataFeature = NodeDataFeature;
			vNodeDataResponse = NodeDataResponse;
			//叶子的各样本mp
			nodeResponseMPSet = pTrainingSet->calReCombineDataMP(NodeDataFeature, NodeDataResponse, vFeatures, 0);
			////叶子的样本中心点mp
			//nodeMidResponseMPSet = pTrainingSet->calmidMP(NodeDataFeature, NodeDataResponse, vFeatures, 0);
			float sumOfMP = 0.0F;
			float pbyMP = 0.0F;
			for (int k = 0; k < nodeResponseMPSet.size(); ++k)
			{
				sumOfMP += nodeResponseMPSet[k].second;
			}
			for (int k = 0; k < nodeResponseMPSet.size(); ++k)
			{
				nodeResponseMPSet[k].second = nodeResponseMPSet[k].second / sumOfMP;
				pbyMP += nodeResponseMPSet[k].first*nodeResponseMPSet[k].second;
			}
			//叶子各样本mp加权计算预测值
			PredictMPValueOfTree[i]= pbyMP;
		/*	PredictMidMPValueOfTree[i] = nodeMidResponseMPSet[0].first;
			PredictMidMPOfTree[i]= nodeMidResponseMPSet[0].second;*/
			if (&(&LeafNodeSet[i]->getMother())->getRightChild() == LeafNodeSet[i])
			{

				LeafBotherNodeSet[i]=&(&LeafNodeSet[i]->getMother())->getLeftChild();
			}
			else
			{
				LeafBotherNodeSet[i] = &(&LeafNodeSet[i]->getMother())->getRightChild();
			}
		}
			
		
		std::vector<int> childDataIndex;
		std::vector<std::pair<float, float>> BotherResponseMPSet;
		std::vector<std::pair<float, float>> BotherMidResponseMPSet;
		if (LeafBotherNodeSet[i]->isLeafNode() == true)
		{
			//叶子兄弟均值预测
			PredictValueOfBotherTree[i] = m_Trees[i]->predict(*LeafBotherNodeSet[i], vFeatures, NodeWeight[i], vResponseIndex);
			//遍历其子节点信息
			childDataIndex = LeafBotherNodeSet[i]->getNodeDataIndexV();//pWeightedPathNode->calNodeDataIndex(LeafBotherNodeSet[i]);
			std::vector<std::vector<float>> childNodeDataFeature(childDataIndex.size(), std::vector<float>());
			std::vector<float> childNodeDataResponse(childDataIndex.size(), 0.f);
			float sumy = 0.0f;

			for (int k = 0; k < childDataIndex.size(); ++k)
			{
				
				childNodeDataFeature[k] = pTrainingSet->getFeatureInstanceAt(childDataIndex[k]);
				childNodeDataResponse[k] = pTrainingSet->getResponseValueAt(childDataIndex[k]);
				sumy += childNodeDataResponse[k];
				TotalNodeDataFeature.push_back(childNodeDataFeature[k]);
				TotalNodeDataResponse.push_back(childNodeDataResponse[k]);

			}
			vchildNodeDataFeature = childNodeDataFeature;
			vchildNodeDataResponse = childNodeDataResponse;
			//叶子兄弟mp计算
			BotherResponseMPSet = pTrainingSet->calReCombineDataMP(childNodeDataFeature, childNodeDataResponse, vFeatures, 0);
			////叶子兄弟的样本中心点mp
			//BotherMidResponseMPSet = pTrainingSet->calmidMP(childNodeDataFeature, childNodeDataResponse, vFeatures, 0);
			float sumOfMP = 0.0F; 
			float pbyMP = 0.0F;
			for (int k = 0; k < BotherResponseMPSet.size(); ++k)
			{
				sumOfMP += BotherResponseMPSet[k].second;
			}
			for (int k = 0; k < BotherResponseMPSet.size(); ++k)
			{
				BotherResponseMPSet[k].second = BotherResponseMPSet[k].second/sumOfMP;
				pbyMP += BotherResponseMPSet[k].first*BotherResponseMPSet[k].second;
			}
			//叶子兄弟mp加权计算预测值
			PredictMPValueOfBotherTree[i]=pbyMP;
			////叶子兄弟mp中点预测
			//PredictMidMPValueOfBotherTree[i] = BotherMidResponseMPSet[0].first;
			//PredictMidMPOfBotherTree[i] = BotherMidResponseMPSet[0].second;
		}
		else
		{
			//遍历其子节点信息
			childDataIndex = pWeightedPathNode->calNodeDataIndex(LeafBotherNodeSet[i]);
			std::vector<std::vector<float>> childNodeDataFeature(childDataIndex.size(), std::vector<float>());
			std::vector<float> childNodeDataResponse(childDataIndex.size(), 0.f);
			float sumy = 0.0f;

			for (int k = 0; k < childDataIndex.size(); ++k)
			{
				childNodeDataFeature[k] = pTrainingSet->getFeatureInstanceAt(childDataIndex[k]);
				childNodeDataResponse[k] = pTrainingSet->getResponseValueAt(childDataIndex[k]);
				sumy += childNodeDataResponse[k];
				TotalNodeDataFeature.push_back(childNodeDataFeature[k]);
				TotalNodeDataResponse.push_back(childNodeDataResponse[k]);
			}
			vchildNodeDataFeature = childNodeDataFeature;
			vchildNodeDataResponse = childNodeDataResponse;
			//用该兄弟节点的所有子节点的样本 求均值
			PredictValueOfBotherTree[i] = sumy / childDataIndex.size();
			BotherResponseMPSet = pTrainingSet->calReCombineDataMP(childNodeDataFeature, childNodeDataResponse, vFeatures, 0);
			
			////叶子兄弟的样本中心点mp
			//BotherMidResponseMPSet = pTrainingSet->calmidMP(childNodeDataFeature, childNodeDataResponse, vFeatures, 0);
			float sumOfMP = 0.0F;
			float pbyMP = 0.0F;
			for (int k = 0; k < BotherResponseMPSet.size(); ++k)
			{
				sumOfMP += BotherResponseMPSet[k].second;
			}
			for (int k = 0; k < BotherResponseMPSet.size(); ++k)
			{
				BotherResponseMPSet[k].second = BotherResponseMPSet[k].second / sumOfMP;
				pbyMP += BotherResponseMPSet[k].first*BotherResponseMPSet[k].second;
			}
			PredictMPValueOfBotherTree[i] = pbyMP;
			////叶子兄弟mp中点预测
			//PredictMidMPValueOfBotherTree[i] = BotherMidResponseMPSet[0].first;
			//PredictMidMPOfBotherTree[i] = BotherMidResponseMPSet[0].second;
		}
		//利用相似度计算权重，联合叶子与叶子兄弟计算 预测值
		//std::vector<std::pair<float, float>> leafAndBotherMPSet;//first 为预测Y值，second 为MP值
		//leafAndBotherMPSet = pTrainingSet->calLeafAndBotherMP(vNodeDataFeature, vNodeDataResponse, vchildNodeDataFeature, vchildNodeDataResponse, 
		//	                                                  TotalNodeDataFeature, TotalNodeDataResponse, vFeatures, 0);
		//float sumOfMP =  pow(leafAndBotherMPSet[0].second,-2) +   pow(leafAndBotherMPSet[1].second,-2);
		//voIFMPPerdict += leafAndBotherMPSet[0].first*(pow(leafAndBotherMPSet[0].second,-2)/sumOfMP)+ leafAndBotherMPSet[1].first*(pow(leafAndBotherMPSet[1].second,-2)/ sumOfMP);
		// 
		 
		//将叶子和其兄弟节点mp相似度高的样本加入到叶子中，重新计算均值
		/*PredictKNNBotherTree[i] = pTrainingSet->calLeafAndBotherKNNResponse(vNodeDataFeature, vNodeDataResponse, vchildNodeDataFeature, vchildNodeDataResponse,
			                                           TotalNodeDataFeature, TotalNodeDataResponse, vFeatures, 0);*/
		//根据相似度计算每棵树的叶子的权重----------------------------- 
		PredictLPWeightTree[i]= pTrainingSet->calLeafWeightForTreeByLp(vNodeDataFeature, vFeatures);
	
		/*PredictNodeInfo << i << "," << (&LeafNodeSet[i]->getMother())->getBestSplitFeatureIndex() << "," << (&LeafNodeSet[i]->getMother())->getBestGap() << ","
			<< LeafNodeSet[i] << "," << LeafNodeSet[i]->isLeafNode() << "," << LeafNodeSet[i]->getNodeSize() << "," << LeafNodeSet[i]->getLevel() << ","
			<< PredictValueOfTree[i] << "," << PredictValueOfBotherTree[i] << "," << PredictMPValueOfTree[i] << "," << PredictMPValueOfBotherTree[i] << ","
			<< leafAndBotherMPSet[0].first << "," << leafAndBotherMPSet[0].second << ","
			<< leafAndBotherMPSet[1].first << "," << leafAndBotherMPSet[1].second << ","
			<< PredictKNNBotherTree[i]<<","<< PredictLPWeightTree[i]<<","
			<< LeafBotherNodeSet[i] << "," << LeafNodeSet[i]->isLeafNode() << "," << LeafBotherNodeSet[i]->getNodeSize() << ","<< LeafBotherNodeSet[i]->getLevel() << ",";
		   */
		/*std::pair<std::vector<float>, std::vector<float>> LeafNodeFeatRange = LeafNodeSet[i]->getFeatureRange();
		std::pair<std::vector<float>, std::vector<float>> LeafBotherNodeFeatRange = LeafBotherNodeSet[i]->getFeatureRange();
		int splitFeatureIndex = (&LeafNodeSet[i]->getMother())->getBestSplitFeatureIndex();*/
		/*for (int k = 0; k < vFeatures.size(); ++k)
		{*/
			//PredictNodeInfo << vFeatures[splitFeatureIndex] << ","<< LeafNodeFeatRange.first[splitFeatureIndex] << "," << LeafNodeFeatRange.second[splitFeatureIndex] << "," << LeafBotherNodeFeatRange.first[splitFeatureIndex] << "," << LeafBotherNodeFeatRange.second[splitFeatureIndex];
		/*}*/
		//PredictNodeInfo << std::endl;

		/*voMPPredictSet += pWeightedPathNode->predictWithMinMPOnWholeDimension(m_Trees[i], vFeatures, NodeDataIndex[i]);*/
		//modifed by zy 2.18 
		std::pair<float, float> voMPAndLPSet = pWeightedPathNode->predictWithMinMPAndLPOnWholeDimension(m_Trees[i], vFeatures, NodeDataIndex[i], InstanceGetMethodInter, InstanceNumberInter, InstanceNumberRatioInter);
		
		voMPPredictSet += voMPAndLPSet.first;//根据MP取最相似点
		voLLPerdict += voMPAndLPSet.second;//根据欧式距离取最相似点
		TotalDataIndex.insert(TotalDataIndex.end(), NodeDataIndex[i].begin(), NodeDataIndex[i].end());
	}
	


	//遍历森林，计算每棵树落入叶子的权重（即树的权重MP）,将所有叶子的样本合成的总样本空间作为n，来计算测试点到每个叶子中心点的mp，通过mp计算权重
	 PredictMPWeightTree = pTrainingSet->calLeafWeightForTreeByMp(treeNodeFeatureMid, PredictValueOfTree, treeTotalNodeDataFeature, treeTotalNodeDataResponse, vFeatures);
	////输出数据进行分析
	//std::ofstream NodeInfo;
	//NodeInfo.open("NodeInfo.csv", std::ios::app);
 //for (int i = 0;i < treeTotalNodeDataFeature.size();i++)
 //{
	//for (int j = 0;j < vFeatures.size();j++)
	//{
	//	 NodeInfo << vFeatures[j] << ",";
	//}
	//NodeInfo << i << ","; 
	//for (int k = 0;k < treeTotalNodeDataFeature[0].size();k++)
	//{
	//	NodeInfo << treeTotalNodeDataFeature[i][k] << ",";

	//}
	//NodeInfo << treeTotalNodeDataResponse[i]<<","<<std::endl;
	// 
	//}
	
	PredictKmeansForest = pTrainingSet->calKMeansForest(treeNodeFeatureMid, PredictValueOfTree, treeTotalNodeDataFeature, treeTotalNodeDataResponse, vFeatures);
	std::vector<float> distanceOfTree(PredictMPWeightTree.size());
	//float SumWeight = std::accumulate(PredictLPWeightTree.begin(), PredictLPWeightTree.end(), 0.f);
	float SumPredict = std::accumulate(PredictValueOfTree.begin(), PredictValueOfTree.end(), 0.f);
	/*voIFMPPerdict = std::accumulate(PredictKNNBotherTree.begin(), PredictKNNBotherTree.end(), 0.f);*/
	std::vector<std::vector<float>> LocatedNodeDataFeature(TotalDataIndex.size(), std::vector<float>());
	std::vector<float> LocatedNodeDataResponse(TotalDataIndex.size(), 0.f);
	for (int k = 0; k < PredictMPWeightTree.size(); ++k)
	{
		distanceOfTree[k]= PredictMPWeightTree[k].second;

	}

	/*for (int k = 0; k < PredictLPWeightTree.size(); ++k)
	{
		distanceOfTree[k] = PredictLPWeightTree[k];

	}*/
	std::vector<float> wiOfTree = pTrainingSet->antiDistanceWeight(distanceOfTree,-2);
	for (int k = 0; k < PredictMPWeightTree.size(); ++k)
	{
		voIFMPPerdict += PredictMPWeightTree[k].first * wiOfTree[k];

	}
	//for (int k = 0; k < PredictLPWeightTree.size(); ++k)
	//{
	//	voIFMPPerdict += PredictValueOfTree[k] * wiOfTree[k];

	//}
     voIFMPPerdict = PredictKmeansForest;//测试-----------------
	//voIFMPPerdict = PredictMPWeightTree[0].first;
	if (callMethod == 1)
	{
       /*#pragma omp parallel for*/
		for (int i = 0; i < TotalDataIndex.size(); ++i)
		{
			LocatedNodeDataFeature[i] = pTrainingSet->getFeatureInstanceAt(TotalDataIndex[i]);
			LocatedNodeDataResponse[i] = pTrainingSet->getResponseValueAt(TotalDataIndex[i]);
		}
		std::vector<std::pair<float, float>> ResponseMPSet = pTrainingSet->calReCombineDataMP(LocatedNodeDataFeature, LocatedNodeDataResponse, vFeatures, SumPredict / vNumOfUsingTrees);
		////modify by zy 2019218--计算欧式距离 -----------计算KNN-LP
		//std::vector<std::pair<float, float>> ResponseLPSet = pTrainingSet->calReCombineDataLP(LocatedNodeDataFeature, LocatedNodeDataResponse, vFeatures, SumPredict / vNumOfUsingTrees);


		//modify by zy 20190214
		int InstanceGetMethod = CRegressionForestConfig::getInstance()->getAttribute<int>(KEY_WORDS::INSTANCE_GET_METHOD);
		int InstanceNumber = CRegressionForestConfig::getInstance()->getAttribute<int>(KEY_WORDS::INSTANCE_NUMBER);
		float InstanceNumberRatio = CRegressionForestConfig::getInstance()->getAttribute<float>(KEY_WORDS::INSTANCE_NUMBER_RATIO);
		
		if (InstanceGetMethod == 0)
		{
			if (InstanceNumber <= 0) InstanceNumber = 1;
			if (InstanceNumber > ResponseMPSet.size()) InstanceNumber = ResponseMPSet.size();
		}
		else
		{
			int allResponseNumber = ResponseMPSet.size();
			InstanceNumber = (int)(allResponseNumber *InstanceNumberRatio);
		}
		
		
		
		//modify by zy 20190214
		int InstanceByWeight = CRegressionForestConfig::getInstance()->getAttribute<int>(KEY_WORDS::INSTANCE_BY_WEIGHT);
		if (InstanceByWeight == 1)//根据MP重复个数和排序计算权重均值
		{
			float countWeight = CRegressionForestConfig::getInstance()->getAttribute<float>(KEY_WORDS::INSTANCE_COUNT_WEIGHT);
			float orderWeight = 1 - countWeight;
			//统计ResponseMPSet中重复的MP值，重复个数和按MP的排序
			std::vector<float> ResponseValue;
			std::vector<std::pair<int, int>> MPCountAndOrder;
			int j = 0;

			for (int i = 0; i < InstanceNumber; ++i)
			{

				if (i == 0 && j == 0)
				{
					ResponseValue.push_back(ResponseMPSet[i].first);
					MPCountAndOrder.push_back({ 1, 1 });

				}
				else
				{
					if (ResponseMPSet[i].first == ResponseValue[j])
					{

						MPCountAndOrder[j].first = MPCountAndOrder[j].first + 1;
						MPCountAndOrder[j].second = MPCountAndOrder[j].second;

					}
					else
					{
						ResponseValue.push_back(ResponseMPSet[i].first);
						MPCountAndOrder.push_back({ 1, MPCountAndOrder[j].second + 1 });
						j++;
					}
				}

			}
			//加权计算--------------------------
			//---倒序----
			int maxOrder = MPCountAndOrder[MPCountAndOrder.size() - 1].second;
			for (int k = 0; k<MPCountAndOrder.size(); k++)
			{
				MPCountAndOrder[k].second = maxOrder--;

			}
			float sumOfWeight = 0.0f;
			std::vector<float> MPWeight;

			for (int k = 0; k < MPCountAndOrder.size(); k++)
			{
				MPWeight.push_back(MPCountAndOrder[k].first*countWeight + MPCountAndOrder[k].second*orderWeight);
				sumOfWeight += MPCountAndOrder[k].first*countWeight + MPCountAndOrder[k].second*orderWeight;

			}

			for (int k = 0; k < MPCountAndOrder.size(); k++)
			{
				MPWeight[k] /= sumOfWeight;
				voLPPredictSet += MPWeight[k] * ResponseValue[k];
			}
			
			 

		}
		else //直接计算均值
		{
			for (int i = 0; i < InstanceNumber; ++i)
			{
				voLPPredictSet += ResponseMPSet[i].first;
			}
			voLPPredictSet = voLPPredictSet / InstanceNumber;

			
		}

		////lp按均值计算--KNN
		//for (int i = 0; i < InstanceNumber; ++i)
		//{
		//	voLLPSet += ResponseLPSet[i].first;
		//}
		//voLLPSet = voLLPSet / InstanceNumber;

		/*IFMP按均值计算
		add IForestWayOfMP--modify by zy 20190313*/

		/*TotalDataIndexByOrder = TotalDataIndex;
		std::vector<std::pair<int, float>>  predictIndexByIFMP = pTrainingSet->calIfMp(vNumOfUsingTrees, TotalDataIndexByOrder, NodeDataIndex);
		std::vector<std::pair<float, float>> predictValueByIFMP;
		for (int i = 0; i < predictIndexByIFMP.size(); ++i)
		{

			float ResponseByIFMP = pTrainingSet->getResponseValueAt(predictIndexByIFMP[i].first);
			predictValueByIFMP.push_back({ ResponseByIFMP, predictIndexByIFMP[i].second });
		}
		int InstanceIFNumber = CRegressionForestConfig::getInstance()->getAttribute<float>(KEY_WORDS::IForestMP_INSTANCE_NUMBER);
		float IForestMPNumRatio = CRegressionForestConfig::getInstance()->getAttribute<float>(KEY_WORDS::IForestMP_NUMBER_RATIO);
		int IForestMPGetMethod = CRegressionForestConfig::getInstance()->getAttribute<float>(KEY_WORDS::IForestMP_GET_METHOD);

		int number = predictValueByIFMP.size();
		if (IForestMPGetMethod == 0)
		{
			if (InstanceIFNumber <= 0) InstanceIFNumber = 1;
			if (InstanceIFNumber >number) InstanceIFNumber = number;
		}
		else
		{
			 
			InstanceIFNumber = (int)(number *IForestMPNumRatio);
		}

		for (int i = 0; i <InstanceIFNumber; ++i)
		{
			voIFMPPerdict += predictValueByIFMP[i].first;
		}
		voIFMPPerdict = voIFMPPerdict / InstanceIFNumber;*/
	}

	_ASSERTE(PredictValueOfTree.size() == NodeWeight.size());

	if (vIsWeightedPrediction)
	{
		for (int k = 0; k < vNumOfUsingTrees; ++k)
			PredictValue += PredictValueOfTree[k] * NodeWeight[k];
		float SumWeight = std::accumulate(NodeWeight.begin(), NodeWeight.end(), 0.0f);
		_ASSERTE(SumWeight > 0);

		return PredictValue / SumWeight;
	}
	else
	{
		/*voMPPredictSet = voMPPredictSet / vNumOfUsingTrees;
		voLLPerdict = voLLPerdict / vNumOfUsingTrees;*/
		//voIFMPPerdict = voIFMPPerdict / vNumOfUsingTrees;
		 
		return SumPredict / vNumOfUsingTrees;
	}

	
}

//根据对叶子节点做KMeans聚类再做加权预测---------------------------------add by zy 2019.3.29
float CRegressionForest::__predictByKMeansForest(const std::vector<float>& vFeatures, unsigned int vNumOfUsingTrees, float& PredictKmeansForest, unsigned int vResponseIndex) const
{
	_ASSERTE(!vFeatures.empty() && vNumOfUsingTrees > 0);
	float PredictValue = 0.0f;
	std::vector<float> PredictValueOfTree(vNumOfUsingTrees, 0.0f); 
	std::vector<float> NodeWeight(vNumOfUsingTrees, 0.0f);
	std::vector<std::vector<int>> NodeDataIndex(vNumOfUsingTrees, std::vector<int>());
	CTrainingSet* pTrainingSet = CTrainingSet::getInstance();
	CWeightedPathNodeMethod* pWeightedPathNode = CWeightedPathNodeMethod::getInstance();
	static std::vector<const CNode*> LeafNodeSet;
	LeafNodeSet.resize(this->getNumOfTrees());
	std::vector<std::vector<float>> treeTotalNodeDataFeature;
	std::vector<int> treeTotalNodeDataIndex; 
	std::vector<float> treeTotalNodeDataResponse;
	std::vector<std::vector<float>>  treeNodeFeatureMid;//每棵树的每个叶子的中心点 
	float maxResponseVarRatio = 1.0f;
	float maxResponseVarOfTree = 0.0f;
	std::vector<float> treeNodeVar(vNumOfUsingTrees, 0.0f);
	std::ofstream VarInfo;
	VarInfo.open("VarInfo.csv", std::ios::app);
	for (int i = 0; i < vNumOfUsingTrees; ++i)
	{
		std::vector<float>  NodeFeatureMid;//每个叶子的中心点 
		LeafNodeSet[i] = m_Trees[i]->locateLeafNode(vFeatures);
		//叶子均值预测
		PredictValueOfTree[i] = m_Trees[i]->predict(*LeafNodeSet[i], vFeatures, NodeWeight[i], vResponseIndex);
		VarInfo << i << "," << PredictValueOfTree[i]<<",";
		std::vector<int> nodeDataIndex;
		//遍历其节点信息
		nodeDataIndex = LeafNodeSet[i]->getNodeDataIndexV();//pWeightedPathNode->calNodeDataIndex(LeafBotherNodeSet[i]); 

	    //去重复点---------------------------比不去掉效果提升千分之8-9点
		//nodeDataIndex.erase(unique(nodeDataIndex.begin(), nodeDataIndex.end()), nodeDataIndex.end());
		////去重复点---------------------------
		//std::vector<int>  uniqueNodeDataIndex = pWeightedPathNode->uniqueData(nodeDataIndex);
		//nodeDataIndex.resize(uniqueNodeDataIndex.size());
		//nodeDataIndex.assign(uniqueNodeDataIndex.begin(), uniqueNodeDataIndex.end()); 
		std::vector<std::vector<float>> NodeDataFeature(nodeDataIndex.size(), std::vector<float>());
		std::vector<float> NodeDataResponse(nodeDataIndex.size(), 0.f);
		for (int k = 0; k < nodeDataIndex.size(); ++k)
		{
			NodeDataFeature[k] = pTrainingSet->getFeatureInstanceAt(nodeDataIndex[k]);
			NodeDataResponse[k] = pTrainingSet->getResponseValueAt(nodeDataIndex[k]);
			treeTotalNodeDataIndex.push_back(nodeDataIndex[k]);
		}
		 //计算每个叶子的中心点
		for (int j = 0;j < NodeDataFeature[0].size();j++)//列
		{
			float featureMid = 0.0f;
			for (int i = 0; i < NodeDataFeature.size(); i++)//行
			{
				featureMid += NodeDataFeature[i][j];
			}
			NodeFeatureMid.push_back({ featureMid / NodeDataFeature.size() });//每列/维度的均值
		}


		//计算当前叶子节点的y的方差
		treeNodeVar[i] = pTrainingSet->calTreeResponseVar(NodeDataResponse);
		treeNodeFeatureMid.push_back({ NodeFeatureMid });//每棵树的叶子中心点  
		VarInfo << treeNodeVar[i] << ","<< NodeDataResponse.size()<<std::endl;
		

	}
	////去重复点---------------------------
	//std::vector<int>  uniquetreeTotalNodeDataIndex = pWeightedPathNode->uniqueData(treeTotalNodeDataIndex);
	//treeTotalNodeDataIndex.resize(uniquetreeTotalNodeDataIndex.size());
	//treeTotalNodeDataIndex.assign(uniquetreeTotalNodeDataIndex.begin(), uniquetreeTotalNodeDataIndex.end());
   
	//集合到一起
	float sumOfuniqueResponse = 0.0f;
	for (int k = 0; k < treeTotalNodeDataIndex.size(); ++k)
	{
		treeTotalNodeDataFeature.push_back({ pTrainingSet->getFeatureInstanceAt(treeTotalNodeDataIndex[k]) }); 
		treeTotalNodeDataResponse.push_back({ pTrainingSet->getResponseValueAt(treeTotalNodeDataIndex[k]) });
		sumOfuniqueResponse += treeTotalNodeDataResponse[k];
	}
	 
	// PredictKmeansForest = pTrainingSet->calKMeansForest(treeNodeFeatureMid, PredictValueOfTree, treeTotalNodeDataFeature, treeTotalNodeDataResponse, vFeatures);
	maxResponseVarOfTree = treeNodeVar[0];
	//记录叶子的最大方差
	for (int j = 1;j < treeNodeVar.size();j++)
	{
		if (maxResponseVarOfTree < treeNodeVar[j])
			maxResponseVarOfTree = treeNodeVar[j];
	}
	
	// PredictKmeansForest = pTrainingSet->calKMeansForestByResponseVarRatio(treeNodeVar, maxResponseVarOfTree, maxResponseVarRatio,treeNodeFeatureMid, PredictValueOfTree, treeTotalNodeDataFeature, treeTotalNodeDataResponse, vFeatures);
	//PredictKmeansForest = sumOfuniqueResponse / treeTotalNodeDataIndex.size();

	float SumPredict = std::accumulate(PredictValueOfTree.begin(), PredictValueOfTree.end(), 0.f);
	return SumPredict / vNumOfUsingTrees;

	

}
//根据Var加权每棵树进行预测------------------------------add by zy 2019.3.29-------------
float CRegressionForest::__predictByVarWeightTree(const std::vector<float>& vFeatures, unsigned int vNumOfUsingTrees, float& voLpWeightTreePredict, unsigned int vResponseIndex) const
{
	_ASSERTE(!vFeatures.empty() && vNumOfUsingTrees > 0); 
	float PredictValue = 0.0f;
	std::vector<float> PredictValueOfTree(vNumOfUsingTrees, 0.0f); 
	std::vector<float> PredictVarWeightTree(vNumOfUsingTrees, 0.0f); 
	std::vector<float> NodeWeight(vNumOfUsingTrees, 0.0f); 
	std::vector<std::vector<int>> NodeDataIndex(vNumOfUsingTrees, std::vector<int>()); 
	CTrainingSet* pTrainingSet = CTrainingSet::getInstance();
	CWeightedPathNodeMethod* pWeightedPathNode = CWeightedPathNodeMethod::getInstance();
	static std::vector<const CNode*> LeafNodeSet; 
	LeafNodeSet.resize(this->getNumOfTrees());  
	std::vector<float> treeNodeVar(vNumOfUsingTrees, 0.0f);
	std::vector<std::vector<float>> treeNodeFeatureVar(vNumOfUsingTrees);//记录树的叶子的每个维度的方差
	std::vector<std::vector<float>> treeFeatsVarChangedRatio(vNumOfUsingTrees);//记录投入特征之后叶子的方差变化率-变化率大，相似度低
	std::ofstream VarInfo;
	VarInfo.open("VarInfo.csv", std::ios::app);
	/*std::ofstream LeafNodeDisInfo;
	LeafNodeDisInfo.open("LeafNodeDisInfo.csv", std::ios::app);*/
	for (int i = 0; i < vNumOfUsingTrees; ++i)
	{
		
			LeafNodeSet[i] = m_Trees[i]->locateLeafNode(vFeatures);
			//叶子均值预测
			PredictValueOfTree[i] = m_Trees[i]->predict(*LeafNodeSet[i], vFeatures, NodeWeight[i], vResponseIndex);
			VarInfo << i << "," << PredictValueOfTree[i] << ",";
			std::vector<int> nodeDataIndex; 
			//遍历其节点信息
			nodeDataIndex = LeafNodeSet[i]->getNodeDataIndexV(); 
			
			std::vector<std::vector<float>> NodeDataFeature(nodeDataIndex.size(), std::vector<float>());
			std::vector<float> NodeDataResponse(nodeDataIndex.size(), 0.f);
			float sumy = 0.0f;

			for (int k = 0; k < nodeDataIndex.size(); ++k)
			{
				NodeDataFeature[k] = pTrainingSet->getFeatureInstanceAt(nodeDataIndex[k]);
				NodeDataResponse[k] = pTrainingSet->getResponseValueAt(nodeDataIndex[k]);
				//LeafNodeDisInfo << i << "," << nodeDataIndex[k] << "," << NodeDataResponse[k] << std::endl;
			} 
			 
			PredictVarWeightTree[i] = pTrainingSet->calLeafWeightForTreeByLp(NodeDataFeature, vFeatures);
			// 计算当前叶子节点的y的方差
			treeNodeVar[i] = pTrainingSet->calTreeResponseVar(NodeDataResponse);
			VarInfo << treeNodeVar[i] << ",";
			//计算当前叶子节点各个维度的方差--加入测试点,同时计算测试点带来的方差变化差-----------------------
			std::vector<float> FeatsVarChangedRatio(vFeatures.size(), 0.0f);
			treeNodeFeatureVar[i] = pTrainingSet->calTreeFeaturesVar(vFeatures,NodeDataFeature, FeatsVarChangedRatio);
			treeFeatsVarChangedRatio[i] = FeatsVarChangedRatio;
			for (int k = 0; k < treeNodeFeatureVar[i].size(); k++)
			{
				VarInfo << treeNodeFeatureVar[i][k] << ",";
			}
			VarInfo << NodeDataResponse.size() << "," << LeafNodeSet[i]->getLevel() << std::endl;
	}

	float SumPredict = std::accumulate(PredictValueOfTree.begin(), PredictValueOfTree.end(), 0.f);
	std::vector<float> distanceOfTree(PredictVarWeightTree.size());
	for (int k = 0; k < PredictVarWeightTree.size(); ++k)
	{
	distanceOfTree[k] = PredictVarWeightTree[k];

	}
	//相似度计算：方法1-计算LP距离，方法2-计算投入测试点后的方差变化率
	std::vector<float> wiDisOfTree = pTrainingSet->antiDistanceWeight(distanceOfTree, -1); 
	std::vector<float> wiVarOfTree = pTrainingSet->antiVarWeight(treeNodeVar, -1);
	std::vector<std::vector<float>>  wiFeatsOfTree = pTrainingSet->antiFeatsWeight(treeNodeFeatureVar, -1);
	std::vector<std::vector<float>>  wiFeatChangedRatioOfTree = pTrainingSet->antiFeatsWeight(treeFeatsVarChangedRatio, -1);
	std::vector<float> combWiOfTree(vNumOfUsingTrees);

	// combWiOfTree = __calWeightOfTree(vNumOfUsingTrees, wiFeatChangedRatioOfTree );// 测试点引起特征方差变换率计算权重
	//  combWiOfTree = __calWeightOfTree(vNumOfUsingTrees, wiVarOfTree, wiFeatChangedRatioOfTree,2);//Y方差、测试点引起特征方差变换率计算权重
	combWiOfTree=__calWeightOfTree(vNumOfUsingTrees, wiFeatsOfTree, wiVarOfTree, wiFeatChangedRatioOfTree,3);//特征方差、Y方差、测试点引起特征方差变换率计算权重
	//combWiOfTree = __calWeightOfTree(vNumOfUsingTrees, wiFeatsOfTree, wiVarOfTree, wiDisOfTree, 3);//特征方差、Y方差、测试点到叶子中心点距离计算权重
	//combWiOfTree = __calWeightOfTree(vNumOfUsingTrees, wiFeatsOfTree, wiVarOfTree,2);//特征方差、Y方差计算权重，比例5：5
	// combWiOfTree = __calWeightOfTree(vNumOfUsingTrees, wiFeatsOfTree, wiVarOfTree);//特征方差、Y方差计算权重，所有特征和y方差比例1：1：1
	_ASSERTE(PredictValueOfTree.size() == NodeWeight.size());
	for (int k = 0; k < PredictVarWeightTree.size(); ++k)
	{
		voLpWeightTreePredict += PredictValueOfTree[k] * combWiOfTree[k];

	}


	SumPredict = SumPredict / vNumOfUsingTrees;
	return  SumPredict;
	 

}

std::vector<float> CRegressionForest::__calWeightOfTree(int vNumOfUsingTrees, std::vector<std::vector<float>>&  wiFeatsOfTree, std::vector<float>& wiVarOfTree, std::vector<float> wiDisOfTree)const
{
	std::vector<float> combWiOfTree(vNumOfUsingTrees, 0.0f);
	for (int k = 0; k < vNumOfUsingTrees; ++k)
	{
		combWiOfTree[k] = wiVarOfTree[k]+ wiDisOfTree[k];
		for (int j = 0;j < wiFeatsOfTree[k].size();j++)
		{
			combWiOfTree[k] += wiFeatsOfTree[k][j];

		}

	}

	std::ofstream WiInfo;
	WiInfo.open("WiInfo.csv", std::ios::app);
	for (int k = 0; k < vNumOfUsingTrees; ++k)
	{

		combWiOfTree[k] = combWiOfTree[k] / (wiFeatsOfTree[0].size() + 2);

		WiInfo << combWiOfTree[k] << ",";
	}
	WiInfo << std::endl;

	return combWiOfTree;



}
//Response 和各项特征占权重比例一样1：1：1。。。。：1
std::vector<float> CRegressionForest::__calWeightOfTree(int vNumOfUsingTrees, std::vector<std::vector<float>>&  wiFeatsOfTree, std::vector<float>& wiVarOfTree)const
{
	std::vector<float> combWiOfTree(vNumOfUsingTrees,0.0f);
	for (int k = 0; k < vNumOfUsingTrees; ++k)
	{
	    combWiOfTree[k] = wiVarOfTree[k]; 
		for (int j = 0;j < wiFeatsOfTree[k].size();j++)
		{
			combWiOfTree[k] += wiFeatsOfTree[k][j];

		}

	}

	std::ofstream WiInfo;
	WiInfo.open("WiInfo.csv", std::ios::app);
	for (int k = 0; k < vNumOfUsingTrees; ++k)
	{ 

	    combWiOfTree[k] = combWiOfTree[k] / (wiFeatsOfTree[0].size() + 1); 
		 
		WiInfo << combWiOfTree[k] << ",";
	}
	WiInfo << std::endl; 

	return combWiOfTree;



}
//Response 和特征总和 占权重比例一样 5：5
std::vector<float> CRegressionForest::__calWeightOfTree(int vNumOfUsingTrees, std::vector<std::vector<float>>&  wiFeatsOfTree, std::vector<float>& wiVarOfTree,int numOfWi)const
{
	std::vector<float> combWiOfTree(vNumOfUsingTrees, 0.0f);

	for (int k = 0; k < vNumOfUsingTrees; ++k)
	{
		
		for (int j = 0;j < wiFeatsOfTree[k].size();j++)
		{
			combWiOfTree[k] += wiFeatsOfTree[k][j];
			 
		}


	}
	std::ofstream WiInfo;
	WiInfo.open("WiInfo.csv", std::ios::app);
	for (int k = 0; k < vNumOfUsingTrees; ++k)
	{

		combWiOfTree[k] = combWiOfTree[k] / wiFeatsOfTree[0].size();
		float sumOfFeatAndResp = combWiOfTree[k] + wiVarOfTree[k];
		combWiOfTree[k] = sumOfFeatAndResp / numOfWi;
		WiInfo << combWiOfTree[k] << ",";
	}
	WiInfo << std::endl;

	return combWiOfTree;



}
//根据测试点加入叶子后引起的方差变化率计算相似度计算权重
std::vector<float> CRegressionForest::__calWeightOfTree(int vNumOfUsingTrees,  std::vector<std::vector<float>>&  wiFeatsChangedRatioOfTree)const
{
	std::vector<float> combWiOfTree(vNumOfUsingTrees, 0.0f);
	std::vector<float> FeatVarRatioOfTree(vNumOfUsingTrees, 0.0f);
	for (int k = 0; k < vNumOfUsingTrees; ++k)
	{

		for (int j = 0;j < wiFeatsChangedRatioOfTree[k].size();j++)
		{

			FeatVarRatioOfTree[k] += wiFeatsChangedRatioOfTree[k][j];
		}

	}
	std::ofstream WiInfo;
	WiInfo.open("WiInfo.csv", std::ios::app);
	for (int k = 0; k < vNumOfUsingTrees; ++k)
	{

		FeatVarRatioOfTree[k] = FeatVarRatioOfTree[k] / wiFeatsChangedRatioOfTree[0].size(); 
		combWiOfTree[k] = FeatVarRatioOfTree[k];
		WiInfo << combWiOfTree[k] << ",";
	}
	WiInfo << std::endl;

	return combWiOfTree;



}

//根据Y的方差 、测试点加入叶子后引起的方差变化率计算相似度计算权重
std::vector<float> CRegressionForest::__calWeightOfTree(int vNumOfUsingTrees,  std::vector<float>& wiVarOfTree, std::vector<std::vector<float>>&  wiFeatsChangedRatioOfTree, int numOfWi)const
{
	std::vector<float> combWiOfTree(vNumOfUsingTrees, 0.0f); 
	std::vector<float> FeatVarRatioOfTree(vNumOfUsingTrees, 0.0f);
	for (int k = 0; k < vNumOfUsingTrees; ++k)
	{

		for (int j = 0;j < wiFeatsChangedRatioOfTree[k].size();j++)
		{
			 
			FeatVarRatioOfTree[k] += wiFeatsChangedRatioOfTree[k][j];
		}

	}
	std::ofstream WiInfo;
	WiInfo.open("WiInfo.csv", std::ios::app);
	for (int k = 0; k < vNumOfUsingTrees; ++k)
	{
		 
		FeatVarRatioOfTree[k] = FeatVarRatioOfTree[k] / wiFeatsChangedRatioOfTree[0].size();
		float sumOfFeatAndResp =  wiVarOfTree[k] + FeatVarRatioOfTree[k];
		combWiOfTree[k] = sumOfFeatAndResp / numOfWi;
		WiInfo << combWiOfTree[k] << ",";
	}
	WiInfo << std::endl;

	return combWiOfTree;



}
//根据树的特征方差，Y的方差 和测试点加入叶子后引起的方差变化率计算相似度，三者计算权重，三者所占比例均分
std::vector<float> CRegressionForest::__calWeightOfTree(int vNumOfUsingTrees, std::vector<std::vector<float>>&  wiFeatsOfTree, std::vector<float>& wiVarOfTree, std::vector<std::vector<float>>&  wiFeatsChangedRatioOfTree, int numOfWi)const
{
	std::vector<float> combWiOfTree(vNumOfUsingTrees, 0.0f);
	std::vector<float> FeatVarWiOfTree(vNumOfUsingTrees, 0.0f);
	std::vector<float> FeatVarRatioOfTree(vNumOfUsingTrees, 0.0f);
	for (int k = 0; k < vNumOfUsingTrees; ++k)
	{

		for (int j = 0;j < wiFeatsOfTree[k].size();j++)
		{
			FeatVarWiOfTree[k] += wiFeatsOfTree[k][j];
			FeatVarRatioOfTree[k] += wiFeatsChangedRatioOfTree[k][j];
		}

	}
	std::ofstream WiInfo;
	WiInfo.open("WiInfo.csv", std::ios::app);
	for (int k = 0; k < vNumOfUsingTrees; ++k)
	{

		FeatVarWiOfTree[k] = FeatVarWiOfTree[k] / wiFeatsOfTree[0].size();
		FeatVarRatioOfTree[k]= FeatVarRatioOfTree[k]/ wiFeatsOfTree[0].size();
		float sumOfFeatAndResp = FeatVarWiOfTree[k] + wiVarOfTree[k] + FeatVarRatioOfTree[k];
		combWiOfTree[k] = sumOfFeatAndResp / numOfWi;
		WiInfo << combWiOfTree[k] << ",";
	}
	WiInfo << std::endl;

	return combWiOfTree;



}
//根据树的特征方差，Y的方差 和测试点到叶子中心的相似度计算权重，三者所占比例均分
std::vector<float> CRegressionForest::__calWeightOfTree(int vNumOfUsingTrees, std::vector<std::vector<float>>&  wiFeatsOfTree, std::vector<float>& wiVarOfTree, std::vector<float> wiDisOfTree, int numOfWi)const
{
	std::vector<float> combWiOfTree(vNumOfUsingTrees, 0.0f);
	for (int k = 0; k < vNumOfUsingTrees; ++k)
	{

		for (int j = 0;j < wiFeatsOfTree[k].size();j++)
		{
			combWiOfTree[k] += wiFeatsOfTree[k][j];

		}

	}
	std::ofstream WiInfo;
	WiInfo.open("WiInfo.csv", std::ios::app);
	for (int k = 0; k < vNumOfUsingTrees; ++k)
	{

		combWiOfTree[k] = combWiOfTree[k] / wiFeatsOfTree[0].size();
		float sumOfFeatAndResp = combWiOfTree[k] + wiVarOfTree[k]+ wiDisOfTree[k];
		combWiOfTree[k] = sumOfFeatAndResp / numOfWi;
		WiInfo << combWiOfTree[k] << ",";
	}
	WiInfo << std::endl;

	return combWiOfTree;



}
//根据MP加权每棵树进行预测------------------------------add by zy 2019.3.29-------------
float CRegressionForest::__predictByMpWeightTree(const std::vector<float>& vFeatures, unsigned int vNumOfUsingTrees,   float& voMpWeightTreePredict, unsigned int vResponseIndex) const
{
	_ASSERTE(!vFeatures.empty() && vNumOfUsingTrees > 0); 
	float PredictValue = 0.0f;
	std::vector<float> PredictValueOfTree(vNumOfUsingTrees, 0.0f); 
	std::vector<std::pair<float, float>>  PredictMPWeightTree;
	std::vector<float> NodeWeight(vNumOfUsingTrees, 0.0f);
	std::vector<int> TotalDataIndex;
	std::vector<std::vector<int>> NodeDataIndex(vNumOfUsingTrees, std::vector<int>()); 
	CTrainingSet* pTrainingSet = CTrainingSet::getInstance();
	CWeightedPathNodeMethod* pWeightedPathNode = CWeightedPathNodeMethod::getInstance(); 
	static std::vector<const CNode*> LeafNodeSet; 
	std::vector<std::vector<float>> treeTotalNodeDataFeature;
	std::vector<float> treeTotalNodeDataResponse;
	std::vector<std::vector<float>>  treeNodeFeatureMid;//每棵树的每个叶子的中心点 
	LeafNodeSet.resize(this->getNumOfTrees());
	std::ofstream Yinfo;
	//Yinfo.open("Yinfo.csv", std::ios::app);
	for (int i = 0; i < vNumOfUsingTrees; ++i)
	{ 
		std::vector<float>  NodeFeatureMid;//每个叶子的中心点 
		LeafNodeSet[i] = m_Trees[i]->locateLeafNode(vFeatures);
		//叶子均值预测
		PredictValueOfTree[i] = m_Trees[i]->predict(*LeafNodeSet[i], vFeatures, NodeWeight[i], vResponseIndex);
	    std::vector<int> nodeDataIndex; 
	    //遍历其节点信息
		nodeDataIndex = LeafNodeSet[i]->getNodeDataIndexV();//pWeightedPathNode->calNodeDataIndex(LeafBotherNodeSet[i]);
		
	 
		//去重复点---------------------------比不去掉效果提升千分之8-9点
		nodeDataIndex.erase(unique(nodeDataIndex.begin(), nodeDataIndex.end()), nodeDataIndex.end());

		std::vector<std::vector<float>> NodeDataFeature(nodeDataIndex.size(), std::vector<float>());
		std::vector<float> NodeDataResponse(nodeDataIndex.size(), 0.f);
			for (int k = 0; k < nodeDataIndex.size(); ++k)
			{
				NodeDataFeature[k] = pTrainingSet->getFeatureInstanceAt(nodeDataIndex[k]);
				NodeDataResponse[k] = pTrainingSet->getResponseValueAt(nodeDataIndex[k]); 

				treeTotalNodeDataFeature.push_back(NodeDataFeature[k]);
				treeTotalNodeDataResponse.push_back(NodeDataResponse[k]);
				 
			}
			 
			for (int j = 0;j < NodeDataFeature[0].size();j++)//列
			{
				float featureMid = 0.0f;
				for (int i = 0; i < NodeDataFeature.size(); i++)//行
				{
					featureMid += NodeDataFeature[i][j];
				}
				NodeFeatureMid.push_back({ featureMid / NodeDataFeature.size() });//每列/维度的均值
			}

			treeNodeFeatureMid.push_back({ NodeFeatureMid });//每棵树的叶子中心点  
	 
		 

	
	}



	//遍历森林，计算每棵树落入叶子的权重（即树的权重MP）,将所有叶子的样本合成的总样本空间作为n，来计算测试点到每个叶子中心点的mp，通过mp计算权重
	PredictMPWeightTree = pTrainingSet->calLeafWeightForTreeByMp(treeNodeFeatureMid, PredictValueOfTree, treeTotalNodeDataFeature, treeTotalNodeDataResponse, vFeatures);
	 
	std::vector<float> distanceOfTree(PredictMPWeightTree.size()); 
	float SumPredict = std::accumulate(PredictValueOfTree.begin(), PredictValueOfTree.end(), 0.f);  
	for (int k = 0; k < PredictMPWeightTree.size(); ++k)
	{
		distanceOfTree[k] = PredictMPWeightTree[k].second;

	}
 
	std::vector<float> wiOfTree = pTrainingSet->antiDistanceWeight(distanceOfTree, -2);
	for (int k = 0; k < PredictMPWeightTree.size(); ++k)
	{
		voMpWeightTreePredict += PredictMPWeightTree[k].first * wiOfTree[k];

	}
	 
	_ASSERTE(PredictValueOfTree.size() == NodeWeight.size()); 

	 return SumPredict / vNumOfUsingTrees;
	
}


 
//********************************************************************************************************
//FUNCTION:
bool CRegressionForest::operator==(const CRegressionForest& vRegressionForest) const
{
	// NOTES : 这里没有比较 OOB Error，两个原因：
	//         1、由于如果森林模型中树都一致，那么 OOB Error也会一致

	if (m_Trees.size() != vRegressionForest.getNumOfTrees()) return false;

	for (auto Index = 0; Index < m_Trees.size(); ++Index)
	{
		if (!(m_Trees[Index]->operator==(*(vRegressionForest.getTreeAt(Index))))) return false;
	}

	return true;
}

//****************************************************************************************************
//FUNCTION:
void CRegressionForest::__initForest()
{
	int NumOfTrees = CRegressionForestConfig::getInstance()->getAttribute<int>(KEY_WORDS::NUMBER_OF_TREE);
	m_Trees.resize(NumOfTrees);

	for (unsigned int i = 0; i < m_Trees.size(); ++i)
	{
		// NOTES : 判断建树的方式（2-stage等）
		if (!CRegressionForestConfig::getInstance()->isAttributeExisted(KEY_WORDS::BUILD_TREE_TYPE))
			m_Trees[i] = new CTree();
		else
			m_Trees[i] = dynamic_cast<CTree*>(hiveOO::CProductFactoryData::getInstance()->createProduct(CRegressionForestConfig::getInstance()->getAttribute<std::string>(KEY_WORDS::BUILD_TREE_TYPE)));
	}
}

//****************************************************************************************************
//FUNCTION:
void CRegressionForest::__initForestParameters(IBootstrapSelector*& voBootstrapSelector, IFeatureSelector*& voFeatureSelector, INodeSpliter*& voNodeSpliter, IBaseTerminateCondition*& voTerminateCondition, IFeatureWeightGenerator*& voFeatureWeightMethod)
{
	const CRegressionForestConfig *pRegressionForestConfig = CRegressionForestConfig::getInstance();
	
	std::string BootstrapSelectorSig = pRegressionForestConfig->getAttribute<std::string>(KEY_WORDS::BOOTSTRAP_SELECTOR);
	voBootstrapSelector = dynamic_cast<IBootstrapSelector*>(hiveOO::CProductFactoryData::getInstance()->createProduct(BootstrapSelectorSig));
	
	std::string FeatureSelectorSig = pRegressionForestConfig->isAttributeExisted(KEY_WORDS::FEATURE_SELECTOR) ? pRegressionForestConfig->getAttribute<std::string>(KEY_WORDS::FEATURE_SELECTOR) : KEY_WORDS::UNIFORM_FEATURE_SELECTOR;
	voFeatureSelector = dynamic_cast<IFeatureSelector*>(hiveOO::CProductFactoryData::getInstance()->createProduct(FeatureSelectorSig));
	
	std::string SpliterSig = pRegressionForestConfig->getAttribute<std::string>(KEY_WORDS::NODE_SPLIT_METHOD);
	voNodeSpliter = dynamic_cast<INodeSpliter*>(hiveOO::CProductFactoryData::getInstance()->createProduct(SpliterSig));
																										
	std::string TerminateConditionSig = pRegressionForestConfig->getAttribute<std::string>(KEY_WORDS::LEAF_NODE_CONDITION);
	voTerminateCondition = dynamic_cast<IBaseTerminateCondition*>(hiveOO::CProductFactoryData::getInstance()->createProduct(TerminateConditionSig));

	std::string FeatureWeightSig = pRegressionForestConfig->getAttribute<std::string>(KEY_WORDS::FEATURE_WEIGHT_CALCULATE_METHOD);
	voFeatureWeightMethod = dynamic_cast<IFeatureWeightGenerator*>(hiveOO::CProductFactoryData::getInstance()->createProduct(FeatureWeightSig));
}

//****************************************************************************************************
//FUNCTION:
void CRegressionForest::__clearForest()
{
	for (unsigned int i = 0; i < m_Trees.size(); ++i)
		if (m_Trees[i]) _SAFE_DELETE(m_Trees[i]);
}

void CRegressionForest::outputOOBInfo(const std::string& vOutputFileName) const
{
	std::fstream OutPutFile(vOutputFileName, std::ios::out);
	if (OutPutFile.is_open())
	{
		OutPutFile << "Node Number" << "," << "OOB Number" <<std::endl;
		for (auto Itr:m_Trees)
		{
			STreeInfo TreeInfo;
			Itr->fetchTreeInfo(TreeInfo);
			OutPutFile << TreeInfo.m_NumOfLeafNodes << "," << Itr->getOOBIndexSet().size() << std::endl;
		}
	}
}
