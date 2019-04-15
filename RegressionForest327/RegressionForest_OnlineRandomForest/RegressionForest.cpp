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
//FUNCTION:Ҫ��д��
float CRegressionForest::predict(const std::vector<float>& vFeatures, unsigned int vNumOfUsingTrees, float& voKnnPredictSet,    float& voVarWiPerdict, bool vIsWeightedPrediction) const
{
	//return __predictByKMeansForest(vFeatures, vNumOfUsingTrees, voIFMPPerdict, false);//��Ҷ�Ӽ��ϣ���KMean����Ԥ��
	//return __predictByMpWeightTree(vFeatures, vNumOfUsingTrees, voIFMPPerdict, false);//MP������Ȩ
	return	__predictByVarWeightTree(vFeatures, vNumOfUsingTrees, voVarWiPerdict, false);//Var������Ȩ
	//return __predictCertainResponse(vFeatures, vNumOfUsingTrees, vIsWeightedPrediction, voLPPredictSet, voMPPredictSet,  voLLPerdict,  voLLPSet, voMPDissimilarity,  voIFMPPerdict);
}
//****************************************************************************************************
//FUNCTION:
float CRegressionForest::predict(const std::vector<float>& vFeatures, unsigned int vNumOfUsingTrees, float& voLPPredictSet, float& voMPPredictSet, float& voLLPerdict, float& voLLPSet, float& voMPDissimilarity, float& voIFMPPerdict, bool vIsWeightedPrediction) const
{ 
	 //return __predictByKMeansForest(vFeatures, vNumOfUsingTrees, voIFMPPerdict, false);//��Ҷ�Ӽ��ϣ���KMean����Ԥ��
   // return __predictByMpWeightTree(vFeatures, vNumOfUsingTrees, voIFMPPerdict, false);//MP������Ȩ
	 return	__predictByVarWeightTree(vFeatures, vNumOfUsingTrees, voIFMPPerdict, false);//LP������Ȩ
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

//���ݶ�Ҷ�ӽڵ���KMeans����������ȨԤ��---------------------------------add by zy 2019.3.29
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
	std::vector<std::vector<float>>  treeNodeFeatureMid;//ÿ������ÿ��Ҷ�ӵ����ĵ� 
	float maxResponseVarRatio = 1.0f;
	float maxResponseVarOfTree = 0.0f;
	std::vector<float> treeNodeVar(vNumOfUsingTrees, 0.0f);
	std::ofstream VarInfo;
	VarInfo.open("VarInfo.csv", std::ios::app);
	for (int i = 0; i < vNumOfUsingTrees; ++i)
	{
		std::vector<float>  NodeFeatureMid;//ÿ��Ҷ�ӵ����ĵ� 
		LeafNodeSet[i] = m_Trees[i]->locateLeafNode(vFeatures);
		//Ҷ�Ӿ�ֵԤ��
		PredictValueOfTree[i] = m_Trees[i]->predict(*LeafNodeSet[i], vFeatures, NodeWeight[i], vResponseIndex);
		VarInfo << i << "," << PredictValueOfTree[i]<<",";
		std::vector<int> nodeDataIndex;
		//������ڵ���Ϣ
		nodeDataIndex = LeafNodeSet[i]->getNodeDataIndexV();//pWeightedPathNode->calNodeDataIndex(LeafBotherNodeSet[i]); 

	    //ȥ�ظ���---------------------------�Ȳ�ȥ��Ч������ǧ��֮8-9��
		//nodeDataIndex.erase(unique(nodeDataIndex.begin(), nodeDataIndex.end()), nodeDataIndex.end());
		////ȥ�ظ���---------------------------
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
		 //����ÿ��Ҷ�ӵ����ĵ�
		for (int j = 0;j < NodeDataFeature[0].size();j++)//��
		{
			float featureMid = 0.0f;
			for (int i = 0; i < NodeDataFeature.size(); i++)//��
			{
				featureMid += NodeDataFeature[i][j];
			}
			NodeFeatureMid.push_back({ featureMid / NodeDataFeature.size() });//ÿ��/ά�ȵľ�ֵ
		}


		//���㵱ǰҶ�ӽڵ��y�ķ���
		treeNodeVar[i] = pTrainingSet->calTreeResponseVar(NodeDataResponse);
		treeNodeFeatureMid.push_back({ NodeFeatureMid });//ÿ������Ҷ�����ĵ�  
		VarInfo << treeNodeVar[i] << ","<< NodeDataResponse.size()<<std::endl;
		

	}
	////ȥ�ظ���---------------------------
	//std::vector<int>  uniquetreeTotalNodeDataIndex = pWeightedPathNode->uniqueData(treeTotalNodeDataIndex);
	//treeTotalNodeDataIndex.resize(uniquetreeTotalNodeDataIndex.size());
	//treeTotalNodeDataIndex.assign(uniquetreeTotalNodeDataIndex.begin(), uniquetreeTotalNodeDataIndex.end());
   
	//���ϵ�һ��
	float sumOfuniqueResponse = 0.0f;
	for (int k = 0; k < treeTotalNodeDataIndex.size(); ++k)
	{
		treeTotalNodeDataFeature.push_back({ pTrainingSet->getFeatureInstanceAt(treeTotalNodeDataIndex[k]) }); 
		treeTotalNodeDataResponse.push_back({ pTrainingSet->getResponseValueAt(treeTotalNodeDataIndex[k]) });
		sumOfuniqueResponse += treeTotalNodeDataResponse[k];
	}
	 
	// PredictKmeansForest = pTrainingSet->calKMeansForest(treeNodeFeatureMid, PredictValueOfTree, treeTotalNodeDataFeature, treeTotalNodeDataResponse, vFeatures);
	maxResponseVarOfTree = treeNodeVar[0];
	//��¼Ҷ�ӵ���󷽲�
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
//����Var��Ȩÿ��������Ԥ��------------------------------add by zy 2019.3.29-------------Ҫ��д��
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
	std::vector<std::vector<float>> treeNodeFeatureVar(vNumOfUsingTrees);//��¼����Ҷ�ӵ�ÿ��ά�ȵķ���
	std::vector<std::vector<float>> treeFeatsVarChangedRatio(vNumOfUsingTrees);//��¼Ͷ������֮��Ҷ�ӵķ���仯��-�仯�ʴ����ƶȵ�
	std::ofstream VarInfo;
	VarInfo.open("VarInfo.csv", std::ios::app);
	/*std::ofstream LeafNodeDisInfo;
	LeafNodeDisInfo.open("LeafNodeDisInfo.csv", std::ios::app);*/
	for (int i = 0; i < vNumOfUsingTrees; ++i)
	{
		
			LeafNodeSet[i] = m_Trees[i]->locateLeafNode(vFeatures);
			//Ҷ�Ӿ�ֵԤ��
			PredictValueOfTree[i] = m_Trees[i]->predict(*LeafNodeSet[i], vFeatures, NodeWeight[i], vResponseIndex);
			VarInfo << i << "," << PredictValueOfTree[i] << ",";
			std::vector<int> nodeDataIndex; 
			//������ڵ���Ϣ
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
			// ���㵱ǰҶ�ӽڵ��y�ķ���
			treeNodeVar[i] = pTrainingSet->calTreeResponseVar(NodeDataResponse);
			VarInfo << treeNodeVar[i] << ",";
			//���㵱ǰҶ�ӽڵ����ά�ȵķ���--������Ե�,ͬʱ������Ե�����ķ���仯��-----------------------
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
	//���ƶȼ��㣺����1-����LP���룬����2-����Ͷ����Ե��ķ���仯��
	std::vector<float> wiDisOfTree = pTrainingSet->antiDistanceWeight(distanceOfTree, -1); 
	std::vector<float> wiVarOfTree = pTrainingSet->antiVarWeight(treeNodeVar, -1);
	std::vector<std::vector<float>>  wiFeatsOfTree = pTrainingSet->antiFeatsWeight(treeNodeFeatureVar, -1);
	std::vector<std::vector<float>>  wiFeatChangedRatioOfTree = pTrainingSet->antiFeatsWeight(treeFeatsVarChangedRatio, -1);
	std::vector<float> combWiOfTree(vNumOfUsingTrees);

	// combWiOfTree = __calWeightOfTree(vNumOfUsingTrees, wiFeatChangedRatioOfTree );// ���Ե�������������任�ʼ���Ȩ��
	//  combWiOfTree = __calWeightOfTree(vNumOfUsingTrees, wiVarOfTree, wiFeatChangedRatioOfTree,2);//Y������Ե�������������任�ʼ���Ȩ��
	combWiOfTree=__calWeightOfTree(vNumOfUsingTrees, wiFeatsOfTree, wiVarOfTree, wiFeatChangedRatioOfTree,3);//�������Y������Ե�������������任�ʼ���Ȩ��
	//combWiOfTree = __calWeightOfTree(vNumOfUsingTrees, wiFeatsOfTree, wiVarOfTree, wiDisOfTree, 3);//�������Y������Ե㵽Ҷ�����ĵ�������Ȩ��
	//combWiOfTree = __calWeightOfTree(vNumOfUsingTrees, wiFeatsOfTree, wiVarOfTree,2);//�������Y�������Ȩ�أ�����5��5
	// combWiOfTree = __calWeightOfTree(vNumOfUsingTrees, wiFeatsOfTree, wiVarOfTree);//�������Y�������Ȩ�أ�����������y�������1��1��1
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
//Response �͸�������ռȨ�ر���һ��1��1��1����������1
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
//Response �������ܺ� ռȨ�ر���һ�� 5��5
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
//���ݲ��Ե����Ҷ�Ӻ�����ķ���仯�ʼ������ƶȼ���Ȩ��
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

//����Y�ķ��� �����Ե����Ҷ�Ӻ�����ķ���仯�ʼ������ƶȼ���Ȩ��
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
//���������������Y�ķ��� �Ͳ��Ե����Ҷ�Ӻ�����ķ���仯�ʼ������ƶȣ����߼���Ȩ�أ�������ռ��������
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
//���������������Y�ķ��� �Ͳ��Ե㵽Ҷ�����ĵ����ƶȼ���Ȩ�أ�������ռ��������
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
//����MP��Ȩÿ��������Ԥ��------------------------------add by zy 2019.3.29-------------
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
	std::vector<std::vector<float>>  treeNodeFeatureMid;//ÿ������ÿ��Ҷ�ӵ����ĵ� 
	LeafNodeSet.resize(this->getNumOfTrees());
	std::ofstream Yinfo;
	//Yinfo.open("Yinfo.csv", std::ios::app);
	for (int i = 0; i < vNumOfUsingTrees; ++i)
	{ 
		std::vector<float>  NodeFeatureMid;//ÿ��Ҷ�ӵ����ĵ� 
		LeafNodeSet[i] = m_Trees[i]->locateLeafNode(vFeatures);
		//Ҷ�Ӿ�ֵԤ��
		PredictValueOfTree[i] = m_Trees[i]->predict(*LeafNodeSet[i], vFeatures, NodeWeight[i], vResponseIndex);
	    std::vector<int> nodeDataIndex; 
	    //������ڵ���Ϣ
		nodeDataIndex = LeafNodeSet[i]->getNodeDataIndexV();//pWeightedPathNode->calNodeDataIndex(LeafBotherNodeSet[i]);
		
	 
		//ȥ�ظ���---------------------------�Ȳ�ȥ��Ч������ǧ��֮8-9��
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
			 
			for (int j = 0;j < NodeDataFeature[0].size();j++)//��
			{
				float featureMid = 0.0f;
				for (int i = 0; i < NodeDataFeature.size(); i++)//��
				{
					featureMid += NodeDataFeature[i][j];
				}
				NodeFeatureMid.push_back({ featureMid / NodeDataFeature.size() });//ÿ��/ά�ȵľ�ֵ
			}

			treeNodeFeatureMid.push_back({ NodeFeatureMid });//ÿ������Ҷ�����ĵ�  
	 
		 

	
	}



	//����ɭ�֣�����ÿ��������Ҷ�ӵ�Ȩ�أ�������Ȩ��MP��,������Ҷ�ӵ������ϳɵ��������ռ���Ϊn����������Ե㵽ÿ��Ҷ�����ĵ��mp��ͨ��mp����Ȩ��
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
	// NOTES : ����û�бȽ� OOB Error������ԭ��
	//         1���������ɭ��ģ��������һ�£���ô OOB ErrorҲ��һ��

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
		// NOTES : �жϽ����ķ�ʽ��2-stage�ȣ�
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
