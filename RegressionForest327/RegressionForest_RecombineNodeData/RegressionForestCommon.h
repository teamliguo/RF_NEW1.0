#pragma once
#include <string>

#define _LOG_(str) std::cout << str << std::endl;

namespace hiveRegressionForest
{
	namespace KEY_WORDS
	{
		//Basic regression forest key words
		const std::string NUMBER_OF_TREE                   = "NUMBER_OF_TREE";
		const std::string MAX_TREE_DEPTH                   = "MAX_TREE_DEPTH";
		const std::string MAX_LEAF_NODE_INSTANCE_SIZE      = "MAX_LEAF_NODE_INSTANCE_SIZE";
		const std::string NUMBER_OF_RESPONSE			   = "NUMBER_OF_RESPONSE";
		const std::string NUMBER_CANDIDATE_FEATURE		   = "NUMBER_CANDIDATE_FEATURE";

		//Node split method
		const std::string NODE_SPLIT_METHOD				   = "NODE_SPLIT_METHOD";
		const std::string INFORMATION_GAIN_METHOD          = "INFORMATION_GAIN_METHOD";
		const std::string RESIDUAL_SUM_OF_SQUARES_METHOD   = "RESIDUAL_SUM_OF_SQUARES_METHOD";
		const std::string MULTI_INFO_GAIN_METHOD		   = "MULTI_INFO_GAIN_METHOD";

		//Leaf node model
		const std::string LEAF_NODE_MODEL_SIGNATURE		   = "LEAF_NODE_MODEL_SIGNATURE";
		const std::string REGRESSION_MODEL_AVERAGE         = "REGRESSION_MODEL_AVERAGE";
		const std::string REGRESSION_MODEL_LEAST_SQUARES   = "REGRESSION_MODEL_LEAST_SQUARES";

		//Bootstrap Key Words
		const std::string BOOTSTRAP_SELECTOR               = "BOOTSTRAP_SELECTOR";
		const std::string UNIFORM_BOOTSTRAP_SELECTOR       = "UNIFORM_BOOTSTRAP_SELECTOR";
		const std::string WEIGHTED_BOOTSTRAP_SELECTOR      = "WEIGHTED_BOOTSTRAP_SELECTOR";
		const std::string INSTANCE_WEIGHT_CALCULATE_METHOD = "INSTANCE_WEIGHT_CALCULATE_METHOD";
		const std::string RESPONSE_METHOD                  = "RESPONSE_METHOD";

		//Features selected method
		const std::string FEATURE_SELECTOR				   = "FEATURE_SELECTOR";
		const std::string UNIFORM_FEATURE_SELECTOR		   = "UNIFORM_FEATURE_SELECTOR";
		const std::string WEIGHTED_FEATURE_SELECTOR		   = "WEIGHTED_FEATURE_SELECTOR";
		const std::string TWO_STAGE_FEATURE_SELECTOR	   = "TWO_STAGE_FEATURE_SELECTOR";
		const std::string LIVE_UPDATE_FEATURES_WEIGHT      = "LIVE_UPDATE_FEATURES_WEIGHT";
		const std::string FEATURE_WEIGHT_CALCULATE_METHOD  = "FEATURE_WEIGHT_CALCULATE_METHOD";
		const std::string PEARSON_METHOD                   = "PEARSON_METHOD";
		const std::string RSS_METHOD                       = "RSS_METHOD";
		const std::string INVOKE_FEATURES_METHOD		   = "INVOKE_FEATURES_METHOD";
		const std::string VI_FEATURES_METHOD	    	   = "VI_FEATURES_METHOD";
		
		//Leaf node condition
		const std::string LEAF_NODE_CONDITION              = "LEAF_NODE_CONDITION";
		const std::string BASIC_CONDITION                  = "BASIC_CONDITION";
		const std::string EARLY_FITTING_CONDITION          = "EARLY_FITTING_CONDITION";
		const std::string MAX_MSE_FITTING_THRESHOLD		   = "MAX_MSE_FITTING_THRESHOLD";
		
		//OpenMP build tree
		const std::string OPENMP_PARALLEL_BUILD_TREE       = "OPENMP_PARALLEL_BUILD_TREE";

		//Build forest method
		const std::string BUILD_TREE_TYPE				   = "BUILD_TREE_TYPE";
		const std::string TWO_STAGE_TREE				   = "TWO_STAGE_TREE";
		
		//Create node type
		const std::string CREATE_NODE_TYPE				   = "CREATE_NODE_TYPE";
		const std::string SINGLE_RESPONSE_NODE			   = "SINGLE_RESPONSE_NODE";
		const std::string MULTI_RESPONSES_NODE			   = "MULTI_RESPONSES_NODE";

		//Control out dimension
		const std::string OUT_DIMENSION					   = "OUT_DIMENSION";
		
		//Select instance to predict
		const std::string INSTANCE_NUMBER				   = "INSTANCE_NUMBER";
	}
}