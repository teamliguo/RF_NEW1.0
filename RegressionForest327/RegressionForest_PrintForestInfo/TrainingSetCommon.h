#pragma once
#include <string>

namespace hiveRegressionForest
{
	namespace KEY_WORDS
	{
		const std::string IS_BINARY_TRAINGSETFILE	= "IS_BINARY_TRAINGSETFILE";
		const std::string TRAININGSET_PATH			= "TRAININGSET_PATH";
		const std::string NUM_OF_INSTANCE			= "NUM_OF_INSTANCE";
		const std::string NUM_OF_FEATURE			= "NUM_OF_FEATURE";
		const std::string NUM_OF_RESPONSE			= "NUM_OF_RESPONSE";

		const std::string TESTSET_PATH				= "TESTSET_PATH";

		const std::string IS_NORMALIZE				= "IS_NORMALIZE";

		//divede test file into new files
		const std::string IS_DIVIDE_FILE			= "IS_DIVIDE_FILE";
		const std::string NEW_FILE_DATA_SIZE		= "NEW_FILE_DATA_SIZE";
		const std::string GOOD_TEST_FILE			= "GOOD_TEST_FILE";
		const std::string BAD_TEST_FILE				= "BAD_TEST_FILE";

		//print info
		const std::string IS_PRINT_LEAF_NODE		= "IS_PRINT_LEAF_NODE";
		const std::string PRINT_TREE_NUMBER			= "PRINT_TREE_NUMBER";
		const std::string BEST_TREE_PATH			= "BEST_TREE_PATH";
		const std::string BAD_TREE_PATH				= "BAD_TREE_PATH";
	}
}