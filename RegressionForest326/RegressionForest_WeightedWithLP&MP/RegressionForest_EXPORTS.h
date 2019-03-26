#pragma once
#ifndef REGRESSION_FOREST_DLL_EXPORTS
#define REGRESSION_FOREST_EXPORTS __declspec(dllimport)
#else 
#define REGRESSION_FOREST_EXPORTS __declspec(dllexport)
#endif