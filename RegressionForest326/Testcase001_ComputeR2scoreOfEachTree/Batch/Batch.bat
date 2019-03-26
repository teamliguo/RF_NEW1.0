@echo off
set Num=10
for %%i in (1,1,%Num%) do (
	start Testcase001_ComputeR2scoreOfEachTree.exe ./BatchRunConfig.xml ./Config.xml ./TrainingSetConfig.xml
)
pause