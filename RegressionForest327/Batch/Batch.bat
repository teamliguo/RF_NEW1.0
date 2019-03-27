@echo off
set TIME=1
for /l %%i in (1,1,%TIME%) do (
	Testcase003_InterNodePrediction.exe ./BatchRunConfig.xml ./Config.xml ./TrainingSetConfig.xml
)
pause