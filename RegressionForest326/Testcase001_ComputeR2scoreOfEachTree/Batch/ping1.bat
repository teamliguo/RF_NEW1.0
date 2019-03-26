@echo off
set Num = 3
for %%i in (1,1,%Num%) do (
	ping www.baidu.com
)
pause