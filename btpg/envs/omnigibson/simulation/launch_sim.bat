@echo off
setlocal enabledelayedexpansion

:: 获取当前文件夹路径
set "current_folder=%~dp0"

set "ini_path=%current_folder%..\global_config.ini"




:: 读取INI文件中的ISAACSIM路径
for /f "tokens=1,2 delims==" %%a in ('type "%ini_path%" ^| findstr /i "isaacsim_path"') do (
    set "isaacsim_path=%%b"
)

echo ISAACSIM_PATH: %isaacsim_path%
cd /d "%isaacsim_path%"


:: 将反斜杠替换为正斜杠
set "current_folder=%current_folder:\=/%"
echo current folder: %current_folder%
set "ini_path=%ini_path:\=/%"
echo ini_path: %ini_path%

:: 启动Isaac Sim
call isaac-sim.bat --/isaac/startup/ros_bridge_extension= --/rtx/ecoMode/enabled=True --ext-folder "%current_folder%" --enable btpg.sim

endlocal

exit /b