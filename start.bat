@echo off
chcp 65001 >nul
REM EchoSelf 一键启动（Windows CMD）
REM 双击或在「命令提示符」中运行；优先使用项目内 .venv

cd /d "%~dp0"

if not exist "app.py" (
  echo 错误：当前目录下未找到 app.py。
  exit /b 1
)

if exist ".venv\Scripts\python.exe" (
  " .\.venv\Scripts\python.exe app.py" app.py
  exit /b %ERRORLEVEL%
)

echo 提示：未检测到 .venv\Scripts\python.exe，将尝试系统 Python。
echo       若报错缺少模块，请先按 README「快速开始」创建虚拟环境并安装依赖。
echo.

where py >nul 2>&1
if %ERRORLEVEL%==0 (
  py -3 app.py
  exit /b %ERRORLEVEL%
)

where python >nul 2>&1
if %ERRORLEVEL%==0 (
  python app.py
  exit /b %ERRORLEVEL%
)

echo 错误：未找到 Python。请安装 Python 3.10+ 并勾选「Add to PATH」，或在本目录用 uv 创建 .venv。
exit /b 1
