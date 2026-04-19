@echo off
echo 正在启动日语ASMR翻译工具（调试模式）...
echo 首次运行会自动下载模型，请耐心等待
echo.

REM 检查Python
python --version >nul 2>&1
if errorlevel 1 (
    echo 未找到Python，请先安装Python 3.8+
    pause
    exit /b
)

REM 检查依赖
python -c "import pydub" >nul 2>&1
if errorlevel 1 (
    echo 正在安装依赖...
    pip install -r requirements.txt
)

REM 运行程序（使用 python 而不是 pythonw 以显示调试输出）
python main.py

REM 程序结束后保持窗口打开
echo.
echo 程序已退出，按任意键关闭窗口...
pause >nul