@echo off
chcp 65001 >nul
cd /d "%~dp0"

if not exist ".venv" (
    echo Создаю виртуальное окружение...
    python -m venv .venv
)

call .venv\Scripts\activate.bat

echo Устанавливаю зависимости...
pip install -r requirements.txt
pip install tensorboard rich

echo Запускаю полный пайплайн...
python run_all.py

pause