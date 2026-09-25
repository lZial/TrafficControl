@echo off
chcp 65001 >nul
cd /d "%~dp0"

if not exist ".venv\Scripts\activate.bat" (
    echo ОШИБКА: не найдено виртуальное окружение .venv
    echo Сначала выполните: python -m venv .venv
    pause
    exit /b 1
)

call .venv\Scripts\activate.bat
python run_all.py
pause