@echo off
echo Choose environment to activate:
echo 1. bisaya-stt-venv (Python 3.11)
echo 2. bisaya-stt-venv310 (Python 3.10)
set /p choice=Enter your choice (1 or 2):

if "%choice%"=="1" (
    call bisaya-stt-venv\Scripts\activate
) else if "%choice%"=="2" (
    call bisaya-stt-venv310\Scripts\activate
) else (
    echo Invalid choice.
)
