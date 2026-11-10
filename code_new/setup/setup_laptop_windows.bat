@echo off
rem SurgicalSlicing - laptop setup (Windows, CPU).
rem Uses the existing venv in the repository root (.venv, or venv);
rem creates .venv with Python 3.11 only if neither exists.
rem Usage:  code_new\setup\setup_laptop_windows.bat

setlocal
set "REPO=%~dp0..\.."
pushd "%REPO%"

set "VENV=.venv"
if not exist ".venv\Scripts\python.exe" if exist "venv\Scripts\python.exe" set "VENV=venv"
if not exist "%VENV%\Scripts\python.exe" (
    echo [setup] no venv found - creating .venv with Python 3.11
    py -3.11 -m venv .venv 2>nul || python -m venv .venv
    if errorlevel 1 (
        echo [setup] could not create the venv. Install Python 3.10-3.13 from python.org and retry.
        popd & exit /b 1
    )
)

set "PY=%VENV%\Scripts\python.exe"
"%PY%" -m pip install --upgrade pip
if errorlevel 1 goto :fail

rem NVIDIA GPU present -> CUDA 12.6 build (runs on any driver that supports CUDA 12, >= 528);
rem otherwise the PyPI wheel, which on Windows is the CPU build.
where nvidia-smi >nul 2>nul
if not errorlevel 1 (
    echo [setup] NVIDIA GPU found - installing the CUDA build of torch
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    "%PY%" -m pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu126
) else (
    echo [setup] no NVIDIA GPU found - installing the CPU build of torch
    "%PY%" -m pip install --upgrade torch torchvision
)
if errorlevel 1 goto :fail

"%PY%" -m pip install -r code_new\requirements.txt
if errorlevel 1 goto :fail

"%PY%" code_new\setup\check_env.py
echo.
echo [setup] done. Train with:
echo     %PY% code_new\train_surgical.py --config code_new\configs\definitor_laptop.yaml
echo     %PY% code_new\train_surgical.py --config code_new\configs\classifier_laptop.yaml
popd
exit /b 0

:fail
echo [setup] FAILED - see the messages above.
popd
exit /b 1
