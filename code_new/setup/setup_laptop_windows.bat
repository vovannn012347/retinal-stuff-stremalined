@echo off
rem SurgicalSlicing - laptop setup (Windows, CPU).
rem Uses the existing venv in the repository root (retinal-stuff-streamlined\venv);
rem creates it with Python 3.11 only if it does not exist.
rem Usage:  code_new\setup\setup_laptop_windows.bat

setlocal
set "REPO=%~dp0..\.."
pushd "%REPO%"

if not exist "venv\Scripts\python.exe" (
    echo [setup] venv not found - creating it with Python 3.11
    py -3.11 -m venv venv 2>nul || python -m venv venv
    if errorlevel 1 (
        echo [setup] could not create venv. Install Python 3.10-3.13 from python.org and retry.
        popd & exit /b 1
    )
)

set "PY=venv\Scripts\python.exe"
"%PY%" -m pip install --upgrade pip
if errorlevel 1 goto :fail

rem On Windows the PyPI torch wheel is the CPU build (no CUDA download).
"%PY%" -m pip install --upgrade torch torchvision
if errorlevel 1 goto :fail

"%PY%" -m pip install -r code_new\requirements.txt
if errorlevel 1 goto :fail

"%PY%" code_new\setup\check_env.py
echo.
echo [setup] done. Train with:
echo     venv\Scripts\python.exe code_new\train_surgical.py --config code_new\configs\laptop.yaml
popd
exit /b 0

:fail
echo [setup] FAILED - see the messages above.
popd
exit /b 1
