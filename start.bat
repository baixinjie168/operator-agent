@echo off
setlocal

:: Switch to script directory (project root)
cd /d "%~dp0"

:: Activate virtual environment
if not exist ".venv\Scripts\activate.bat" (
    echo [ERROR] .venv not found. Run 'uv sync' or 'python -m venv .venv && pip install -r requirements.txt' first.
    pause
    exit /b 1
)
call .venv\Scripts\activate.bat

:: Set PYTHONPATH (src directories of all three packages)
set "PYTHONPATH=%CD%\packages\shared\src;%CD%\packages\mcp-server\src;%CD%\packages\agent\src"

:: Clean all .pyc compiled cache
del /s /q *.pyc >nul 2>&1

:: Start FastAPI server
echo ========================================================
echo   operator-agent  starting...
echo   PYTHONPATH = %PYTHONPATH%
echo   URL        = http://127.0.0.1:8000
echo ========================================================
echo.

uvicorn agent.main:create_app --factory --host 127.0.0.1 --port 8000

endlocal
\r