@echo off
REM Start both backend and frontend servers

echo.
echo ============================================
echo OpenGazeLab - Web App
echo ============================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    pause
    exit /b 1
)

REM Check if Node.js is installed
node --version >nul 2>&1
if errorlevel 1 (
    echo Error: Node.js is not installed or not in PATH
    pause
    exit /b 1
)

echo Starting backend server...
start cmd /k "cd backend && python main.py"

timeout /t 3 /nobreak

echo Starting frontend server...
start cmd /k "cd frontend && npm run start"

echo.
echo ============================================
echo Servers starting...
echo Backend: http://127.0.0.1:5000
echo Frontend: http://localhost:8000
echo ============================================
echo.
pause
