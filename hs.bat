@echo off
chcp 65001 >nul
setlocal
title HS Terminal Agent
cd /d "%~dp0"

REM try `python` first, then the `py` launcher
where python >nul 2>nul
if %errorlevel%==0 (
    set "PY=python"
) else (
    set "PY=py -3"
)

%PY% cli.py

echo.
echo Press any key to exit.
pause >nul
endlocal