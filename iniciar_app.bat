@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

echo ======================================================
echo   Aplicacion - Practica Experimental 1 Mineria de Datos
echo ======================================================
echo.

where py >nul 2>nul
if %errorlevel%==0 (
    set PYTHON=py
) else (
    set PYTHON=python
)

if not exist ".venv\Scripts\python.exe" (
    echo [1/3] Creando entorno virtual...
    %PYTHON% -m venv .venv
)

echo [2/3] Instalando/verificando dependencias...
call .venv\Scripts\python.exe -m pip install --upgrade pip
call .venv\Scripts\python.exe -m pip install -r requirements.txt

echo [3/3] Iniciando aplicacion...
call .venv\Scripts\python.exe app.py

pause
