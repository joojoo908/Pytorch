@echo off
setlocal EnableExtensions EnableDelayedExpansion

rem Usage:
rem   1. Open "x64 Native Tools Command Prompt for VS"
rem   2. cd C:\Cooding\Pytorch\SAC_PathFind_3d\native
rem   3. build_windows_detour_module.bat
rem
rem Optional:
rem   build_windows_detour_module.bat C:\Path\To\python.exe

set "SCRIPT_DIR=%~dp0"
if "%SCRIPT_DIR:~-1%"=="\" set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"
set "RUNTIME_DIR=%SCRIPT_DIR%\runtime"
if not exist "%RUNTIME_DIR%" mkdir "%RUNTIME_DIR%"

if not "%~1"=="" (
  set "PYTHON_LAUNCH="%~1""
) else (
  where py >nul 2>nul
  if not errorlevel 1 (
    set "PYTHON_LAUNCH=py -3.10"
  ) else (
    set "PYTHON_LAUNCH=python"
  )
)

where cl >nul 2>nul
if errorlevel 1 (
  echo [ERROR] cl.exe not found. Run this from "x64 Native Tools Command Prompt for VS".
  exit /b 1
)

set "PY_QUERY_INCLUDE=%TEMP%\codex_detour_query_include_%RANDOM%_%RANDOM%.py"
set "PY_QUERY_PYBIND=%TEMP%\codex_detour_query_pybind_%RANDOM%_%RANDOM%.py"
set "PY_QUERY_EXT=%TEMP%\codex_detour_query_ext_%RANDOM%_%RANDOM%.py"
set "PY_QUERY_LIB=%TEMP%\codex_detour_query_lib_%RANDOM%_%RANDOM%.py"
set "PY_OUT_INCLUDE=%TEMP%\codex_detour_out_include_%RANDOM%_%RANDOM%.txt"
set "PY_OUT_PYBIND=%TEMP%\codex_detour_out_pybind_%RANDOM%_%RANDOM%.txt"
set "PY_OUT_EXT=%TEMP%\codex_detour_out_ext_%RANDOM%_%RANDOM%.txt"
set "PY_OUT_LIB=%TEMP%\codex_detour_out_lib_%RANDOM%_%RANDOM%.txt"

> "%PY_QUERY_INCLUDE%" echo import sysconfig
>> "%PY_QUERY_INCLUDE%" echo print(sysconfig.get_path('include'))

> "%PY_QUERY_PYBIND%" echo import pybind11
>> "%PY_QUERY_PYBIND%" echo print(pybind11.get_include())

> "%PY_QUERY_EXT%" echo import sysconfig
>> "%PY_QUERY_EXT%" echo print(sysconfig.get_config_var('EXT_SUFFIX') or '.pyd')

> "%PY_QUERY_LIB%" echo import os, sys
>> "%PY_QUERY_LIB%" echo print(os.path.join(sys.base_prefix, 'libs', 'python%%d%%d.lib' %% (sys.version_info[0], sys.version_info[1])))

call %PYTHON_LAUNCH% "%PY_QUERY_INCLUDE%" > "%PY_OUT_INCLUDE%"
if errorlevel 1 goto :python_fail
call %PYTHON_LAUNCH% "%PY_QUERY_PYBIND%" > "%PY_OUT_PYBIND%"
if errorlevel 1 goto :python_fail
call %PYTHON_LAUNCH% "%PY_QUERY_EXT%" > "%PY_OUT_EXT%"
if errorlevel 1 goto :python_fail
call %PYTHON_LAUNCH% "%PY_QUERY_LIB%" > "%PY_OUT_LIB%"
if errorlevel 1 goto :python_fail

set /p PYTHON_INCLUDE=<"%PY_OUT_INCLUDE%"
set /p PYBIND11_INCLUDE=<"%PY_OUT_PYBIND%"
set /p EXT_SUFFIX=<"%PY_OUT_EXT%"
set /p PYTHON_LIB=<"%PY_OUT_LIB%"

del /q "%PY_QUERY_INCLUDE%" "%PY_QUERY_PYBIND%" "%PY_QUERY_EXT%" "%PY_QUERY_LIB%" "%PY_OUT_INCLUDE%" "%PY_OUT_PYBIND%" "%PY_OUT_EXT%" "%PY_OUT_LIB%" 2>nul

if not exist "%PYTHON_INCLUDE%\Python.h" (
  echo [ERROR] Python headers not found: %PYTHON_INCLUDE%
  exit /b 1
)
if not exist "%PYBIND11_INCLUDE%\pybind11\pybind11.h" (
  echo [ERROR] pybind11 headers not found: %PYBIND11_INCLUDE%
  exit /b 1
)
if not exist "%PYTHON_LIB%" (
  echo [ERROR] Python import library not found: %PYTHON_LIB%
  exit /b 1
)
if not exist "%SCRIPT_DIR%\Detour.lib" (
  echo [ERROR] Detour.lib not found: %SCRIPT_DIR%\Detour.lib
  exit /b 1
)

set "OUT_PYD=%RUNTIME_DIR%\detour_navmesh_py%EXT_SUFFIX%"

echo [BUILD] Python      = %PYTHON_LAUNCH%
echo [BUILD] Include     = %PYTHON_INCLUDE%
echo [BUILD] pybind11    = %PYBIND11_INCLUDE%
echo [BUILD] Python lib  = %PYTHON_LIB%
echo [BUILD] Detour lib  = %SCRIPT_DIR%\Detour.lib
echo [BUILD] Output      = %OUT_PYD%

cl /nologo /O2 /MD /utf-8 /EHsc /std:c++17 /LD ^
  "%SCRIPT_DIR%\detour_navmesh_wrapper.cpp" ^
  "%SCRIPT_DIR%\python_module.cpp" ^
  /I "%SCRIPT_DIR%" ^
  /I "%PYTHON_INCLUDE%" ^
  /I "%PYBIND11_INCLUDE%" ^
  /link /OUT:"%OUT_PYD%" "%SCRIPT_DIR%\Detour.lib" "%PYTHON_LIB%"

if errorlevel 1 (
  echo [ERROR] Build failed.
  exit /b 1
)

if not exist "%OUT_PYD%" (
  echo [ERROR] Build command finished but output file was not created:
  echo [ERROR]   %OUT_PYD%
  exit /b 1
)

del /q "%SCRIPT_DIR%\*.obj" "%SCRIPT_DIR%\*.exp" 2>nul

echo [OK] Built %OUT_PYD%
echo [INFO] The env will auto-load modules placed under:
echo [INFO]   %RUNTIME_DIR%

endlocal
exit /b 0

:python_fail
echo [ERROR] Failed to run Python command: %PYTHON_LAUNCH%
echo [ERROR] Try running with an explicit Python path, for example:
echo [ERROR]   build_windows_detour_module.bat C:\Path\To\python.exe
del /q "%PY_QUERY_INCLUDE%" "%PY_QUERY_PYBIND%" "%PY_QUERY_EXT%" "%PY_QUERY_LIB%" "%PY_OUT_INCLUDE%" "%PY_OUT_PYBIND%" "%PY_OUT_EXT%" "%PY_OUT_LIB%" 2>nul
exit /b 1
