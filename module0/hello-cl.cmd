@echo off
setlocal

set OUT=hello_opencl_output.txt
set SRC=hello_world_cl.c
set EXE=hello_cl.exe

> "%OUT%" echo OpenCL HelloWorld Execution Log (Windows / MSVC)
>>"%OUT%" echo ==============================================
>>"%OUT%" echo.

REM Show compiler
>>"%OUT%" echo ^> where cl
where cl >>"%OUT%" 2>&1
>>"%OUT%" echo.

REM Show source code
>>"%OUT%" echo ^> type %SRC%
type "%SRC%" >>"%OUT%" 2>&1
>>"%OUT%" echo.

REM Compile
>>"%OUT%" echo ^> cl /EHsc %SRC%
cl /EHsc "%SRC%" ^
  /I "%USERPROFILE%\vcpkg\installed\x64-windows\include" ^
  "%USERPROFILE%\vcpkg\installed\x64-windows\lib\OpenCL.lib" ^
  /Fe:%EXE% >>"%OUT%" 2>&1
>>"%OUT%" echo.

REM Run
>>"%OUT%" echo ^> %EXE%
"%EXE%" >>"%OUT%" 2>&1
>>"%OUT%" echo.

REM Exit code
>>"%OUT%" echo ^> echo %%ERRORLEVEL%%
echo %ERRORLEVEL% >>"%OUT%" 2>&1
>>"%OUT%" echo.

type "%OUT%"
endlocal

