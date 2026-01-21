@echo off
REM GTCRN C Library Build Script for Windows
REM Supports: Visual Studio, MinGW, or direct cl.exe compilation

echo === GTCRN C Library Build ===
echo.

REM Clean old build
if exist build rmdir /s /q build
mkdir build
cd build

REM Try to detect available build system
where cl >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo Detected: MSVC compiler
    goto :BUILD_MSVC
)

where gcc >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo Detected: GCC/MinGW compiler
    goto :BUILD_MINGW
)

where cmake >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo Detected: CMake, trying Visual Studio generator...
    goto :BUILD_VS
)

echo ERROR: No compiler found!
echo.
echo Please install one of the following:
echo   1. Visual Studio 2019/2022 with C++ workload
echo   2. MinGW-w64 (add to PATH)
echo   3. Or run from Visual Studio Developer Command Prompt
echo.
cd ..
exit /b 1

:BUILD_MSVC
echo Building with MSVC (NMake)...
cmake .. -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release
if %ERRORLEVEL% NEQ 0 goto :CMAKE_FAIL
nmake
goto :BUILD_DONE

:BUILD_MINGW
echo Building with MinGW...
cmake .. -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release
if %ERRORLEVEL% NEQ 0 goto :CMAKE_FAIL
mingw32-make -j4
goto :BUILD_DONE

:BUILD_VS
echo Trying Visual Studio generator...
REM Try VS2022 first, then VS2019
cmake .. -G "Visual Studio 17 2022" -A x64 2>nul
if %ERRORLEVEL% EQU 0 (
    cmake --build . --config Release
    goto :BUILD_DONE
)
cmake .. -G "Visual Studio 16 2019" -A x64 2>nul
if %ERRORLEVEL% EQU 0 (
    cmake --build . --config Release
    goto :BUILD_DONE
)
cmake .. -G "Visual Studio 15 2017" -A x64 2>nul
if %ERRORLEVEL% EQU 0 (
    cmake --build . --config Release
    goto :BUILD_DONE
)
goto :CMAKE_FAIL

:CMAKE_FAIL
echo.
echo CMake configuration failed!
echo.
echo Please run CMake GUI manually and select the correct generator:
echo   - Visual Studio 17 2022 (for VS2022)
echo   - Visual Studio 16 2019 (for VS2019)
echo   - MinGW Makefiles (for MinGW)
echo.
cd ..
exit /b 1

:BUILD_DONE
echo.
echo === Build Complete ===
echo.
if exist Release\gtcrn_demo.exe (
    echo Executable: build\Release\gtcrn_demo.exe
) else if exist gtcrn_demo.exe (
    echo Executable: build\gtcrn_demo.exe
)
echo.
echo Usage:
echo   gtcrn_demo.exe ..\weights\gtcrn_weights.bin input.wav output.wav
cd ..
