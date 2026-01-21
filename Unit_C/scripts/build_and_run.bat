@echo off
echo ================================================================
echo ConvTranspose2d C Implementation - Build and Run
echo ================================================================
echo.

:menu
echo Choose an option:
echo.
echo 1. Build and run examples (conv_transpose2d_example)
echo 2. Build and run visualizations (conv_transpose2d_visual)
echo 3. Build and run full test suite (test_conv2d)
echo 4. Build all
echo 5. Clean all
echo 6. Exit
echo.
set /p choice="Enter your choice (1-6): "

if "%choice%"=="1" goto examples
if "%choice%"=="2" goto visual
if "%choice%"=="3" goto tests
if "%choice%"=="4" goto buildall
if "%choice%"=="5" goto clean
if "%choice%"=="6" goto end
echo Invalid choice. Please try again.
echo.
goto menu

:examples
echo.
echo ================================================================
echo Building and running examples...
echo ================================================================
gcc -Wall -O2 -std=c99 -c conv2d.c -o conv2d.o
gcc -Wall -O2 -std=c99 -c conv_transpose2d_example.c -o conv_transpose2d_example.o
gcc conv2d.o conv_transpose2d_example.o -o conv_transpose2d_example.exe -lm
if exist conv_transpose2d_example.exe (
    echo.
    echo Build successful! Running...
    echo.
    conv_transpose2d_example.exe
) else (
    echo Build failed!
)
echo.
pause
goto menu

:visual
echo.
echo ================================================================
echo Building and running visualizations...
echo ================================================================
gcc -Wall -O2 -std=c99 -c conv2d.c -o conv2d.o
gcc -Wall -O2 -std=c99 -c conv_transpose2d_visual.c -o conv_transpose2d_visual.o
gcc conv2d.o conv_transpose2d_visual.o -o conv_transpose2d_visual.exe -lm
if exist conv_transpose2d_visual.exe (
    echo.
    echo Build successful! Running...
    echo.
    conv_transpose2d_visual.exe
) else (
    echo Build failed!
)
echo.
pause
goto menu

:tests
echo.
echo ================================================================
echo Building and running full test suite...
echo ================================================================
gcc -Wall -O2 -std=c99 -c conv2d.c -o conv2d.o
gcc -Wall -O2 -std=c99 -c test_conv2d.c -o test_conv2d.o
gcc conv2d.o test_conv2d.o -o test_conv2d.exe -lm
if exist test_conv2d.exe (
    echo.
    echo Build successful! Running...
    echo.
    test_conv2d.exe
) else (
    echo Build failed!
)
echo.
pause
goto menu

:buildall
echo.
echo ================================================================
echo Building all targets...
echo ================================================================
gcc -Wall -O2 -std=c99 -c conv2d.c -o conv2d.o

echo Building examples...
gcc -Wall -O2 -std=c99 -c conv_transpose2d_example.c -o conv_transpose2d_example.o
gcc conv2d.o conv_transpose2d_example.o -o conv_transpose2d_example.exe -lm

echo Building visualizations...
gcc -Wall -O2 -std=c99 -c conv_transpose2d_visual.c -o conv_transpose2d_visual.o
gcc conv2d.o conv_transpose2d_visual.o -o conv_transpose2d_visual.exe -lm

echo Building tests...
gcc -Wall -O2 -std=c99 -c test_conv2d.c -o test_conv2d.o
gcc conv2d.o test_conv2d.o -o test_conv2d.exe -lm

echo.
echo Build complete!
echo.
pause
goto menu

:clean
echo.
echo ================================================================
echo Cleaning all build files...
echo ================================================================
del /Q *.o *.exe 2>nul
echo Clean complete!
echo.
pause
goto menu

:end
echo.
echo Goodbye!
exit /b 0
