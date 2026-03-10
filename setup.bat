@echo off
echo ============================================
echo   TAPS Toolkit Setup
echo   Tap Assessment Protocol Standard v0.1
echo ============================================
echo.

echo [1/3] Installing Python dependencies...
pip install numpy pandas scipy
echo.

echo [2/3] Cloning and installing Tap SDK...
if not exist "tap-python-sdk" (
    git clone https://github.com/TapWithUs/tap-python-sdk
    cd tap-python-sdk
    pip install .
    cd ..
) else (
    echo   Tap SDK already cloned.
)
echo.

echo [3/3] Creating data directories...
if not exist "data\calibration" mkdir data\calibration
echo.

echo ============================================
echo   Setup complete.
echo.
echo   Next steps:
echo   1. Pair Tap Strap 2 in Windows Bluetooth
echo   2. Enable Developer Mode in TapManager app
echo   3. Run: python taps_calibrate.py
echo   4. Run: python taps_logger.py
echo ============================================
pause
