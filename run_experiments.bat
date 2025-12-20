@echo off
REM One-command evaluation for HB-BIS (Windows)
REM Runs the complete experimental protocol and writes results to results/
python -m experiments.run_experiments --all
if %errorlevel% neq 0 (
  echo.
  echo Evaluation failed. Check the messages above.
  pause
  exit /b %errorlevel%
)
echo.
echo Done. See results\SUMMARY.md, results\*.csv and results\plots\.
pause
