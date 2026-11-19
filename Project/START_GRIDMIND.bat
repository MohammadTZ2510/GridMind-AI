@echo off
cd /d "%~dp0"
python web_app\app.py
if errorlevel 1 pause
