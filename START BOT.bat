@echo off
title Trading Bot
echo Stopping any running bot...
taskkill /F /IM python.exe /T >nul 2>&1
timeout /t 3 /nobreak >nul
echo Starting bot...
cd /d "c:\Users\USER\Documents\Telegram bot"
python bot.py
pause
