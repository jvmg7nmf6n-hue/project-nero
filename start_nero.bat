@echo off
cd /d "%~dp0"
start "" http://localhost:8503/
C:\Users\HP\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe -m streamlit run app.py --server.port 8503
