@echo off
echo === OLMoE Coordinator ===
call venv\Scripts\activate.bat
python -m uvicorn coordinator.main:app --host 0.0.0.0 --port 8000
pause