@echo off
echo === OLMoE Worker ===
set /p COORDINATOR_HOST=Coordinator IP (e.g. 192.168.1.100): 
set COORDINATOR_PORT=8000
set WORKER_HTTP_PORT=8001
set ZMQ_IN_PORT=5555
set ZMQ_RESULT_PORT=5556
set ZMQ_RECV_TIMEOUT_MS=30000
set SIMULATE_VRAM_GB=4.0
rem MY_IP is auto-detected. Uncomment below only if detection fails:
rem set MY_IP=192.168.1.X
call venv\Scripts\activate.bat
python -m worker.main
pause