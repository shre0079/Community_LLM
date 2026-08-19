@echo off
echo === Distributed OLMoE Setup ===

python -m venv venv
call venv\Scripts\activate

pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

echo.
echo Setup complete. Run workers with:
echo   venv\Scripts\activate
echo   python -m worker.main --coordinator http://COORDINATOR_IP:8000
pause