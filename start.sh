#!/bin/bash
echo "============================================"
echo "  Agent Identity Authorization System 2.2"
echo "  Starting..."
echo "============================================"
echo ""

[ ! -f ".env" ] && echo "[WARN] .env file not found, using default settings"

pip install -r requirements.txt --quiet 2>/dev/null

echo "[INFO] Starting server at http://127.0.0.1:8000"
echo "[INFO] Press Ctrl+C to stop"
echo ""

python main.py
