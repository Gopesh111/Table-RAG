#!/usr/bin/env bash
# ==============================================================================
# TableRAG-Core Local Startup Script
# ==============================================================================

set -e

echo "=== Starting TableRAG-Core Local Environment ==="

# 1. Check Python version
python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo "Detected Python version: $python_version"

# 2. Set PYTHONPATH
export PYTHONPATH=.

# 3. Generate sample documents if missing
if [ ! -f "data/sample_docs/techcorp_q3_report.pdf" ]; then
    echo "Generating sample multi-page financial report..."
    python3 data/generate_sample_pdf.py
fi

# 4. Check Ollama connectivity
echo "Verifying local Ollama service..."
if curl -s http://localhost:11434/api/tags > /dev/null; then
    echo "Ollama is online and reachable."
else
    echo "Notice: Ollama is offline or unreachable on :11434. The system will start in fallback retrieval mode."
fi

# 5. Launch FastAPI application
echo "Starting FastAPI ASGI server on http://localhost:8000..."
exec uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
