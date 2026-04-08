#!/bin/sh
set -e

# Lancer FastAPI (background)
uvicorn backend.app:app --host 0.0.0.0 --port 8000 &

# Lancer Streamlit (foreground)
exec streamlit run frontend/streamlit_app.py --server.port 8501 --server.address 0.0.0.0
