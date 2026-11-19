# BELUM DIUJI: Docker tidak tersedia di lingkungan pengembangan. Lihat docs/panduan_deployment.md.
# Untuk host yang menjalankan Docker (mis. Hugging Face Docker Space, Render, VPS sendiri).
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HF_HOME=/tmp/hf ARTIFACTS_DIR=artifacts_real
WORKDIR /app

# PyTorch versi CPU (jauh lebih kecil daripada paket default yang menarik CUDA). Indeks resmi PyTorch;
# ketersediaan wheel untuk versi terbaru harus diverifikasi saat build.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
# Unduh model e5 saat build supaya tidak diunduh pada permintaan pertama pengunjung.
RUN python -c "from sentence_transformers import SentenceTransformer as S; S('intfloat/multilingual-e5-small')"

# Hugging Face Space memakai port 7860 secara default (atur app_port di README Space bila berbeda).
EXPOSE 7860
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.headless=true"]
