FROM python:3.11-slim
WORKDIR /app

# libgl1 + libglib2.0-0 : requis par OpenCV (dépendance d'ultralytics/YOLO).
# Sans elles : "libGL.so.1: cannot open shared object file".
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# 1) PyTorch CPU AVANT ultralytics : évite le téléchargement des wheels CUDA (~2 Go)
#    et l'échec de chargement faute de mémoire/GPU.
RUN pip install --no-cache-dir \
        torch==2.3.1 torchvision==0.18.1 \
        --index-url https://download.pytorch.org/whl/cpu
# 2) Le reste des dépendances.
RUN pip install --no-cache-dir -r requirements.txt
# 3) Garde-fou : on force numpy < 2 (TensorFlow 2.16 est incompatible avec numpy 2.x).
RUN pip install --no-cache-dir "numpy==1.26.4"

COPY app ./app
COPY models ./models

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
