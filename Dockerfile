FROM python:3.12-slim

# System-Abhängigkeiten für Tesseract installieren
# (OpenCV-Abhängigkeiten entfallen durch die headless-Version)
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# uv Binary direkt aus dem offiziellen Image kopieren
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Abhängigkeiten systemweit über uv installieren
RUN uv pip install --system \
    fastapi \
    uvicorn \
    python-multipart \
    opencv-python-headless \
    numpy \
    pytesseract \
    jinja2

# Quellcode kopieren
COPY main.py analyzer.py server.py ./

# Ordner für statische Bilder anlegen
RUN mkdir -p static

EXPOSE 8000

# Server starten
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
