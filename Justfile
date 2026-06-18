# Installiere Abhängigkeiten
setup:
    uv add fastapi uvicorn python-multipart opencv-python numpy pytesseract jinja2

# Starte den Entwicklungs-Server
serve:
    uv run uvicorn server:app --reload --port 8000

# Starte den Container im Hintergrund auf Port 8000 mit persistentem Speicher
run:
    podman run -d \
        --name target-analyzer \
        -p 8000:8000 \
        -v ./static:/app/static:Z \
        --restart unless-stopped \
        targeter

# Räume den statischen Bilder-Cache auf
clean:
    rm -rf target/*.png
