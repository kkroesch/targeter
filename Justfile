# Installiere Abhängigkeiten
setup:
    uv add fastapi uvicorn python-multipart opencv-python numpy pytesseract jinja2

# Starte den Entwicklungs-Server
serve:
    uv run uvicorn server:app --reload --port 8000

# Räume den statischen Bilder-Cache auf
clean:
    rm -rf target/*.png
