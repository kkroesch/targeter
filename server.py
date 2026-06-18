import os
import uuid

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from jinja2 import Template

# Bildverarbeitungs-Logik importieren
from analyzer import (
    detect_and_draw_rings,
    detect_holes_by_color,
    evaluate_and_draw_scores,
    find_and_warp_target,
    read_target_number,
)

# --- Verzeichnisse initialisieren ---
os.makedirs("static", exist_ok=True)

app = FastAPI(title="ISSF Target Analyzer")
app.mount("/static", StaticFiles(directory="static"), name="static")

# --- HTML Templates (Tailwind CSS) ---
INDEX_HTML = """
<!DOCTYPE html>
<html lang="de">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ISSF Analyzer</title>
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-gray-100 min-h-screen flex items-center justify-center p-4">
    <div class="bg-white p-8 rounded-lg shadow-md w-full max-w-md">
        <h1 class="text-2xl font-bold mb-6 text-center text-gray-800">Zielscheibe scannen</h1>

        <!-- Kombinierter Upload / Native Kamera -->
        <form action="/analyze" method="post" enctype="multipart/form-data" class="space-y-4" id="uploadForm">
            <div class="flex items-center justify-center w-full">
                <label for="dropzone-file" class="flex flex-col items-center justify-center w-full h-64 border-2 border-gray-300 border-dashed rounded-lg cursor-pointer bg-gray-50 hover:bg-gray-100 transition-colors">
                    <div class="flex flex-col items-center justify-center pt-5 pb-6">
                        <svg class="w-12 h-12 mb-4 text-gray-500" aria-hidden="true" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 9a2 2 0 012-2h.93a2 2 0 001.664-.89l.812-1.22A2 2 0 0110.07 4h3.86a2 2 0 011.664.89l.812 1.22A2 2 0 0018.07 7H19a2 2 0 012 2v9a2 2 0 01-2 2H5a2 2 0 01-2-2V9z" />
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 13a3 3 0 11-6 0 3 3 0 016 0z" />
                        </svg>
                        <p class="mb-2 text-sm text-gray-500 text-center"><span class="font-semibold">Kamera öffnen</span><br>oder Bild hochladen</p>
                    </div>
                    <!-- capture="environment" erzwingt auf Handys die Rückkamera-App -->
                    <input id="dropzone-file" type="file" name="file" accept="image/*" capture="environment" class="hidden" required onchange="showLoading()" />
                </label>
            </div>
            <button type="submit" id="submit-btn" class="w-full text-white bg-blue-600 hover:bg-blue-700 font-medium rounded-lg text-sm px-5 py-2.5 text-center hidden">Analysieren</button>
        </form>

        <div id="loading-indicator" class="hidden text-center mt-4">
            <p class="text-blue-600 font-medium animate-pulse">Bild wird analysiert...</p>
        </div>
    </div>

    <script>
        // Formular automatisch abschicken, sobald ein Bild gewählt/gemacht wurde
        function showLoading() {
            document.getElementById('uploadForm').style.display = 'none';
            document.getElementById('loading-indicator').classList.remove('hidden');
            document.getElementById('uploadForm').submit();
        }
    </script>
</body>
</html>
"""

RESULT_HTML = """
<!DOCTYPE html>
<html lang="de">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Auswertung - {{ number }}</title>
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-gray-100 min-h-screen p-8">
    <div class="max-w-6xl mx-auto grid grid-cols-1 md:grid-cols-3 gap-8">

        <!-- Linke Seite: Bild -->
        <div class="md:col-span-2 bg-white p-4 rounded-lg shadow-md">
            <h2 class="text-xl font-bold mb-4">Scheibe: {{ number }}</h2>
            <img src="/static/{{ image_filename }}" alt="Analysiertes Ziel" class="w-full h-auto rounded border">
        </div>

        <!-- Rechte Seite: Daten & Formular -->
        <div class="space-y-6">
            <div class="bg-white p-6 rounded-lg shadow-md">
                <h3 class="text-lg font-bold mb-4 text-gray-800">Trefferliste</h3>
                <div class="overflow-x-auto">
                    <table class="w-full text-sm text-left text-gray-500">
                        <thead class="text-xs text-gray-700 uppercase bg-gray-50">
                            <tr>
                                <th scope="col" class="px-4 py-2">Schuss</th>
                                <th scope="col" class="px-4 py-2 text-right">Punkte</th>
                            </tr>
                        </thead>
                        <tbody>
                            {% for score in scores %}
                            <tr class="bg-white border-b">
                                <td class="px-4 py-2 font-medium text-gray-900">{{ loop.index }}</td>
                                <td class="px-4 py-2 text-right font-bold">{{ score.score }}</td>
                            </tr>
                            {% endfor %}
                        </tbody>
                        <tfoot>
                            <tr class="font-bold text-gray-900 text-base bg-gray-50">
                                <td class="px-4 py-3">Summe</td>
                                <td class="px-4 py-3 text-right text-blue-600">{{ total_score }}</td>
                            </tr>
                        </tfoot>
                    </table>
                </div>
            </div>

            <!-- Zuweisung Formular -->
            <div class="bg-white p-6 rounded-lg shadow-md">
                <h3 class="text-lg font-bold mb-4 text-gray-800">Schütze zuordnen</h3>
                <form action="/" method="get">
                    <div class="mb-4">
                        <label class="block mb-2 text-sm font-medium text-gray-900">Name</label>
                        <input type="text" class="bg-gray-50 border border-gray-300 text-gray-900 text-sm rounded-lg focus:ring-blue-500 focus:border-blue-500 block w-full p-2.5" placeholder="Max Muster" required>
                    </div>
                    <button type="button" onclick="alert('Daten gespeichert! (Dummy)')" class="w-full text-white bg-green-600 hover:bg-green-700 font-medium rounded-lg text-sm px-5 py-2.5 text-center">Speichern</button>
                    <a href="/" class="block mt-4 text-center text-sm text-blue-600 hover:underline">Neue Scheibe scannen</a>
                </form>
            </div>
        </div>

    </div>
</body>
</html>
"""


# --- Routen ---
@app.get("/", response_class=HTMLResponse)
async def read_root():
    return HTMLResponse(content=INDEX_HTML)


@app.post("/analyze", response_class=HTMLResponse)
async def analyze_target(file: UploadFile = File(...)):
    try:
        # Bild in den Speicher laden und für OpenCV dekodieren
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        original = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if original is None:
            return HTMLResponse("Fehler beim Dekodieren des Bildes.", status_code=400)

        # Pipeline ausführen
        warped = find_and_warp_target(original)
        number = read_target_number(warped) or "Unbekannt"
        img_with_rings, center, radii = detect_and_draw_rings(warped)
        holes = detect_holes_by_color(warped)
        final_img, scores_data = evaluate_and_draw_scores(
            img_with_rings, holes, center, radii
        )

        # Resultat speichern
        unique_id = uuid.uuid4().hex[:8]
        filename = f"target_{number}_{unique_id}.png"
        cv2.imwrite(os.path.join("static", filename), final_img)

        # Gesamtpunkte berechnen
        total_score = sum(s["score"] for s in scores_data)

        # HTML mit Jinja2 rendern
        template = Template(RESULT_HTML)
        html_content = template.render(
            number=number,
            image_filename=filename,
            scores=scores_data,
            total_score=total_score,
        )

        return HTMLResponse(content=html_content)

    except Exception as e:
        import traceback

        traceback.print_exc()
        return HTMLResponse(
            f"<h3>Fehler in der Analyse:</h3><p>{e}</p><br><a href='/'>Zurück</a>",
            status_code=500,
        )
