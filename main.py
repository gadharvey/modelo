from fastapi import FastAPI, File, UploadFile, Query, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import torch
import cv2
import numpy as np
import base64
import time
import os
from uuid import uuid4

app = FastAPI()

# 🛡️ CORS para permitir conexión con React
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Puedes restringir a ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 📦 Cargar el modelo YOLO de personas
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = YOLO("best5.pt").to(device)
print(f"✅ Modelo cargado en: {model.device}")


def detectar_objetos(frame):
    results = model(frame, imgsz=640, verbose=False)

    boxes = results[0].boxes
    boxes = boxes[boxes.conf >= 0.2]
    results[0].boxes = boxes

    annotated = results[0].plot()

    names = model.names
    conteo_clases = {}

    if boxes and boxes.cls is not None:
        for cls_id in boxes.cls.tolist():
            nombre = names[int(cls_id)]
            conteo_clases[nombre] = conteo_clases.get(nombre, 0) + 1

    return annotated, conteo_clases



@app.get("/")
def root():
    return {"message": "✅ API de detección de personas activa."}


@app.post("/detectar/")
async def detectar(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(status_code=400, detail="Imagen inválida")

    frame, conteo_clases = detectar_objetos(image)

    _, buffer = cv2.imencode(".jpg", frame)
    img_base64 = base64.b64encode(buffer).decode("utf-8")

    return JSONResponse(content={
        "conteo": conteo_clases,
        "image": img_base64
    })



@app.get("/video_feed")
def video_feed(
    source: str = Query(...),
    path: str = Query(None)
):
    return StreamingResponse(
        generate_frames(source, path),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.post("/video_upload/")
async def upload_video(file: UploadFile = File(...)):
    filename = f"{uuid4()}.mp4"
    save_path = os.path.join("videos", filename)
    os.makedirs("videos", exist_ok=True)

    with open(save_path, "wb") as f:
        f.write(await file.read())

    return {"url": f"http://localhost:8000/video_feed?source=video&path={save_path}"}


def generate_frames(source_type: str, path: str = None):
    if source_type == "cam":
        cap = cv2.VideoCapture(0)
    elif source_type == "video":
        if not path:
            raise HTTPException(status_code=400, detail="Falta el path del video")
        cap = cv2.VideoCapture(path)
    else:
        raise HTTPException(status_code=400, detail="source debe ser 'cam' o 'video'")

    if not cap.isOpened():
        raise HTTPException(status_code=500, detail="No se pudo abrir la fuente")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.resize(frame, (640, 480))
        annotated, total = detectar_objetos(frame)

        

        ret, buffer = cv2.imencode('.jpg', annotated)
        frame_bytes = buffer.tobytes()
        time.sleep(0.03)

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n'
        )

    cap.release()
