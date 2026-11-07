import collections
from contextlib import asynccontextmanager
from typing import Deque, List, Optional
import cv2
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
import numpy as np
from pydantic import BaseModel

from engine import SortingInferenceEngine
from stream import ThreadedCamera

class SortingTelemetry(BaseModel):
    status: str
    camera_active: bool
    frame_id: int
    rolling_fps: float
    p95_latency_ms: float
    last_action: Optional[str] = None

class ActuatorTriggerResponse(BaseModel):
    frame_id: int
    material_class: str
    confidence: float
    bbox_xyxy: List[int] # <-- Change this line
    inference_latency_ms: float
    actuator_trigger_delay_ms: float
    action: str

class ServiceContext:
    camera: Optional[ThreadedCamera] = None
    engine: Optional[SortingInferenceEngine] = None
    latencies: Deque[float] = collections.deque(maxlen=100)
    last_trigger: Optional[dict] = None

ctx = ServiceContext()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initializes Mac Camera and ONNX session on boot
    ctx.camera = ThreadedCamera(src=0, target_fps=30).start()
    ctx.engine = SortingInferenceEngine()
    yield
    if ctx.camera:
        ctx.camera.stop()

app = FastAPI(title="Abnoba Systems - Mac Vision Prototype", lifespan=lifespan)

@app.get("/health", response_model=SortingTelemetry)
async def health_check():
    if not ctx.latencies:
        return SortingTelemetry(status="waiting", camera_active=True, frame_id=0, rolling_fps=0.0, p95_latency_ms=0.0)
    
    avg_latency = float(np.mean(ctx.latencies))
    avg_fps = round(1000.0 / avg_latency, 1) if avg_latency > 0 else 0.0
    p95 = round(float(np.percentile(ctx.latencies, 95)), 2)

    return SortingTelemetry(
        status="healthy",
        camera_active=ctx.camera.running if ctx.camera else False,
        frame_id=ctx.camera.frame_id if ctx.camera else 0,
        rolling_fps=avg_fps,
        p95_latency_ms=p95,
        last_action=ctx.last_trigger.get("action") if ctx.last_trigger else None
    )

@app.post("/api/v1/trigger", response_model=ActuatorTriggerResponse)
async def trigger_actuator_cycle():
    if not ctx.camera or not ctx.engine:
        raise HTTPException(status_code=503, detail="Service uninitialized")

    frame_id, frame = ctx.camera.read_latest()
    if frame is None:
        raise HTTPException(status_code=500, detail="Camera frame buffer unavailable")

    result, latency = ctx.engine.infer(frame)
    ctx.latencies.append(latency)
    ctx.last_trigger = result

    return ActuatorTriggerResponse(frame_id=frame_id, **result)

def generate_mjpeg_stream():
    while True:
        if not ctx.camera or not ctx.engine:
            continue

        _, frame = ctx.camera.read_latest()
        if frame is None:
            continue

        result, _ = ctx.engine.infer(frame)

        action = result["action"]
        color = (0, 255, 0) if "AIR_BLAST" in action else (0, 0, 255)
        
        # --- NEW CODE: Draw the actual bounding box! ---
        if result["confidence"] > 0.40:
            x1, y1, x2, y2 = result["bbox_xyxy"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4)
        # -----------------------------------------------

        text = f"{result['material_class']} ({result['confidence']:.2f}) -> {action}"
        cv2.putText(frame, text, (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(
            frame,
            f"Mac Inference: {result['inference_latency_ms']}ms",
            (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2
        )

        _, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + encoded.tobytes() + b"\r\n")

@app.get("/stream/live")
async def live_stream():
    return StreamingResponse(
        generate_mjpeg_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)