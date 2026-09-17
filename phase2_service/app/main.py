"""
FastAPI application for CellScope asynchronous medical image analysis service.

Endpoints:
- POST /jobs          : Upload image file, create queued job, dispatch to RabbitMQ, return job_id
- GET  /jobs/{job_id} : Query job status (queued, processing, done, failed)
- GET  /jobs/{job_id}/result  : Retrieve morphological features & summary once done (409 if processing)
- GET  /jobs/{job_id}/overlay : Retrieve colorized instance overlay image
- GET  /health        : Healthcheck validating MongoDB and RabbitMQ connectivity
"""

import os
import sys
import uuid
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

# Ensure repository root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import aiofiles
except ImportError:
    aiofiles = None

from fastapi import FastAPI, UploadFile, File, HTTPException, status, Request
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

try:
    import pika
except ImportError:
    pika = None

from phase2_service.app.schemas import (
    JobSubmitResponse,
    JobStatusResponse,
    JobResultResponse,
    JobResultSummary,
    NucleusFeature,
    HealthResponse
)
from phase2_service.app.db import (
    init_db_indexes,
    create_job,
    get_job,
    check_mongo_health_async,
    get_rabbitmq_uri
)
from phase2_service.app.inference import is_model_loaded

# Shared volume 
SHARED_DATA_DIR = Path(os.environ.get("SHARED_DATA_DIR", "/data" if os.path.exists("/data") else "shared-data"))
UPLOADS_DIR = SHARED_DATA_DIR / "uploads"
OVERLAYS_DIR = SHARED_DATA_DIR / "overlays"

for d in [UPLOADS_DIR, OVERLAYS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

FRONTEND_INDEX_PATH = Path(__file__).resolve().parent.parent / "frontend" / "index.html"
SAMPLE_IMAGE_PATH = REPO_ROOT / "sample.png"




# RabbitMQ configuration
RABBITMQ_URI = get_rabbitmq_uri()
QUEUE_NAME = "nuclei_inference_jobs"




app = FastAPI(
    title="CellScope Nuclei Segmentation & Quantification API",
    description="Asynchronous medical computer vision service backed by RabbitMQ and MongoDB.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


def publish_to_queue(message: dict) -> None:
    parameters = pika.URLParameters(RABBITMQ_URI)
    connection = pika.BlockingConnection(parameters)
    channel = connection.channel()
    channel.queue_declare(queue=QUEUE_NAME, durable=True)
    channel.basic_publish(
        exchange="",
        routing_key=QUEUE_NAME,
        body=json.dumps(message),
        properties=pika.BasicProperties(
            delivery_mode=2  # make message persistent
        )
    )
    connection.close()


def check_rabbitmq_health() -> bool:
    try:
        parameters = pika.URLParameters(RABBITMQ_URI)
        connection = pika.BlockingConnection(parameters)
        connection.close()
        return True
    except Exception:
        return False


@app.on_event("startup")
async def startup_event():
    try:
        await init_db_indexes()
        print("[API] MongoDB indexes initialized.")
    except Exception as e:
        print(f"[API] Warning: MongoDB index init skipped ({e}).")


@app.get("/", response_class=HTMLResponse, tags=["Frontend"])
@app.get("/ui", response_class=HTMLResponse, tags=["Frontend"])
async def serve_frontend():
    if FRONTEND_INDEX_PATH.exists():
        return FileResponse(FRONTEND_INDEX_PATH, media_type="text/html")
    return HTMLResponse("<h2>CellScope frontend index.html not found.</h2>", status_code=404)



@app.get("/sample.png", tags=["Frontend"])
async def serve_sample_image():
    if SAMPLE_IMAGE_PATH.exists():
        return FileResponse(SAMPLE_IMAGE_PATH, media_type="image/png")
    raise HTTPException(status_code=404, detail="Sample image not found on server.")


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
async def healthcheck():
    mongo_ok = await check_mongo_health_async()
    rabbit_ok = check_rabbitmq_health()
    overall = "healthy" if (mongo_ok and rabbit_ok) else "degraded"

    return HealthResponse(
        status=overall,
        mongodb="connected" if mongo_ok else "disconnected",
        rabbitmq="connected" if rabbit_ok else "disconnected",
        model_loaded=is_model_loaded(),
        timestamp=datetime.now(timezone.utc)
    )


@app.post(
    "/jobs",
    response_model=JobSubmitResponse,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["Inference Jobs"]
)
async def submit_inference_job(file: UploadFile = File(...)):
 
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded.")

    # Validate file extension
    ext = Path(file.filename).suffix.lower()
    if ext not in [".png", ".jpg", ".jpeg", ".tif", ".tiff"]:
        raise HTTPException(status_code=400, detail=f"Unsupported file format: {ext}. Expected PNG, JPG, or TIFF.")

    job_id = str(uuid.uuid4())
    saved_filename = f"{job_id}{ext}"
    target_image_path = UPLOADS_DIR / saved_filename

    # Save uploaded file asynchronously to shared volume
    try:
        if aiofiles:
            async with aiofiles.open(target_image_path, "wb") as out_file:
                while content := await file.read(1024 * 1024):  # 1MB chunks
                    await out_file.write(content)
        else:
            with open(target_image_path, "wb") as out_file:
                while content := await file.read(1024 * 1024):
                    out_file.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to persist image to disk: {e}")

    # Register job document in MongoDB with status 'queued'
    try:
        await create_job(job_id=job_id, image_path=str(target_image_path))
    except Exception as e:
        # Cleanup saved image on DB failure
        if target_image_path.exists():
            target_image_path.unlink()
        raise HTTPException(status_code=500, detail=f"Failed to register job in database: {e}")

    # Publish message to RabbitMQ queue
    try:
        publish_to_queue({"job_id": job_id, "image_path": str(target_image_path)})
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Failed to enqueue job to RabbitMQ message broker: {e}")

    return JobSubmitResponse(job_id=job_id, status="queued")


@app.get(
    "/jobs/{job_id}",
    response_model=JobStatusResponse,
    tags=["Inference Jobs"]
)
async def get_job_status(job_id: str):
    doc = await get_job(job_id)
    if not doc:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return JobStatusResponse(
        job_id=doc["_id"],
        status=doc["status"],
        submitted_at=doc["submitted_at"],
        completed_at=doc.get("completed_at"),
        error=doc.get("error")
    )


@app.get(
    "/jobs/{job_id}/result",
    response_model=JobResultResponse,
    tags=["Inference Jobs"]
)
async def get_job_result(job_id: str, request: Request):
 
    doc = await get_job(job_id)
    if not doc:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")

    current_status = doc.get("status")
    if current_status in ["queued", "processing"]:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Job is currently {current_status}. Please retry once completed."
        )

    if current_status == "failed":
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference job failed: {doc.get('error', 'Unknown execution error')}"
        )

    base_url = str(request.base_url).rstrip("/")
    overlay_url = f"{base_url}/jobs/{job_id}/overlay"

    return JobResultResponse(
        job_id=doc["_id"],
        summary=JobResultSummary(**doc.get("summary", {})),
        nuclei=[NucleusFeature(**n) for n in doc.get("nuclei", [])],
        overlay_image_url=overlay_url
    )


@app.get(
    "/jobs/{job_id}/overlay",
    tags=["Inference Jobs"]
)
async def get_job_overlay(job_id: str):
    doc = await get_job(job_id)
    if not doc:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")

    overlay_path_str = doc.get("overlay_path")
    if not overlay_path_str or not os.path.exists(overlay_path_str):
        raise HTTPException(
            status_code=404,
            detail="Overlay image is not yet available or job did not produce an overlay."
        )

    return FileResponse(
        path=overlay_path_str,
        media_type="image/png",
        filename=f"{job_id}_overlay.png"
    )
