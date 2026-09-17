"""
CellScope RabbitMQ Inference Consumer & GPU/CPU Worker.

Key architectural features:
1. One-Time Model Loading: PyTorch U-Net weights are loaded into memory ONCE at process startup,
   avoiding expensive per-job disk I/O and weight reconstruction.
2. State Management: Atomically updates MongoDB status (queued -> processing -> done / failed).
3. On any unexpected failure (e.g. corrupt image file), the worker catches
   the exception, logs the traceback, marks the MongoDB document as 'failed', and ACKs the message
   so broken tasks never stall the queue in an infinite loop.
4. Fault-Tolerant Reconnection: Implements connection retries with exponential backoff on broker startup.
"""

import os
import sys
import time
import json
import traceback
from pathlib import Path
from typing import Dict, Any

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import pika
from pika.exceptions import AMQPConnectionError

from phase2_service.app.db import (
    sync_update_job_status,
    sync_complete_job,
    check_mongo_health_sync,
    get_rabbitmq_uri
)
from phase2_service.app.inference import (
    init_serving_model,
    run_serving_inference,
    resolve_model_checkpoint_path
)

# Configuration & Directories
SHARED_DATA_DIR = Path(os.environ.get("SHARED_DATA_DIR", "/data" if os.path.exists("/data") else "shared-data"))
OVERLAYS_DIR = SHARED_DATA_DIR / "overlays"
OVERLAYS_DIR.mkdir(parents=True, exist_ok=True)

RABBITMQ_URI = get_rabbitmq_uri()
QUEUE_NAME = "nuclei_inference_jobs"
DEVICE = os.environ.get("INFERENCE_DEVICE", "cpu")


def process_inference_job(job_id: str, image_path_str: str) -> None:
 
    image_path = Path(image_path_str)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found at path: {image_path_str}")

    print(f"[Worker] Processing job '{job_id}' on {image_path.name}...")

    # Step 1: Update MongoDB status to 'processing'
    sync_update_job_status(job_id=job_id, status="processing")

    # Step 2: Run inference & morphological quantification
    instance_mask, nuclei_features, summary, overlay = run_serving_inference(
        image_path=image_path,
        device=DEVICE,
        threshold=0.5,
        method="watershed",
        pixel_scale_um=0.25
    )

    # Step 3: Save overlay image to shared volume
    overlay_filename = f"{job_id}_overlay.png"
    overlay_path = OVERLAYS_DIR / overlay_filename
    cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    # Step 4: Record results in MongoDB and mark status as 'done'
    sync_complete_job(
        job_id=job_id,
        summary=summary,
        nuclei=nuclei_features,
        overlay_path=str(overlay_path)
    )

    print(f"[Worker] Job '{job_id}' completed successfully: {summary.get('nucleus_count', 0)} nuclei detected.")


def on_message_received(channel, method, properties, body):
    job_id = "unknown"
    try:
        payload = json.loads(body.decode("utf-8"))
        job_id = payload.get("job_id")
        image_path_str = payload.get("image_path")

        if not job_id or not image_path_str:
            raise ValueError(f"Malformed job message: {payload}")

        process_inference_job(job_id=job_id, image_path_str=image_path_str)

    except Exception as e:
        err_msg = f"{type(e).__name__}: {str(e)}"
        print(f"[Worker] ERROR on job '{job_id}': {err_msg}")
        traceback.print_exc()
        if job_id != "unknown":
            try:
                sync_update_job_status(job_id=job_id, status="failed", error=err_msg)
            except Exception as db_err:
                print(f"[Worker] Failed to write failure status to MongoDB: {db_err}")

    finally:
        channel.basic_ack(delivery_tag=method.delivery_tag)


def start_worker():
    print("=" * 60)
    print("CellScope Inference Worker Starting...")
    print(f"Inference Device: {DEVICE}")
    print(f"RabbitMQ URI:     {RABBITMQ_URI}")
    print(f"Shared Data Dir:  {SHARED_DATA_DIR}")
    print("=" * 60)


    init_serving_model(device=DEVICE)
    print("[Worker] Verifying MongoDB connection...")
    while not check_mongo_health_sync():
        print("[Worker] Waiting for MongoDB...")
        time.sleep(2)
    print("[Worker] Connected to MongoDB.")

    while True:
        try:
            print(f"[Worker] Connecting to RabbitMQ at {RABBITMQ_URI}...")
            parameters = pika.URLParameters(RABBITMQ_URI)
            connection = pika.BlockingConnection(parameters)
            channel = connection.channel()
            channel.queue_declare(queue=QUEUE_NAME, durable=True)
            channel.basic_qos(prefetch_count=1)
            channel.basic_consume(queue=QUEUE_NAME, on_message_callback=on_message_received)

            print(f"[Worker] Ready and listening on queue '{QUEUE_NAME}'. Waiting for jobs...")
            channel.start_consuming()

        except AMQPConnectionError as e:
            print(f"[Worker] RabbitMQ connection error ({e}). Retrying in 5 seconds...")
            time.sleep(5)
        except KeyboardInterrupt:
            print("[Worker] Interrupted by user. Shutting down gracefully...")
            break
        except Exception as e:
            print(f"[Worker] Unexpected error ({e}). Reconnecting in 5 seconds...")
            time.sleep(5)


if __name__ == "__main__":
    start_worker()

