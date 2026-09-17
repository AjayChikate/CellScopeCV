"""
Document Schema in 'jobs' collection:
{
  "_id": "job_id (uuid string)",
  "status": "queued" | "processing" | "done" | "failed",
  "submitted_at": datetime,
  "completed_at": datetime or None,
  "image_path": str,
  "error": str or None,
  "summary": dict,
  "nuclei": list of dicts,
  "overlay_path": str or None
}
"""

import os
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List

try:
    from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
    MOTOR_AVAILABLE = True
except ImportError:
    MOTOR_AVAILABLE = False
    AsyncIOMotorClient = None
    AsyncIOMotorDatabase = None

try:
    from pymongo import MongoClient, ASCENDING, DESCENDING
    PYMONGO_AVAILABLE = True
except ImportError:
    PYMONGO_AVAILABLE = False
    MongoClient = None

_async_client: Optional[Any] = None
_async_db: Optional[Any] = None
_sync_client: Optional[Any] = None
_sync_db: Optional[Any] = None


def get_mongo_uri() -> str:
    if "MONGO_URI" in os.environ:
        return os.environ["MONGO_URI"]
    if os.path.exists("/.dockerenv"):
        return "mongodb://mongodb:27017"
    return "mongodb://localhost:27017"


def get_rabbitmq_uri() -> str:
    if "RABBITMQ_URI" in os.environ:
        return os.environ["RABBITMQ_URI"]
    if os.path.exists("/.dockerenv"):
        return "amqp://guest:guest@rabbitmq:5672/"
    return "amqp://guest:guest@localhost:5672/"




def get_db_name() -> str:
    return os.environ.get("MONGO_DB_NAME", "cellscope")




async def get_async_db() -> Any:
    global _async_client, _async_db
    if _async_db is None:
        if not MOTOR_AVAILABLE:
            raise RuntimeError("Motor package is not installed.")
        uri = get_mongo_uri()
        _async_client = AsyncIOMotorClient(uri, serverSelectionTimeoutMS=4000)
        _async_db = _async_client[get_db_name()]
    return _async_db


async def init_db_indexes() -> None:
    db = await get_async_db()
    jobs = db["jobs"]
    await jobs.create_index([("status", 1)])
    await jobs.create_index([("submitted_at", -1)])


async def create_job(job_id: str, image_path: str) -> Dict[str, Any]:
    db = await get_async_db()
    now = datetime.now(timezone.utc)
    doc = {
        "_id": job_id,
        "status": "queued",
        "submitted_at": now,
        "completed_at": None,
        "image_path": str(image_path),
        "error": None,
        "summary": {},
        "nuclei": [],
        "overlay_path": None
    }
    await db["jobs"].insert_one(doc)
    return doc


async def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    db = await get_async_db()
    return await db["jobs"].find_one({"_id": job_id})


async def check_mongo_health_async() -> bool:
    try:
        db = await get_async_db()
        await db.command("ping")
        return True
    except Exception:
        return False





def get_sync_db() -> Any:
    global _sync_client, _sync_db
    if _sync_db is None:
        if not PYMONGO_AVAILABLE:
            raise RuntimeError("PyMongo package is not installed.")
        uri = get_mongo_uri()
        _sync_client = MongoClient(uri, serverSelectionTimeoutMS=4000)
        _sync_db = _sync_client[get_db_name()]
    return _sync_db


def sync_update_job_status(job_id: str, status: str, error: Optional[str] = None) -> None:

    db = get_sync_db()
    update_fields = {"status": status}
    if error is not None:
        update_fields["error"] = str(error)
    if status in ["done", "failed"]:
        update_fields["completed_at"] = datetime.now(timezone.utc)

    db["jobs"].update_one(
        {"_id": job_id},
        {"$set": update_fields}
    )


def sync_complete_job(job_id: str,summary: Dict[str, Any],nuclei: List[Dict[str, Any]],overlay_path: str) -> None:
    db = get_sync_db()
    now = datetime.now(timezone.utc)
    db["jobs"].update_one(
        {"_id": job_id},
        {"$set": {
            "status": "done",
            "completed_at": now,
            "summary": summary,
            "nuclei": nuclei,
            "overlay_path": str(overlay_path),
            "error": None
        }}
    )


def sync_get_job(job_id: str) -> Optional[Dict[str, Any]]:
    db = get_sync_db()
    return db["jobs"].find_one({"_id": job_id})


def check_mongo_health_sync() -> bool:
    try:
        db = get_sync_db()
        db.command("ping")
        return True
    except Exception:
        return False

