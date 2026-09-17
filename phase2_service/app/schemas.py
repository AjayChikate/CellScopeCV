from datetime import datetime
from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel, Field


class JobSubmitResponse(BaseModel):
    job_id: str = Field(..., description="Unique UUID for submitted inference job.")
    status: Literal["queued"] = Field(default="queued", description="Initial job status.")


class JobStatusResponse(BaseModel):
    job_id: str
    status: Literal["queued", "processing", "done", "failed"]
    submitted_at: datetime
    completed_at: Optional[datetime] = None
    error: Optional[str] = None


class NucleusFeature(BaseModel):
    id: int = Field(..., description="Unique integer ID of the nucleus instance.")
    area: float = Field(..., description="Area in pixels.")
    area_um2: Optional[float] = Field(None, description="Calibrated physical area in square micrometers.")
    perimeter: float = Field(..., description="Perimeter in pixels.")
    perimeter_um: Optional[float] = Field(None, description="Perimeter in micrometers.")
    circularity: float = Field(..., description="Circularity score (1.0 = perfect circle).")
    eccentricity: float = Field(..., description="Elliptical eccentricity in [0, 1].")
    centroid: List[float] = Field(..., description="[cx, cy] centroid coordinate list.")


class JobResultSummary(BaseModel):
    nucleus_count: int = Field(..., description="Total count of segmented nuclei.")
    density: float = Field(..., description="Nuclei density (nuclei per pixel).")
    density_per_mm2: Optional[float] = Field(None, description="Nuclei density (nuclei per mm^2).")
    mean_area: float = Field(..., description="Mean nucleus area in pixels.")
    std_area: Optional[float] = Field(0.0, description="Standard deviation of nucleus area.")
    mean_circularity: float = Field(..., description="Mean circularity.")
    std_circularity: Optional[float] = Field(0.0, description="Standard deviation of circularity.")
    mean_eccentricity: float = Field(..., description="Mean eccentricity.")
    std_eccentricity: Optional[float] = Field(0.0, description="Standard deviation of eccentricity.")
    pixel_scale_assumed_um: Optional[float] = Field(0.25, description="Assumed physical pixel scale.")
    mean_nearest_neighbor_distance_um: Optional[float] = Field(None, description="Mean nearest-neighbor distance in um.")


class JobResultResponse(BaseModel):
    job_id: str
    summary: JobResultSummary
    nuclei: List[NucleusFeature]
    overlay_image_url: str = Field(..., description="URL endpoint to fetch the colorized overlay image.")


class HealthResponse(BaseModel):
    status: Literal["healthy", "degraded", "unhealthy"]
    mongodb: Literal["connected", "disconnected"]
    rabbitmq: Literal["connected", "disconnected"]
    model_loaded: bool
    timestamp: datetime

