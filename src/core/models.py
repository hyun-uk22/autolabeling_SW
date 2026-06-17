from pydantic import BaseModel, Field
from typing import List, Optional

class ClassificationLabel(BaseModel):
    label: str = Field(..., description="Image-level class name")
    confidence: float = Field(default=1.0, description="Model confidence score")

class BoundingBox(BaseModel):
    label: str = Field(..., description="Object class name")
    xmin: float = Field(..., description="Top-left x (normalized 0-1000 or 0-1)")
    ymin: float = Field(..., description="Top-left y (normalized 0-1000 or 0-1)")
    xmax: float = Field(..., description="Bottom-right x (normalized 0-1000 or 0-1)")
    ymax: float = Field(..., description="Bottom-right y (normalized 0-1000 or 0-1)")
    confidence: float = Field(default=1.0, description="Model confidence score")

class Point(BaseModel):
    x: float = Field(..., description="Normalized x coordinate")
    y: float = Field(..., description="Normalized y coordinate")

class PolygonSegment(BaseModel):
    label: str = Field(..., description="Segment class name")
    polygon: List[Point] = Field(default_factory=list, description="Normalized polygon points")
    confidence: float = Field(default=1.0, description="Model confidence score")

class Keypoint(BaseModel):
    name: str = Field(..., description="Keypoint name")
    x: float = Field(..., description="Normalized x coordinate")
    y: float = Field(..., description="Normalized y coordinate")
    visible: bool = Field(default=True, description="Whether the keypoint is visible")
    confidence: float = Field(default=1.0, description="Model confidence score")

class PoseInstance(BaseModel):
    label: str = Field(default="person", description="Pose instance label")
    keypoints: List[Keypoint] = Field(default_factory=list)
    confidence: float = Field(default=1.0, description="Model confidence score")

class TextRegion(BaseModel):
    text: str = Field(..., description="Recognized text")
    xmin: float = Field(..., description="Top-left x")
    ymin: float = Field(..., description="Top-left y")
    xmax: float = Field(..., description="Bottom-right x")
    ymax: float = Field(..., description="Bottom-right y")
    confidence: float = Field(default=1.0, description="Model confidence score")

class TrackInstance(BaseModel):
    track_id: str = Field(..., description="Object track id")
    frame_id: int = Field(default=0, description="Frame number for video labeling")
    label: str = Field(..., description="Tracked object class name")
    xmin: float = Field(..., description="Top-left x")
    ymin: float = Field(..., description="Top-left y")
    xmax: float = Field(..., description="Bottom-right x")
    ymax: float = Field(..., description="Bottom-right y")
    confidence: float = Field(default=1.0, description="Model confidence score")

class DetectionResult(BaseModel):
    task_type: str = Field(default="object_detection")
    classifications: List[ClassificationLabel] = Field(default_factory=list)
    boxes: List[BoundingBox] = Field(default_factory=list)
    segments: List[PolygonSegment] = Field(default_factory=list)
    poses: List[PoseInstance] = Field(default_factory=list)
    texts: List[TextRegion] = Field(default_factory=list)
    tracks: List[TrackInstance] = Field(default_factory=list)
    source_model: Optional[str] = None
    uncertainty_score: Optional[float] = None
    consistency_score: Optional[float] = None
    mean_confidence: Optional[float] = None
