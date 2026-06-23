import base64
import io
from pathlib import Path
from typing import Any, Dict, List

import streamlit.components.v1 as components
from PIL import Image


_COMPONENT_DIR = Path(__file__).parent / "components" / "polygon_editor"
_polygon_editor_component = components.declare_component(
    "polygon_vertex_editor",
    path=str(_COMPONENT_DIR),
)


def _image_data_url(image_path: str, width: int, height: int) -> str:
    with Image.open(image_path) as image:
        image = image.convert("RGB").resize((width, height))
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def polygon_vertex_editor(
    image_path: str,
    segments: List[Dict[str, Any]],
    boxes: List[Dict[str, Any]],
    width: int,
    height: int,
    key: str,
) -> Dict[str, Any] | None:
    return _polygon_editor_component(
        imageDataUrl=_image_data_url(image_path, width, height),
        segments=segments,
        boxes=boxes,
        canvasWidth=width,
        canvasHeight=height,
        key=key,
        default=None,
    )
