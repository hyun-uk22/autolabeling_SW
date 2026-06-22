import os
from PIL import Image, ImageDraw, ImageFont
from ..core.models import DetectionResult


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def normalized_box_to_pixels(box, width: int, height: int) -> tuple[int, int, int, int]:
    x1 = round(_clamp(box.xmin, 0.0, 1.0) * width)
    y1 = round(_clamp(box.ymin, 0.0, 1.0) * height)
    x2 = round(_clamp(box.xmax, 0.0, 1.0) * width)
    y2 = round(_clamp(box.ymax, 0.0, 1.0) * height)
    return (
        max(0, min(width, min(x1, x2))),
        max(0, min(height, min(y1, y2))),
        max(0, min(width, max(x1, x2))),
        max(0, min(height, max(y1, y2))),
    )


def visualize_boxes(image_path: str, result: DetectionResult, output_dir: str):
    """
    Draws bounding boxes on the image and saves it for paper visualization.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Cannot open image {image_path}: {e}")
        return
        
    draw = ImageDraw.Draw(img)
    width, height = img.size
    
    # Try to load a default font, otherwise use default internal font
    try:
        font = ImageFont.truetype("arial.ttf", size=max(12, int(height*0.02)))
    except IOError:
        font = ImageFont.load_default()

    # Determine color based on model source
    color = "green" if result.source_model and "mini" in result.source_model.lower() else "red"
    if not result.source_model: color = "blue"

    for box in result.boxes:
        x1, y1, x2, y2 = normalized_box_to_pixels(box, width, height)
        
        # Draw box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        
        # Draw label background and text
        text = f"{box.label} ({box.confidence:.2f})"
        
        # Get text bounding box for background
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        
        label_y1 = max(0, y1 - text_h - 4)
        label_y2 = max(text_h + 2, y1) if label_y1 == 0 else y1
        draw.rectangle([x1, label_y1, min(width, x1 + text_w + 4), label_y2], fill=color)
        draw.text((x1 + 2, label_y1 + 2), text, fill="white", font=font)

    for segment in result.segments:
        points = [(point.x * width, point.y * height) for point in segment.polygon]
        if len(points) >= 3:
            draw.polygon(points, outline="yellow")
            draw.line(points + [points[0]], fill="yellow", width=3)
            draw.text(points[0], f"{segment.label} ({segment.confidence:.2f})", fill="yellow", font=font)

    for pose in result.poses:
        visible_points = []
        for point in pose.keypoints:
            if not point.visible:
                continue
            x = point.x * width
            y = point.y * height
            visible_points.append((x, y))
            radius = max(3, int(height * 0.005))
            draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill="cyan")
            draw.text((x + radius + 2, y), point.name, fill="cyan", font=font)
        if visible_points:
            draw.text(visible_points[0], f"{pose.label} ({pose.confidence:.2f})", fill="cyan", font=font)

    for text_region in result.texts:
        x1, y1, x2, y2 = normalized_box_to_pixels(text_region, width, height)
        draw.rectangle([x1, y1, x2, y2], outline="orange", width=3)
        draw.text((x1, y1), text_region.text, fill="orange", font=font)

    for track in result.tracks:
        x1, y1, x2, y2 = normalized_box_to_pixels(track, width, height)
        draw.rectangle([x1, y1, x2, y2], outline="magenta", width=3)
        draw.text((x1, y1), f"{track.track_id}:{track.label}", fill="magenta", font=font)
        
    # Draw Model Source info at top left
    uncertainty = f"{result.uncertainty_score:.2f}" if result.uncertainty_score is not None else "n/a"
    info_text = f"Model: {result.source_model} | Uncert: {uncertainty}"
    draw.rectangle([0, 0, width, max(30, int(height*0.04))], fill="black")
    draw.text((10, 5), info_text, fill="white", font=font)

    # Save
    base_name = os.path.basename(image_path)
    save_path = os.path.join(output_dir, f"vis_{base_name}")
    img.save(save_path)
    return save_path
