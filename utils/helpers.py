import cv2
import numpy as np

def yolo_to_bbox(x_center, y_center, width, height, img_w, img_h):
    """Converts normalized YOLO format coordinates to pixel-level bounding box (x_min, y_min, x_max, y_max)."""
    x1 = int((x_center - width / 2.0) * img_w)
    y1 = int((y_center - height / 2.0) * img_h)
    x2 = int((x_center + width / 2.0) * img_w)
    y2 = int((y_center + height / 2.0) * img_h)
    
    return max(0, x1), max(0, y1), min(img_w, x2), min(img_h, y2)

def bbox_to_yolo(x1, y1, x2, y2, img_w, img_h):
    """Converts pixel bounding box (x_min, y_min, x_max, y_max) to normalized YOLO format (x_center, y_center, width, height)."""
    box_w = float(x2 - x1)
    box_h = float(y2 - y1)
    x_center = float(x1) + box_w / 2.0
    y_center = float(y1) + box_h / 2.0
    
    return [x_center / img_w, y_center / img_h, box_w / img_w, box_h / img_h]

def resize_letterbox(image, target_size=(224, 224), border_color=(128, 128, 128)):
    """Resizes image keeping aspect ratio intact by adding constant color borders (letterboxing)."""
    h, w = image.shape[:2]
    tw, th = target_size
    
    scale = min(tw / w, th / h)
    nw, nh = int(w * scale), int(h * scale)
    
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    
    # Create background canvas
    canvas = np.full((th, tw, 3), border_color, dtype=np.uint8)
    
    # Calculate offset
    dx = (tw - nw) // 2
    dy = (th - nh) // 2
    
    # Paste resized image onto canvas center
    canvas[dy:dy+nh, dx:dx+nw] = resized
    return canvas, scale, (dx, dy)
