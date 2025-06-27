
from typing import Iterable, List, Tuple
import cv2
import numpy as np
import torch
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn

# Detection/tracking utilities using a pre-trained Faster R-CNN model from
# TorchVision.  The detector simultaneously推論s人物と自転車を検出し、色ヒストグラムを
# 用いた簡易トラッキングを行う。

def compute_motion_vectors(frames: Iterable[np.ndarray]) -> np.ndarray:
    """Compute simple frame-to-frame motion vectors."""
    vectors = []
    prev_gray = None
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            vectors.append(flow.mean())
        prev_gray = gray
    return np.array(vectors)


def _load_detector(device: torch.device):
    """Load a pretrained Faster R-CNN model for person/bicycle detection."""
    model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
    model.to(device)
    model.eval()
    transform = transforms.ToTensor()
    return model, transform


def _detect_person_bike(
    frame: np.ndarray,
    model: torch.nn.Module,
    transform: transforms.Compose,
    device: torch.device,
) -> Tuple[np.ndarray | None, np.ndarray | None]:
    """Return person and bike bounding boxes as ``[x1, y1, x2, y2]`` arrays."""
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tensor = transform(img).to(device)
    with torch.no_grad():
        output = model([tensor])[0]

    boxes = output["boxes"].cpu().numpy()
    labels = output["labels"].cpu().numpy()
    scores = output["scores"].cpu().numpy()

    person_boxes = boxes[(labels == 1) & (scores > 0.5)]
    bike_boxes = boxes[(labels == 2) & (scores > 0.5)]

    person_box = None
    if len(person_boxes) > 0:
        areas = (person_boxes[:, 2] - person_boxes[:, 0]) * (
            person_boxes[:, 3] - person_boxes[:, 1]
        )
        person_box = person_boxes[np.argmax(areas)].astype(int)

    bike_box = None
    if len(bike_boxes) > 0:
        areas = (bike_boxes[:, 2] - bike_boxes[:, 0]) * (
            bike_boxes[:, 3] - bike_boxes[:, 1]
        )
        bike_box = bike_boxes[np.argmax(areas)].astype(int)

    return person_box, bike_box


def _color_hist(roi: np.ndarray) -> np.ndarray:
    hist = cv2.calcHist([roi], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist


def _pose_angle(person_roi: np.ndarray) -> float:
    """Very rough pose estimate: average angle of edges inside ROI."""
    edges = cv2.Canny(person_roi, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=30, minLineLength=person_roi.shape[0] // 3, maxLineGap=10)
    if lines is None:
        return 0.0
    angles: List[float] = []
    for x1, y1, x2, y2 in lines[:, 0, :]:
        angles.append(np.arctan2(y2 - y1, x2 - x1))
    return float(np.mean(angles)) if angles else 0.0


def compute_frame_features(frames: Iterable[np.ndarray]) -> np.ndarray:
    """Return extended features per frame using detection, tracking and pose."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, transform = _load_detector(device)

    features: List[List[float]] = []
    prev_person_center = None
    prev_bike_center = None
    prev_hist = None

    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        m = cv2.moments(gray)
        if m["m00"] != 0:
            cx = m["m10"] / m["m00"]
            cy = m["m01"] / m["m00"]
        else:
            h, w = gray.shape
            cx, cy = w / 2, h / 2
        center = np.array([cx, cy])

        person_box, bike_box = _detect_person_bike(frame, model, transform, device)

        if person_box is not None:
            px1, py1, px2, py2 = person_box
            pw = px2 - px1
            ph = py2 - py1
            person_center = np.array([px1 + pw / 2.0, py1 + ph / 2.0])
            aspect = pw / ph
            area = pw * ph
            roi = frame[py1:py2, px1:px2]
            angle = _pose_angle(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY))
            hist = _color_hist(roi)
            hist_diff = (
                float(cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA))
                if prev_hist is not None
                else 0.0
            )
            prev_hist = hist
            if prev_person_center is None:
                person_speed = 0.0
            else:
                person_speed = float(np.linalg.norm(person_center - prev_person_center))
            prev_person_center = person_center
        else:
            person_center = np.zeros(2)
            aspect = 0.0
            area = 0.0
            angle = 0.0
            hist_diff = 0.0
            person_speed = 0.0

        if bike_box is not None:
            bx1, by1, bx2, by2 = bike_box
            bw = bx2 - bx1
            bh = by2 - by1
            bike_center = np.array([bx1 + bw / 2.0, by1 + bh / 2.0])
            bike_area = bw * bh
            if prev_bike_center is None:
                bike_speed = 0.0
            else:
                bike_speed = float(np.linalg.norm(bike_center - prev_bike_center))
            prev_bike_center = bike_center
        else:
            bike_center = np.zeros(2)
            bike_area = 0.0
            bike_speed = 0.0

        distance = float(np.linalg.norm(person_center - bike_center))

        features.append(
            [
                cx,
                cy,
                person_speed,
                person_center[0],
                person_center[1],
                aspect,
                area,
                angle,
                bike_center[0],
                bike_center[1],
                bike_area,
                bike_speed,
                distance,
                hist_diff,
            ]
        )

    return np.array(features, dtype=np.float32)
