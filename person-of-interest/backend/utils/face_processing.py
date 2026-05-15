"""Unified face processing utilities for enrollment and runtime.

Ensures embedding consistency between the enrollment pipeline (OpenCV + OpenVINO)
and DLStreamer runtime.  DLStreamer's model-proc for face-reidentification-retail-0095
specifies ``"input_preproc": []`` — meaning default preprocessing:
    1.  Crop face ROI (raw bbox from face-detection-retail-0004)
    2.  Resize to 128×128
    3.  BGR color order (no conversion)
    4.  Float32 (raw pixel values in [0, 255] — NO /255.0 normalisation)
    5.  NCHW layout

**IMPORTANT**: The model expects input in [0, 255] float32 range, NOT [0, 1].
Dividing by 255.0 crushes the model's discriminative power and produces nearly
identical embeddings for all faces (~0.99 cosine similarity).

This module mirrors that chain and adds optional enhancements (padding, squaring)
that are safe to toggle off when strict DLStreamer parity is needed.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np

log = logging.getLogger("poi.utils.face_processing")

# Model input requirements (face-reidentification-retail-0095)
REID_INPUT_SIZE: Tuple[int, int] = (128, 128)
REID_EMBED_DIM: int = 256

# Minimum face dimensions (pixels) — faces below this are too small for reliable
# embeddings.  DLStreamer may produce detections as small as ~20px; those yield
# noisy embeddings that hurt cosine similarity.
MIN_FACE_SIZE: int = 40


def crop_face(
    image: np.ndarray,
    bbox: Tuple[int, int, int, int],
    padding: float = 0.15,
    make_square: bool = True,
) -> np.ndarray:
    """Crop a face region from *image* with optional padding and squaring.

    Args:
        image: BGR uint8 frame (H, W, 3).
        bbox: (x1, y1, x2, y2) pixel coordinates of the face detection.
        padding: Fractional expansion on each side (0.15 = 15%).
        make_square: If True, expand the shorter side to make the crop square
                     before padding.  Prevents aspect-ratio distortion when the
                     crop is later resized to 128×128.

    Returns:
        Cropped BGR uint8 sub-image.
    """
    img_h, img_w = image.shape[:2]
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1

    if make_square:
        # Expand the shorter axis centred on the bbox mid-point.
        side = max(w, h)
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        x1 = int(cx - side / 2)
        y1 = int(cy - side / 2)
        x2 = int(cx + side / 2)
        y2 = int(cy + side / 2)
        w = h = side

    if padding > 0:
        pad_x = int(w * padding)
        pad_y = int(h * padding)
        x1 -= pad_x
        y1 -= pad_y
        x2 += pad_x
        y2 += pad_y

    # Clip to image boundaries
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(img_w, x2)
    y2 = min(img_h, y2)

    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return image[max(0, bbox[1]):min(img_h, bbox[3]),
                     max(0, bbox[0]):min(img_w, bbox[2])]
    return crop


def preprocess_face(
    face_crop: np.ndarray,
    target_size: Tuple[int, int] = REID_INPUT_SIZE,
) -> np.ndarray:
    """Preprocess a face crop for face-reidentification-retail-0095.

    Mirrors DLStreamer's default ``gvainference`` preprocessing when the model-proc
    ``input_preproc`` is empty:
        resize → float32 (raw [0, 255]) → NCHW

    The model expects pixel values in [0, 255] float32 — do NOT divide by 255.
    Dividing by 255 produces degenerate embeddings with ~0.99 cosine similarity
    between any two faces.

    Args:
        face_crop: BGR uint8 image of any size.
        target_size: (width, height) of the model input.  Default (128, 128).

    Returns:
        np.ndarray of shape (1, 3, H, W), dtype float32, in [0, 255].
    """
    resized = cv2.resize(face_crop, target_size, interpolation=cv2.INTER_LINEAR)
    blob = resized.transpose(2, 0, 1).astype(np.float32)
    return blob.reshape(1, 3, target_size[1], target_size[0])


def build_poi_embedding(
    embeddings: List[np.ndarray],
    strategy: str = "mean",
) -> np.ndarray:
    """Consolidate multiple reference embeddings into a single POI vector.

    Args:
        embeddings: List of L2-normalised 256-d vectors (one per reference image).
        strategy: ``"mean"`` averages and re-normalises; ``"all"`` returns stacked.

    Returns:
        Single L2-normalised 256-d vector (for ``"mean"``) or (N, 256) array (for ``"all"``).
    """
    if len(embeddings) == 1:
        vec = np.array(embeddings[0], dtype=np.float32)
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    arr = np.array(embeddings, dtype=np.float32)

    if strategy == "mean":
        mean_vec = arr.mean(axis=0)
        norm = np.linalg.norm(mean_vec)
        if norm > 0:
            mean_vec /= norm
        log.info(
            "Built mean embedding from %d references (norm after L2 = %.6f)",
            len(embeddings), np.linalg.norm(mean_vec),
        )
        return mean_vec

    # "all" — return all vectors (each stored separately in FAISS)
    return arr


def is_face_usable(
    bbox_w: int,
    bbox_h: int,
    confidence: float,
    min_size: int = MIN_FACE_SIZE,
    min_confidence: float = 0.80,
) -> bool:
    """Check whether a detected face meets quality thresholds.

    Args:
        bbox_w: Face bounding box width in pixels.
        bbox_h: Face bounding box height in pixels.
        confidence: Detection confidence [0, 1].
        min_size: Minimum bbox side length in pixels.
        min_confidence: Minimum detection confidence.

    Returns:
        True if face is usable for embedding.
    """
    if confidence < min_confidence:
        return False
    if bbox_w < min_size or bbox_h < min_size:
        log.debug(
            "Face too small: %dx%d (min %d) — skipping",
            bbox_w, bbox_h, min_size,
        )
        return False
    return True


def compute_blur_score(face_crop: np.ndarray) -> float:
    """Compute Laplacian variance as a blur metric.

    Higher values → sharper image.  Typical threshold: ~50–100 for usable faces.
    """
    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def embedding_norm(vector: np.ndarray) -> float:
    """Return L2 norm of an embedding vector."""
    return float(np.linalg.norm(vector))
