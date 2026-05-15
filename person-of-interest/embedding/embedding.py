# Embedding Service — Generate 256-d face embedding from a photo using OpenVINO

import logging

import cv2
import numpy as np
from openvino import Core

from src.config import DET_MODEL, LM_MODEL, REID_MODEL, INFERENCE_DEVICE

log = logging.getLogger("faceid.embedding")


class EmbeddingService:
    """Generate face embeddings from photos using the 3 OpenVINO face models."""

    def __init__(self):
        self.core = Core()
        self._det = None
        self._lm = None
        self._reid = None

    def _load_models(self):
        """Lazy-load models on first use."""
        if self._det is None:
            log.info(f"Loading face models on {INFERENCE_DEVICE}...")
            self._det = self.core.compile_model(DET_MODEL, INFERENCE_DEVICE)
            self._lm = self.core.compile_model(LM_MODEL, INFERENCE_DEVICE)
            self._reid = self.core.compile_model(REID_MODEL, INFERENCE_DEVICE)
            log.info("Face models loaded")

    def generate_from_image(self, image):
        """Generate embedding from a cv2 image (BGR numpy array).

        Args:
            image: BGR numpy array (from cv2.imread or cv2.imdecode)

        Returns:
            dict with keys: embedding (256-d list), face_bbox, confidence
            or dict with key: error
        """
        self._load_models()

        img_h, img_w = image.shape[:2]

        # 1. Face Detection
        det_input = self._det.input(0)
        det_output = self._det.output(0)
        _, c, h, w = det_input.shape

        blob = cv2.resize(image, (w, h))
        blob = blob.transpose(2, 0, 1).reshape(1, c, h, w).astype(np.float32)
        detections = self._det(blob)[det_output]

        # Find best face
        best_face = None
        best_conf = 0
        for det in detections[0][0]:
            conf = det[2]
            if conf > 0.5 and conf > best_conf:
                x1 = max(0, int(det[3] * img_w))
                y1 = max(0, int(det[4] * img_h))
                x2 = min(img_w, int(det[5] * img_w))
                y2 = min(img_h, int(det[6] * img_h))
                if x2 > x1 and y2 > y1:
                    best_face = (x1, y1, x2, y2)
                    best_conf = float(conf)

        if best_face is None:
            return {"error": "No face detected in image"}

        x1, y1, x2, y2 = best_face
        face_crop = image[y1:y2, x1:x2]

        # 2. Landmarks
        lm_input = self._lm.input(0)
        lm_output = self._lm.output(0)
        _, _, lm_h, lm_w = lm_input.shape

        lm_blob = cv2.resize(face_crop, (lm_w, lm_h))
        lm_blob = lm_blob.transpose(2, 0, 1).reshape(1, 3, lm_h, lm_w).astype(np.float32)
        landmarks = self._lm(lm_blob)[lm_output].flatten()

        # 3. Align face
        face_w = x2 - x1
        face_h = y2 - y1
        src_pts = np.array([
            [float(landmarks[0]) * face_w, float(landmarks[1]) * face_h],
            [float(landmarks[2]) * face_w, float(landmarks[3]) * face_h],
            [float(landmarks[4]) * face_w, float(landmarks[5]) * face_h],
        ], dtype=np.float32)

        dst_pts = np.array([
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
        ], dtype=np.float32) * (128.0 / 112.0)

        M = cv2.getAffineTransform(src_pts, dst_pts)
        aligned = cv2.warpAffine(face_crop, M, (128, 128))

        # 4. Re-identification embedding
        reid_blob = aligned.transpose(2, 0, 1).reshape(1, 3, 128, 128).astype(np.float32)
        embedding = self._reid(reid_blob)[self._reid.output(0)][0]
        embedding = embedding / np.linalg.norm(embedding)

        return {
            "embedding": embedding.tolist(),
            "face_bbox": list(best_face),
            "confidence": best_conf,
        }

    def generate_from_file(self, filepath):
        """Generate embedding from an image file path."""
        image = cv2.imread(filepath)
        if image is None:
            return {"error": f"Cannot read image: {filepath}"}
        return self.generate_from_image(image)

    def generate_from_bytes(self, image_bytes):
        """Generate embedding from raw image bytes (e.g. from HTTP upload)."""
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if image is None:
            return {"error": "Cannot decode image from bytes"}
        return self.generate_from_image(image)
