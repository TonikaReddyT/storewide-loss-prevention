"""Factory pattern implementations."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from backend.core.config import get_config
from backend.utils.face_processing import (
    crop_face,
    embedding_norm,
    preprocess_face,
)

log = logging.getLogger("poi.factory")


class EmbeddingModelFactory:
    """Factory Pattern — Creates embedding model instances based on config.

    Lazy-loads OpenVINO models to generate face embeddings from uploaded images.
    Uses the unified face processing functions from ``face_processing.py`` to
    ensure preprocessing parity with the DLStreamer runtime pipeline.
    """

    _instance: Optional[EmbeddingModelFactory] = None

    def __init__(self) -> None:
        self._cfg = get_config()
        self._det = None
        self._reid = None
        self._core = None

    @classmethod
    def create(cls) -> EmbeddingModelFactory:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _load_models(self) -> None:
        if self._det is not None:
            return
        try:
            from openvino import Core

            self._core = Core()
            log.info("Loading face models on %s...", self._cfg.inference_device)
            self._det = self._core.compile_model(
                self._cfg.det_model, self._cfg.inference_device
            )
            self._reid = self._core.compile_model(
                self._cfg.reid_model, self._cfg.inference_device
            )
            log.info("Face models loaded (no landmark alignment — matches DLStreamer runtime)")
        except Exception:
            log.exception("Failed to load OpenVINO models")
            raise

    def generate_embedding(
        self,
        image: np.ndarray,
        *,
        padding: float = 0.0,
        make_square: bool = False,
        save_crop_path: Optional[str] = None,
    ) -> dict:
        """Generate 256-d face embedding from a BGR image.

        Args:
            image: BGR uint8 image (H, W, 3).
            padding: Fractional bbox expansion (0 = no padding = DLStreamer parity).
                     Use 0.10–0.15 when enrollment images differ from runtime crops.
            make_square: Square the bbox before resize to avoid aspect-ratio distortion.
            save_crop_path: If set, save the preprocessed 128×128 face crop for debugging.

        Returns:
            Dict with keys: embedding, face_bbox, confidence, embedding_norm
            or dict with key: error
        """
        import cv2

        self._load_models()
        img_h, img_w = image.shape[:2]

        # 1. Face detection (face-detection-retail-0004)
        det_input = self._det.input(0)
        det_output = self._det.output(0)
        _, c, h, w = det_input.shape
        blob = cv2.resize(image, (w, h))
        blob = blob.transpose(2, 0, 1).reshape(1, c, h, w).astype(np.float32)
        detections = self._det(blob)[det_output]

        best_face = None
        best_conf = 0.0
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

        # 2. Crop face using unified function
        face_crop = crop_face(image, best_face, padding=padding, make_square=make_square)

        face_w = best_face[2] - best_face[0]
        face_h = best_face[3] - best_face[1]
        log.info(
            "Face detected: bbox=(%d,%d,%d,%d) size=%dx%d conf=%.3f",
            *best_face, face_w, face_h, best_conf,
        )

        # Save debug crop if requested
        if save_crop_path:
            resized_debug = cv2.resize(face_crop, (128, 128))
            cv2.imwrite(save_crop_path, resized_debug)
            log.info("Debug face crop saved to %s", save_crop_path)

        # 3. Preprocess — mirrors DLStreamer's gvainference default:
        #    resize 128×128 → float32 / 255.0 → NCHW
        reid_blob = preprocess_face(face_crop)

        # 4. Inference
        raw_embedding = self._reid(reid_blob)[self._reid.output(0)][0].flatten()
        raw_norm = embedding_norm(raw_embedding)
        embedding = raw_embedding / raw_norm if raw_norm > 0 else raw_embedding
        final_norm = embedding_norm(embedding)

        log.info(
            "Embedding generated: raw_norm=%.6f final_norm=%.6f (should be ~1.0)",
            raw_norm, final_norm,
        )

        return {
            "embedding": embedding.tolist(),
            "face_bbox": list(best_face),
            "confidence": best_conf,
            "embedding_norm": final_norm,
            "face_size": (face_w, face_h),
        }

    def generate_from_bytes(
        self,
        image_bytes: bytes,
        *,
        padding: float = 0.0,
        make_square: bool = False,
        save_crop_path: Optional[str] = None,
    ) -> dict:
        import cv2

        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if image is None:
            return {"error": "Cannot decode image from bytes"}
        log.info("Enrollment image decoded: %dx%d", image.shape[1], image.shape[0])
        return self.generate_embedding(
            image, padding=padding, make_square=make_square, save_crop_path=save_crop_path,
        )

    @classmethod
    def reset(cls) -> None:
        cls._instance = None


class FAISSIndexFactory:
    """Factory Pattern — Creates FAISS index instances."""

    @staticmethod
    def create_flat_ip(dimension: int):
        import faiss

        flat = faiss.IndexFlatIP(dimension)
        return faiss.IndexIDMap(flat)

    @staticmethod
    def create_ivf_pq(dimension: int, nlist: int = 256, m: int = 32):
        import faiss

        quantizer = faiss.IndexFlatIP(dimension)
        index = faiss.IndexIVFPQ(quantizer, dimension, nlist, m, 8)
        return index
