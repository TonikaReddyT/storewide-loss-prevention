"""OpenVINO MCP tools.

Provides tools for face embedding generation using Intel OpenVINO models:
face detection (face-detection-retail-0004), landmark regression
(landmarks-regression-retail-0009), and face re-identification
(face-reidentification-retail-0095).

The model pipeline runs locally within the MCP server process. A threading.Lock
guards singleton model creation to prevent race conditions on first call.
"""

from __future__ import annotations

import base64
import logging
import os
import threading
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from mcp_server.config import MCPConfig

log = logging.getLogger("poi.mcp.openvino")

# Module-level lock and lazy-loaded factory — shared across all tool calls
_factory = None
_factory_lock = threading.Lock()


def _get_factory(cfg: MCPConfig):
    """Return the singleton EmbeddingModelFactory, initialising if needed."""
    global _factory
    if _factory is not None:
        return _factory
    with _factory_lock:
        if _factory is None:
            # Override env vars so EmbeddingModelFactory picks up MCP config
            os.environ.setdefault("MODEL_BASE", cfg.model_base)
            os.environ.setdefault("DET_MODEL", cfg.det_model)
            os.environ.setdefault("LM_MODEL", cfg.lm_model)
            os.environ.setdefault("REID_MODEL", cfg.reid_model)
            os.environ.setdefault("INFERENCE_DEVICE", cfg.inference_device)
            try:
                # Import from the POI backend package (PYTHONPATH must include project root)
                from backend.factory.factories import EmbeddingModelFactory
                from backend.core.config import reset_config

                reset_config()  # force reload with updated env
                _factory = EmbeddingModelFactory.create()
                log.info("EmbeddingModelFactory initialised (device=%s)", cfg.inference_device)
            except ImportError:
                log.warning("backend package not importable — using standalone OpenVINO factory")
                _factory = _StandaloneEmbeddingFactory(cfg)
    return _factory


def register(mcp: FastMCP, cfg: MCPConfig) -> None:
    """Register OpenVINO tools on the MCP server."""

    @mcp.tool()
    def openvino_list_devices() -> list[str]:
        """List available OpenVINO inference devices (CPU, GPU, NPU, etc.).

        Returns:
            List of device name strings available on this host.
        """
        try:
            from openvino import Core
            core = Core()
            return core.available_devices
        except ImportError:
            return ["error: openvino package not installed"]
        except Exception as exc:
            return [f"error: {exc}"]

    @mcp.tool()
    def openvino_list_models() -> list[dict]:
        """List OpenVINO IR model files available in the configured model directory.

        Returns:
            List of dicts with model name, path, and precision.
        """
        model_dir = Path(cfg.model_base)
        if not model_dir.exists():
            return [{"error": f"Model directory '{model_dir}' does not exist"}]
        models = []
        for xml_file in sorted(model_dir.rglob("*.xml")):
            models.append(
                {
                    "name": xml_file.stem,
                    "path": str(xml_file),
                    "precision": xml_file.parent.name,
                    "parent_dir": xml_file.parent.parent.name,
                }
            )
        return models

    @mcp.tool()
    def openvino_get_model_info(model_path: str) -> dict:
        """Get input/output shape information for an OpenVINO IR model.

        Args:
            model_path: Absolute path to the model .xml file.

        Returns:
            Dict with model name, inputs, and outputs (name, shape, element_type).
        """
        try:
            from openvino import Core

            if not Path(model_path).exists():
                return {"error": f"Model file not found: {model_path}"}
            core = Core()
            model = core.read_model(model_path)
            inputs = [
                {
                    "name": inp.any_name,
                    "shape": list(inp.shape),
                    "element_type": str(inp.element_type),
                }
                for inp in model.inputs
            ]
            outputs = [
                {
                    "name": out.any_name,
                    "shape": list(out.shape),
                    "element_type": str(out.element_type),
                }
                for out in model.outputs
            ]
            return {"model": Path(model_path).stem, "inputs": inputs, "outputs": outputs}
        except ImportError:
            return {"error": "openvino package not installed"}
        except Exception as exc:
            return {"error": str(exc)}

    @mcp.tool()
    def openvino_generate_face_embedding(image_b64: str) -> dict:
        """Generate a 256-dimensional face re-identification embedding from an image.

        Runs the full POI inference pipeline:
        1. Face detection (face-detection-retail-0004)
        2. Landmark regression (landmarks-regression-retail-0009)
        3. Face alignment
        4. Re-identification embedding (face-reidentification-retail-0095)

        Args:
            image_b64: Base64-encoded image bytes (JPEG or PNG).

        Returns:
            Dict with:
              - embedding: list of 256 floats (L2-normalised)
              - face_bbox: [x1, y1, x2, y2] pixel coordinates
              - confidence: face detection confidence score
            Or dict with 'error' key on failure.
        """
        try:
            image_bytes = base64.b64decode(image_b64)
        except Exception:
            return {"error": "Invalid base64 image data"}
        try:
            factory = _get_factory(cfg)
            result = factory.generate_from_bytes(image_bytes)
            # Exclude raw embedding from return value if it's too large to be useful;
            # return it as a compact list but clip to first 8 values in the summary.
            if "error" in result:
                return result
            return {
                "embedding_dim": len(result.get("embedding", [])),
                "embedding": result.get("embedding"),
                "face_bbox": result.get("face_bbox"),
                "confidence": result.get("confidence"),
            }
        except Exception as exc:
            log.exception("Error generating embedding")
            return {"error": str(exc)}

    @mcp.tool()
    def openvino_benchmark_inference(image_b64: str, iterations: int = 5) -> dict:
        """Benchmark face embedding inference latency.

        Runs the embedding pipeline multiple times and reports min/avg/max latency.

        Args:
            image_b64: Base64-encoded image bytes (JPEG or PNG).
            iterations: Number of inference runs (default 5, max 20).

        Returns:
            Dict with min_ms, avg_ms, max_ms, and iterations count.
        """
        import time

        try:
            image_bytes = base64.b64decode(image_b64)
        except Exception:
            return {"error": "Invalid base64 image data"}

        factory = _get_factory(cfg)
        n = min(iterations, 20)
        latencies = []
        for _ in range(n):
            t0 = time.perf_counter()
            result = factory.generate_from_bytes(image_bytes)
            latencies.append((time.perf_counter() - t0) * 1000)
            if "error" in result:
                return {"error": result["error"]}

        return {
            "iterations": n,
            "min_ms": round(min(latencies), 2),
            "avg_ms": round(sum(latencies) / n, 2),
            "max_ms": round(max(latencies), 2),
            "device": cfg.inference_device,
        }

    log.info("OpenVINO tools registered (device=%s)", cfg.inference_device)


class _StandaloneEmbeddingFactory:
    """Minimal fallback when backend package is not importable.

    Delegates directly to openvino if available.
    """

    def __init__(self, cfg: MCPConfig) -> None:
        self._cfg = cfg
        self._det = None
        self._lm = None
        self._reid = None
        self._load_lock = threading.Lock()

    def _load(self):
        if self._det is not None:
            return
        with self._load_lock:
            if self._det is not None:
                return
            from openvino import Core
            import numpy as np  # noqa: F401 — ensure available

            core = Core()
            self._det = core.compile_model(self._cfg.det_model, self._cfg.inference_device)
            self._lm = core.compile_model(self._cfg.lm_model, self._cfg.inference_device)
            self._reid = core.compile_model(self._cfg.reid_model, self._cfg.inference_device)

    def generate_from_bytes(self, image_bytes: bytes) -> dict:
        import cv2
        import numpy as np

        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if image is None:
            return {"error": "Cannot decode image from bytes"}

        self._load()
        img_h, img_w = image.shape[:2]

        det_input = self._det.input(0)
        _, c, h, w = det_input.shape
        blob = cv2.resize(image, (w, h)).transpose(2, 0, 1).reshape(1, c, h, w).astype(np.float32)
        detections = self._det(blob)[self._det.output(0)]

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

        x1, y1, x2, y2 = best_face
        face_crop = image[y1:y2, x1:x2]

        lm_input = self._lm.input(0)
        _, _, lm_h, lm_w = lm_input.shape
        lm_blob = cv2.resize(face_crop, (lm_w, lm_h)).transpose(2, 0, 1).reshape(1, 3, lm_h, lm_w).astype(np.float32)
        landmarks = self._lm(lm_blob)[self._lm.output(0)].flatten()

        face_w, face_h = x2 - x1, y2 - y1
        src_pts = np.array(
            [
                [float(landmarks[0]) * face_w, float(landmarks[1]) * face_h],
                [float(landmarks[2]) * face_w, float(landmarks[3]) * face_h],
                [float(landmarks[4]) * face_w, float(landmarks[5]) * face_h],
            ],
            dtype=np.float32,
        )
        dst_pts = (
            np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366]], dtype=np.float32)
            * (128.0 / 112.0)
        )
        M = cv2.getAffineTransform(src_pts, dst_pts)
        aligned = cv2.warpAffine(face_crop, M, (128, 128))

        reid_blob = aligned.transpose(2, 0, 1).reshape(1, 3, 128, 128).astype(np.float32)
        embedding = self._reid(reid_blob)[self._reid.output(0)][0]
        embedding = embedding / np.linalg.norm(embedding)

        return {
            "embedding": embedding.tolist(),
            "face_bbox": list(best_face),
            "confidence": best_conf,
        }
