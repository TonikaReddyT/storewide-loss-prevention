"""Deep Learning MCP tools.

Provides framework-agnostic utilities for inspecting deep learning models
and environments: framework version detection, ONNX model inspection,
PyTorch checkpoint inspection, and ONNX inference benchmarking.

Model path access is restricted to the configured model base directory
(MODEL_BASE env var, default /models/intel) unless MCP_DL_ALLOW_ALL_PATHS
is set to true.

Security note: PyTorch checkpoints are loaded with weights_only=True to
prevent arbitrary code execution via pickle. Legacy checkpoints that require
full pickle loading are rejected unless MCP_ALLOW_MUTATIONS=true (which acts
as an explicit acknowledgement of the risk).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from mcp.server.fastmcp import FastMCP

from mcp_server.config import MCPConfig

log = logging.getLogger("poi.mcp.deep_learning")


def _guard_path(path_str: str, cfg: MCPConfig) -> Optional[dict]:
    """Return an error dict if *path_str* is outside the allowed model directory."""
    if cfg.dl_allow_all_paths:
        return None
    try:
        resolved = Path(path_str).resolve()
        allowed = Path(cfg.model_base).resolve()
        resolved.relative_to(allowed)  # raises ValueError if outside
        return None
    except ValueError:
        return {
            "error": (
                f"Path '{path_str}' is outside the allowed model directory '{cfg.model_base}'. "
                "Set MCP_DL_ALLOW_ALL_PATHS=true to allow arbitrary paths."
            )
        }
    except Exception as exc:
        return {"error": str(exc)}


def register(mcp: FastMCP, cfg: MCPConfig) -> None:
    """Register deep learning utility tools on the MCP server."""

    @mcp.tool()
    def dl_get_framework_versions() -> dict:
        """Detect installed deep learning framework versions in this environment.

        Checks for PyTorch, TensorFlow, ONNX Runtime, OpenVINO, Keras,
        JAX, and Transformers.

        Returns:
            Dict mapping framework name to version string or 'not installed'.
        """
        frameworks: dict[str, str] = {}

        def _try_import(module: str, attr: str = "__version__") -> str:
            try:
                import importlib
                mod = importlib.import_module(module)
                return str(getattr(mod, attr, "installed (version unknown)"))
            except ImportError:
                return "not installed"
            except Exception as exc:
                return f"error: {exc}"

        frameworks["torch"] = _try_import("torch")
        frameworks["torchvision"] = _try_import("torchvision")
        frameworks["torchaudio"] = _try_import("torchaudio")
        frameworks["tensorflow"] = _try_import("tensorflow")
        frameworks["keras"] = _try_import("keras")
        frameworks["jax"] = _try_import("jax")
        frameworks["onnx"] = _try_import("onnx")
        frameworks["onnxruntime"] = _try_import("onnxruntime")
        frameworks["openvino"] = _try_import("openvino")
        frameworks["transformers"] = _try_import("transformers")
        frameworks["diffusers"] = _try_import("diffusers")
        frameworks["timm"] = _try_import("timm")
        frameworks["ultralytics"] = _try_import("ultralytics")
        frameworks["numpy"] = _try_import("numpy")
        frameworks["opencv"] = _try_import("cv2", "__version__")
        return frameworks

    @mcp.tool()
    def dl_get_onnx_model_info(model_path: str) -> dict:
        """Inspect an ONNX model file and return its structure.

        Reports graph inputs/outputs (name, shape, dtype), IR version,
        opset version, node count, and initializer count.

        Args:
            model_path: Absolute path to the .onnx file.

        Returns:
            Dict with ir_version, opset_version, node_count,
            initializer_count, inputs, and outputs.
        """
        err = _guard_path(model_path, cfg)
        if err:
            return err
        p = Path(model_path)
        if not p.exists():
            return {"error": f"File not found: {model_path}"}
        if p.suffix.lower() != ".onnx":
            return {"error": f"Expected an .onnx file, got: {p.suffix}"}
        try:
            import onnx
        except ImportError:
            return {"error": "onnx package not installed. Run: pip install onnx"}
        try:
            model = onnx.load(str(p))
            graph = model.graph

            def _shape(tensor_type) -> list:
                try:
                    return [
                        d.dim_value if d.dim_value > 0 else f"dyn:{d.dim_param or '?'}"
                        for d in tensor_type.shape.dim
                    ]
                except Exception:
                    return []

            def _dtype(elem_type: int) -> str:
                mapping = {
                    1: "float32", 2: "uint8", 3: "int8", 4: "uint16",
                    5: "int16", 6: "int32", 7: "int64", 8: "string",
                    9: "bool", 10: "float16", 11: "float64",
                    12: "uint32", 13: "uint64",
                }
                return mapping.get(elem_type, f"type_{elem_type}")

            opset = next(
                (op.version for op in model.opset_import if op.domain in ("", "ai.onnx")),
                None,
            )
            inputs = [
                {
                    "name": inp.name,
                    "shape": _shape(inp.type.tensor_type),
                    "dtype": _dtype(inp.type.tensor_type.elem_type),
                }
                for inp in graph.input
                if inp.name not in {init.name for init in graph.initializer}
            ]
            outputs = [
                {
                    "name": out.name,
                    "shape": _shape(out.type.tensor_type),
                    "dtype": _dtype(out.type.tensor_type.elem_type),
                }
                for out in graph.output
            ]
            return {
                "model_name": p.stem,
                "path": str(p),
                "ir_version": model.ir_version,
                "opset_version": opset,
                "node_count": len(graph.node),
                "initializer_count": len(graph.initializer),
                "inputs": inputs,
                "outputs": outputs,
            }
        except Exception as exc:
            return {"error": str(exc)}

    @mcp.tool()
    def dl_get_torch_model_info(model_path: str) -> dict:
        """Inspect a PyTorch checkpoint (.pt / .pth) and return its structure.

        Reports top-level state_dict keys, total parameter count, and
        estimated size. Loaded with weights_only=True for safety — checkpoints
        that require full pickle loading are rejected.

        SECURITY NOTE: weights_only=True is enforced to prevent arbitrary code
        execution via malicious pickle payloads. Do not disable this.

        Args:
            model_path: Absolute path to the .pt or .pth file.

        Returns:
            Dict with keys, parameter_count, size_mb, and torch_version.
        """
        err = _guard_path(model_path, cfg)
        if err:
            return err
        p = Path(model_path)
        if not p.exists():
            return {"error": f"File not found: {model_path}"}
        if p.suffix.lower() not in (".pt", ".pth"):
            return {"error": f"Expected a .pt or .pth file, got: {p.suffix}"}
        try:
            import torch
        except ImportError:
            return {"error": "torch package not installed. Run: pip install torch"}
        try:
            checkpoint = torch.load(str(p), map_location="cpu", weights_only=True)
        except Exception as exc:
            return {
                "error": (
                    f"Failed to load checkpoint with weights_only=True: {exc}. "
                    "This checkpoint may require legacy pickle loading which is blocked for security."
                )
            }
        try:
            # Unwrap common checkpoint formats
            if isinstance(checkpoint, dict):
                state_dict = (
                    checkpoint.get("state_dict")
                    or checkpoint.get("model_state_dict")
                    or checkpoint.get("model")
                    or checkpoint
                )
            else:
                state_dict = checkpoint

            keys = list(state_dict.keys()) if hasattr(state_dict, "keys") else []
            param_count = 0
            for v in state_dict.values():
                if hasattr(v, "numel"):
                    param_count += v.numel()

            size_mb = round(p.stat().st_size / 1_048_576, 2)
            return {
                "model_name": p.stem,
                "path": str(p),
                "top_level_keys": keys[:100],  # cap to avoid huge responses
                "total_keys": len(keys),
                "parameter_count": param_count,
                "size_mb": size_mb,
                "torch_version": torch.__version__,
                "loaded_with_weights_only": True,
            }
        except Exception as exc:
            return {"error": str(exc)}

    @mcp.tool()
    def dl_benchmark_onnx_model(
        model_path: str,
        iterations: int = 10,
        input_shapes: Optional[dict] = None,
    ) -> dict:
        """Benchmark ONNX Runtime inference latency on a model.

        Generates random input tensors (or uses provided shapes/dtypes) and
        runs the model for the requested number of iterations.

        Args:
            model_path: Absolute path to the .onnx file.
            iterations: Number of inference runs (1–50, default 10).
            input_shapes: Optional dict mapping input name to shape list,
                e.g. {"input": [1, 3, 224, 224]}. Required for dynamic-shape
                models. If omitted, static shapes from the model graph are used.

        Returns:
            Dict with min_ms, avg_ms, max_ms, iterations, provider (CPU/GPU),
            and per-input shape info used.
        """
        err = _guard_path(model_path, cfg)
        if err:
            return err
        p = Path(model_path)
        if not p.exists():
            return {"error": f"File not found: {model_path}"}
        if p.suffix.lower() != ".onnx":
            return {"error": f"Expected an .onnx file, got: {p.suffix}"}

        try:
            import onnxruntime as ort
            import numpy as np
        except ImportError as exc:
            return {"error": f"Required package not installed: {exc}. Run: pip install onnxruntime numpy"}

        import time

        try:
            sess_opts = ort.SessionOptions()
            sess_opts.log_severity_level = 3  # suppress verbose output
            session = ort.InferenceSession(str(p), sess_opts)
            provider = session.get_providers()[0] if session.get_providers() else "unknown"
            input_meta = session.get_inputs()

            # Build feed dict from static shapes or provided overrides
            feed: dict = {}
            shape_used: dict = {}
            for inp in input_meta:
                shape = list(inp.shape)
                if input_shapes and inp.name in input_shapes:
                    shape = list(input_shapes[inp.name])
                else:
                    # Replace dynamic dims (0, None, negative, symbolic strings) with 1
                    shape = [d if isinstance(d, int) and d > 0 else 1 for d in shape]
                if not shape:
                    return {
                        "error": (
                            f"Input '{inp.name}' has no shape info and no override was provided. "
                            "Pass input_shapes={{'{inp.name}': [...]}} to specify the shape."
                        )
                    }
                shape_used[inp.name] = shape
                # Map ONNX dtype string to numpy dtype
                dtype_map = {
                    "tensor(float)": np.float32,
                    "tensor(float16)": np.float16,
                    "tensor(double)": np.float64,
                    "tensor(int32)": np.int32,
                    "tensor(int64)": np.int64,
                    "tensor(uint8)": np.uint8,
                    "tensor(int8)": np.int8,
                    "tensor(bool)": bool,
                }
                np_dtype = dtype_map.get(inp.type, np.float32)
                feed[inp.name] = np.random.rand(*shape).astype(np_dtype)

            n = max(1, min(iterations, 50))
            latencies = []
            for _ in range(n):
                t0 = time.perf_counter()
                session.run(None, feed)
                latencies.append((time.perf_counter() - t0) * 1000)

            return {
                "model_name": p.stem,
                "iterations": n,
                "min_ms": round(min(latencies), 2),
                "avg_ms": round(sum(latencies) / n, 2),
                "max_ms": round(max(latencies), 2),
                "provider": provider,
                "input_shapes_used": shape_used,
                "note": "Random inputs were used — latency reflects compute time, not accuracy.",
            }
        except Exception as exc:
            return {"error": str(exc)}

    log.info(
        "Deep learning tools registered (model_base=%s, allow_all_paths=%s)",
        cfg.model_base,
        cfg.dl_allow_all_paths,
    )
