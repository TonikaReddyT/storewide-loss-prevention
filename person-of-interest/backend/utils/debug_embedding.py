#!/usr/bin/env python3
"""Debug utility to compare face embeddings between enrollment and runtime.

Usage (inside the poi-backend container or with models available):

    # Compare two images
    python -m backend.utils.debug_embedding img1.jpg img2.jpg

    # Compare with padding/squaring (to test enrollment settings)
    python -m backend.utils.debug_embedding img1.jpg img2.jpg --padding 0.15 --square

    # Compare an image against a base64 DLStreamer embedding
    python -m backend.utils.debug_embedding img1.jpg --runtime-b64 <base64_string>

Outputs:
    - Cosine similarity
    - L2 distance
    - Embedding norms
    - Saved preprocessed 128×128 crops for visual comparison
"""

from __future__ import annotations

import argparse
import base64
import logging
import struct
import sys
from pathlib import Path

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")
log = logging.getLogger("poi.debug.embedding")


def compare_embeddings(
    emb1: np.ndarray,
    emb2: np.ndarray,
    label1: str = "img1",
    label2: str = "img2",
) -> dict:
    """Compare two embedding vectors and return detailed metrics.

    Args:
        emb1: First 256-d embedding (L2-normalised or raw).
        emb2: Second 256-d embedding (L2-normalised or raw).
        label1: Display label for first embedding.
        label2: Display label for second embedding.

    Returns:
        Dict with cosine_similarity, l2_distance, norms, and per-dimension stats.
    """
    v1 = np.array(emb1, dtype=np.float32).flatten()
    v2 = np.array(emb2, dtype=np.float32).flatten()

    norm1 = float(np.linalg.norm(v1))
    norm2 = float(np.linalg.norm(v2))

    # Normalise for cosine
    v1_n = v1 / norm1 if norm1 > 0 else v1
    v2_n = v2 / norm2 if norm2 > 0 else v2

    cosine_sim = float(np.dot(v1_n, v2_n))
    l2_dist = float(np.linalg.norm(v1_n - v2_n))

    # Per-dimension difference stats
    diff = np.abs(v1_n - v2_n)
    max_diff_dim = int(np.argmax(diff))

    result = {
        "cosine_similarity": round(cosine_sim, 6),
        "l2_distance": round(l2_dist, 6),
        f"{label1}_norm": round(norm1, 6),
        f"{label2}_norm": round(norm2, 6),
        "dimension": len(v1),
        "max_diff_dimension": max_diff_dim,
        "max_diff_value": round(float(diff[max_diff_dim]), 6),
        "mean_abs_diff": round(float(diff.mean()), 6),
    }

    # Print formatted report
    print("\n" + "=" * 60)
    print("EMBEDDING COMPARISON REPORT")
    print("=" * 60)
    print(f"  {label1} norm:         {norm1:.6f}  {'✓' if abs(norm1 - 1.0) < 0.01 else '⚠ NOT UNIT'}")
    print(f"  {label2} norm:         {norm2:.6f}  {'✓' if abs(norm2 - 1.0) < 0.01 else '⚠ NOT UNIT'}")
    print(f"  Dimension:           {len(v1)}")
    print("-" * 60)
    print(f"  Cosine similarity:   {cosine_sim:.6f}")
    print(f"  L2 distance:         {l2_dist:.6f}")
    print(f"  Mean |diff|:         {float(diff.mean()):.6f}")
    print(f"  Max |diff|:          {float(diff[max_diff_dim]):.6f} (dim {max_diff_dim})")
    print("-" * 60)

    # Interpretation
    if cosine_sim > 0.75:
        verdict = "STRONG MATCH — same person, good conditions"
    elif cosine_sim > 0.60:
        verdict = "LIKELY MATCH — same person, different conditions"
    elif cosine_sim > 0.45:
        verdict = "WEAK — could be same person at bad angle/lighting"
    else:
        verdict = "NO MATCH — likely different people"
    print(f"  Verdict:             {verdict}")
    print("=" * 60 + "\n")

    return result


def _generate_embedding_from_file(
    image_path: str,
    output_dir: str,
    label: str,
    padding: float = 0.0,
    make_square: bool = False,
) -> np.ndarray:
    """Generate embedding from an image file using EmbeddingModelFactory."""
    from backend.factory.factories import EmbeddingModelFactory

    model = EmbeddingModelFactory.create()
    image = cv2.imread(image_path)
    if image is None:
        print(f"ERROR: Cannot read image: {image_path}")
        sys.exit(1)

    print(f"\nProcessing {label}: {image_path} ({image.shape[1]}x{image.shape[0]})")

    crop_path = str(Path(output_dir) / f"{label}_crop128.jpg")
    result = model.generate_embedding(
        image, padding=padding, make_square=make_square, save_crop_path=crop_path,
    )

    if "error" in result:
        print(f"ERROR: {result['error']}")
        sys.exit(1)

    print(f"  Face bbox:    {result['face_bbox']}")
    print(f"  Confidence:   {result['confidence']:.4f}")
    print(f"  Emb norm:     {result['embedding_norm']:.6f}")
    print(f"  Face size:    {result.get('face_size', 'N/A')}")
    print(f"  Crop saved:   {crop_path}")

    return np.array(result["embedding"], dtype=np.float32)


def _decode_runtime_embedding(b64_str: str) -> np.ndarray:
    """Decode a base64-encoded DLStreamer embedding."""
    raw = base64.b64decode(b64_str)
    n = len(raw) // 4
    values = list(struct.unpack(f"{n}f", raw))
    vec = np.array(values, dtype=np.float32)
    norm = np.linalg.norm(vec)
    print(f"\nRuntime embedding: dim={n} raw_norm={norm:.6f}")
    if norm > 0:
        vec = vec / norm
    return vec


def main():
    parser = argparse.ArgumentParser(description="Compare face embeddings")
    parser.add_argument("img1", help="First image (enrollment)")
    parser.add_argument("img2", nargs="?", help="Second image (runtime or enrollment)")
    parser.add_argument("--runtime-b64", help="Base64-encoded runtime embedding (instead of img2)")
    parser.add_argument("--padding", type=float, default=0.0, help="Face crop padding (0.0-0.2)")
    parser.add_argument("--square", action="store_true", help="Make face crop square")
    parser.add_argument("--output-dir", default="/tmp/embedding_debug", help="Dir for debug crops")
    args = parser.parse_args()

    if not args.img2 and not args.runtime_b64:
        parser.error("Provide either img2 or --runtime-b64")

    output_dir = args.output_dir
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Generate embedding for first image
    emb1 = _generate_embedding_from_file(
        args.img1, output_dir, "enrollment",
        padding=args.padding, make_square=args.square,
    )

    # Generate or decode second embedding
    if args.runtime_b64:
        emb2 = _decode_runtime_embedding(args.runtime_b64)
        label2 = "runtime"
    else:
        emb2 = _generate_embedding_from_file(
            args.img2, output_dir, "runtime",
            padding=args.padding, make_square=args.square,
        )
        label2 = "runtime"

    # Compare
    results = compare_embeddings(emb1, emb2, label1="enrollment", label2=label2)

    # Also test with different padding to show the effect
    if args.padding == 0.0 and args.img2:
        print("\n--- Comparison with padding=0.15 + square ---")
        emb1_padded = _generate_embedding_from_file(
            args.img1, output_dir, "enrollment_padded", padding=0.15, make_square=True,
        )
        emb2_padded = _generate_embedding_from_file(
            args.img2, output_dir, "runtime_padded", padding=0.15, make_square=True,
        )
        compare_embeddings(emb1_padded, emb2_padded, "enroll_padded", "runtime_padded")

    return results


if __name__ == "__main__":
    main()
