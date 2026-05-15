#!/usr/bin/env python3
"""Analyze MQTT dump: correlate scene UUIDs with camera embeddings.

Run mqtt_debug.py first to capture data, then run this to analyze:
    python3 tools/analyze_reid.py --dump-dir tools/mqtt_dump

Checks:
  1. Does the same person get the same UUID across cameras?
  2. Where are person-reid and face-reid embeddings?
  3. Correlates scene UUIDs ↔ camera embeddings by timestamp
"""

import argparse
import base64
import json
import struct
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def decode_embedding(b64_str: str) -> list[float]:
    raw = base64.b64decode(b64_str)
    n = len(raw) // 4
    return list(struct.unpack(f"{n}f", raw))


def cosine_similarity(a: list[float], b: list[float]) -> float:
    a, b = np.array(a), np.array(b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return 0.0
    return float(np.dot(a, b) / norm)


def load_camera_data(dump_dir: Path):
    """Load camera messages, extract timestamps and embeddings."""
    camera_data = {}  # camera_name -> list of frame dicts

    camera_dir = dump_dir / "camera"
    if not camera_dir.exists():
        return camera_data

    for cam_dir in sorted(camera_dir.iterdir()):
        if not cam_dir.is_dir():
            continue
        cam_name = cam_dir.name
        frames = []
        for f in sorted(cam_dir.glob("*.json")):
            d = json.load(open(f))
            ts = d.get("timestamp", "")
            for p in d.get("objects", {}).get("person", []):
                frame = {
                    "file": f.name,
                    "timestamp": ts,
                    "camera": cam_name,
                    "person_id": p.get("id"),
                    "person_reid": None,
                    "face_reid": None,
                    "face_confidence": None,
                }
                # Person-level reid
                reid = p.get("metadata", {}).get("reid")
                if reid and reid.get("embedding_vector"):
                    frame["person_reid"] = decode_embedding(
                        reid["embedding_vector"]
                    )
                    frame["person_reid_model"] = reid.get("model_name", "")

                # Face sub_object reid
                for face in p.get("sub_objects", {}).get("face", []):
                    face_reid = face.get("metadata", {}).get("reid")
                    if face_reid and face_reid.get("embedding_vector"):
                        frame["face_reid"] = decode_embedding(
                            face_reid["embedding_vector"]
                        )
                        frame["face_reid_model"] = face_reid.get(
                            "model_name", ""
                        )
                    frame["face_confidence"] = face.get("confidence")

                    # Also check classificationPolicy metadata (face-reid
                    # tensor "658" ends up here when reidPolicy doesn't match)
                    for key, val in face.get("metadata", {}).items():
                        if "658" in key or "torch-jit-export" in str(
                            val.get("model_name", "")
                            if isinstance(val, dict)
                            else ""
                        ):
                            # This is the face embedding stored by
                            # classificationPolicy
                            if frame["face_reid"] is None and isinstance(
                                val, dict
                            ):
                                frame["face_meta_key"] = key

                frames.append(frame)
        camera_data[cam_name] = frames
    return camera_data


def load_scene_data(dump_dir: Path):
    """Load scene messages, extract UUIDs and timestamps."""
    scene_data = []
    scene_dir = dump_dir / "scene"
    if not scene_dir.exists():
        return scene_data

    for sub in sorted(scene_dir.iterdir()):
        if not sub.is_dir():
            continue
        for f in sorted(sub.glob("*.json")):
            d = json.load(open(f))
            ts = d.get("timestamp", "")
            for obj in d.get("objects", []):
                scene_data.append(
                    {
                        "file": f.name,
                        "timestamp": ts,
                        "uuid": obj.get("id", ""),
                        "category": obj.get("category", ""),
                        "first_seen": obj.get("first_seen", ""),
                        "similarity": obj.get("similarity"),
                        "camera_bounds": obj.get("camera_bounds", {}),
                    }
                )
    return scene_data


def main():
    p = argparse.ArgumentParser(description="Analyze MQTT dump for reid")
    p.add_argument(
        "--dump-dir",
        default="./mqtt_dump",
        help="Path to mqtt_dump directory",
    )
    args = p.parse_args()
    dump_dir = Path(args.dump_dir)

    print("=" * 70)
    print("REID ANALYSIS")
    print("=" * 70)

    # ── Load data ──
    camera_data = load_camera_data(dump_dir)
    scene_data = load_scene_data(dump_dir)

    # ── Camera embedding stats ──
    print("\n── CAMERA EMBEDDING STATS ──")
    for cam, frames in camera_data.items():
        total = len(frames)
        with_person_reid = sum(1 for f in frames if f["person_reid"])
        with_face_reid = sum(1 for f in frames if f["face_reid"])
        print(f"\n{cam}: {total} person detections")
        print(
            f"  Person-reid embeddings: {with_person_reid}"
            f" ({with_person_reid / total * 100:.0f}%)"
            if total
            else ""
        )
        print(
            f"  Face-reid embeddings:   {with_face_reid}"
            f" ({with_face_reid / total * 100:.0f}%)"
            if total
            else ""
        )
        if with_person_reid:
            sample = next(f for f in frames if f["person_reid"])
            print(
                f"  Person-reid dim: {len(sample['person_reid'])}D"
                f" model={sample.get('person_reid_model', '?')}"
            )
        if with_face_reid:
            sample = next(f for f in frames if f["face_reid"])
            print(
                f"  Face-reid dim:   {len(sample['face_reid'])}D"
                f" model={sample.get('face_reid_model', '?')}"
            )

    # ── Scene UUID stats ──
    print("\n── SCENE UUID STATS ──")
    uuid_info = defaultdict(
        lambda: {"count": 0, "first_seen": None, "ts_range": [None, None]}
    )
    for s in scene_data:
        uid = s["uuid"]
        uuid_info[uid]["count"] += 1
        if uuid_info[uid]["first_seen"] is None:
            uuid_info[uid]["first_seen"] = s["first_seen"]
        if (
            uuid_info[uid]["ts_range"][0] is None
            or s["timestamp"] < uuid_info[uid]["ts_range"][0]
        ):
            uuid_info[uid]["ts_range"][0] = s["timestamp"]
        if (
            uuid_info[uid]["ts_range"][1] is None
            or s["timestamp"] > uuid_info[uid]["ts_range"][1]
        ):
            uuid_info[uid]["ts_range"][1] = s["timestamp"]
        if s["camera_bounds"]:
            uuid_info[uid]["cameras"] = set(s["camera_bounds"].keys())
        uuid_info[uid]["similarity"] = s["similarity"]

    print(f"\nUnique UUIDs: {len(uuid_info)}")
    for uid, info in uuid_info.items():
        print(f"\n  UUID: {uid}")
        print(f"    Frames: {info['count']}")
        print(f"    First seen: {info['first_seen']}")
        print(f"    Time range: {info['ts_range'][0]} → {info['ts_range'][1]}")
        print(f"    Similarity: {info['similarity']}")
        cams = info.get("cameras", set())
        print(f"    camera_bounds: {cams if cams else '(empty)'}")

    # ── Timestamp correlation: UUID ↔ camera embeddings ──
    print("\n── UUID ↔ CAMERA EMBEDDING CORRELATION (by timestamp) ──")
    for uid, info in uuid_info.items():
        t_start = info["ts_range"][0]
        t_end = info["ts_range"][1]
        print(f"\n  UUID: {uid[:12]}... ({info['count']} frames)")

        for cam, frames in camera_data.items():
            matching = [
                f
                for f in frames
                if f["person_reid"] and t_start <= f["timestamp"] <= t_end
            ]
            face_matching = [
                f
                for f in frames
                if f["face_reid"] and t_start <= f["timestamp"] <= t_end
            ]
            print(
                f"    {cam}: {len(matching)} person-reid,"
                f" {len(face_matching)} face-reid in time window"
            )

    # ── Cross-camera embedding similarity ──
    print("\n── CROSS-CAMERA EMBEDDING SIMILARITY ──")
    cam_names = list(camera_data.keys())
    if len(cam_names) >= 2:
        cam1, cam2 = cam_names[0], cam_names[1]
        emb1 = [f for f in camera_data[cam1] if f["person_reid"]]
        emb2 = [f for f in camera_data[cam2] if f["person_reid"]]

        if emb1 and emb2:
            # Compare first embedding from each camera
            sim = cosine_similarity(emb1[0]["person_reid"], emb2[0]["person_reid"])
            print(
                f"\n  Person-reid: {cam1} vs {cam2}"
                f" (first embedding each)"
            )
            print(f"    Cosine similarity: {sim:.4f}")

            # Average embeddings per camera and compare
            avg1 = np.mean(
                [f["person_reid"] for f in emb1], axis=0
            ).tolist()
            avg2 = np.mean(
                [f["person_reid"] for f in emb2], axis=0
            ).tolist()
            avg_sim = cosine_similarity(avg1, avg2)
            print(
                f"    Avg embedding similarity: {avg_sim:.4f}"
                f" ({len(emb1)} vs {len(emb2)} embeddings)"
            )
        else:
            print(
                f"\n  Not enough person-reid embeddings:"
                f" {cam1}={len(emb1)}, {cam2}={len(emb2)}"
            )

        # Face-reid cross-camera
        face1 = [f for f in camera_data[cam1] if f["face_reid"]]
        face2 = [f for f in camera_data[cam2] if f["face_reid"]]
        if face1 and face2:
            sim = cosine_similarity(face1[0]["face_reid"], face2[0]["face_reid"])
            print(f"\n  Face-reid: {cam1} vs {cam2} (first embedding each)")
            print(f"    Cosine similarity: {sim:.4f}")

            avg1 = np.mean(
                [f["face_reid"] for f in face1], axis=0
            ).tolist()
            avg2 = np.mean(
                [f["face_reid"] for f in face2], axis=0
            ).tolist()
            avg_sim = cosine_similarity(avg1, avg2)
            print(
                f"    Avg embedding similarity: {avg_sim:.4f}"
                f" ({len(face1)} vs {len(face2)} embeddings)"
            )
    else:
        print("  Need data from 2+ cameras for cross-camera comparison")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
