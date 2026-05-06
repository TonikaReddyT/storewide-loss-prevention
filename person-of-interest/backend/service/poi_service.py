"""POI Service — business logic for POI CRUD operations."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np

from backend.domain.entities.poi import POI
from backend.domain.interfaces.repository import EmbeddingMappingRepository, EmbeddingRepository, POIRepository
from backend.factory.factories import EmbeddingModelFactory
from backend.utils.builder import POIBuilder
from backend.utils.face_processing import build_poi_embedding

log = logging.getLogger("poi.service.poi")

UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "/data/uploads"))

# Enrollment face crop settings — keep at 0 for DLStreamer parity,
# set to 0.10–0.15 if enrollment images are very different from runtime.
ENROLL_PADDING = float(os.getenv("ENROLL_FACE_PADDING", "0.0"))
ENROLL_SQUARE = os.getenv("ENROLL_FACE_SQUARE", "false").lower() == "true"
# "mean" → average all reference embeddings into one FAISS vector
# "all"  → store each reference as a separate FAISS vector
ENROLL_STRATEGY = os.getenv("ENROLL_EMBEDDING_STRATEGY", "mean")


class POIService:
    """Orchestrates POI creation, listing, and deletion."""

    def __init__(
        self,
        poi_repo: POIRepository,
        embedding_repo: EmbeddingRepository,
        mapping_repo: EmbeddingMappingRepository,
    ) -> None:
        self._poi_repo = poi_repo
        self._embedding_repo = embedding_repo
        self._mapping_repo = mapping_repo

    async def create_poi(
        self,
        images: list[bytes],
        severity: str = "medium",
        notes: str = "",
    ) -> dict:
        poi_id = POI.generate_id()
        builder = POIBuilder().with_id(poi_id).with_severity(severity).with_notes(notes)
        model = EmbeddingModelFactory.create()

        raw_embeddings = []
        for idx, img_bytes in enumerate(images):
            # Save debug crop alongside reference image
            img_dir = UPLOAD_DIR / poi_id
            img_dir.mkdir(parents=True, exist_ok=True)
            crop_debug_path = str(img_dir / f"ref_{idx}_crop128.jpg")

            result = model.generate_from_bytes(
                img_bytes,
                padding=ENROLL_PADDING,
                make_square=ENROLL_SQUARE,
                save_crop_path=crop_debug_path,
            )
            if "error" in result:
                log.warning("Image %d failed: %s", idx, result["error"])
                continue

            emb_id = f"emb-{poi_id}-ref-{idx:02d}"
            # Save original image to disk
            img_path = img_dir / f"ref_{idx}.jpg"
            img_path.write_bytes(img_bytes)

            builder.add_image(emb_id, f"/uploads/{poi_id}/ref_{idx}.jpg")
            emb = np.array(result["embedding"], dtype=np.float32)
            raw_embeddings.append(emb)
            log.info(
                "Ref image %d: bbox=%s conf=%.3f norm=%.6f face_size=%s",
                idx, result.get("face_bbox"), result.get("confidence"),
                result.get("embedding_norm", 0.0), result.get("face_size"),
            )

        if not raw_embeddings:
            return {"error": "No faces detected in any uploaded image"}

        # Build final embeddings using the configured strategy
        if ENROLL_STRATEGY == "mean" and len(raw_embeddings) > 1:
            mean_vec = build_poi_embedding(raw_embeddings, strategy="mean")
            embeddings = [mean_vec]
            log.info(
                "POI %s: averaged %d reference embeddings into 1 (strategy=%s)",
                poi_id, len(raw_embeddings), ENROLL_STRATEGY,
            )
        else:
            embeddings = raw_embeddings
            log.info(
                "POI %s: storing %d individual embeddings (strategy=%s)",
                poi_id, len(embeddings), ENROLL_STRATEGY,
            )

        poi = builder.build()

        # Store vectors in FAISS
        faiss_ids = self._embedding_repo.add(poi_id, embeddings)

        # Map FAISS IDs → POI ID
        for fid in faiss_ids:
            self._mapping_repo.map_faiss_to_poi(fid, poi_id)

        # Save metadata in Redis
        self._poi_repo.save(poi)

        log.info("Created POI %s with %d embeddings", poi_id, len(embeddings))
        return poi.to_dict()

    def list_pois(self) -> list[dict]:
        pois = self._poi_repo.list_all()
        return [p.to_dict() for p in pois]

    def get_poi(self, poi_id: str) -> Optional[dict]:
        poi = self._poi_repo.get(poi_id)
        return poi.to_dict() if poi else None

    def delete_poi(self, poi_id: str) -> bool:
        # Remove metadata first (authoritative source) — if this fails,
        # embeddings remain searchable which is the safer failure mode.
        deleted = self._poi_repo.delete(poi_id)
        if not deleted:
            return False
        # Remove from FAISS
        self._embedding_repo.remove(poi_id)
        # Remove FAISS→POI mappings
        self._mapping_repo.remove_mappings_for_poi(poi_id)
        log.info("Deleted POI %s", poi_id)
        return True
