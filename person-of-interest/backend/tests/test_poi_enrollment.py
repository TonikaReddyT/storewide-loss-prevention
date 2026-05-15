"""Tests for POI enrollment flow."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Mock redis and faiss before any backend imports attempt them
for mod in ("redis", "faiss"):
    if mod not in sys.modules:
        sys.modules[mod] = MagicMock()

from backend.domain.entities.poi import POI, Severity, POIStatus  # noqa: E402


@pytest.fixture
def mock_poi_repo():
    repo = MagicMock()
    repo.save = MagicMock()
    repo.get = MagicMock(return_value=None)
    repo.list_all = MagicMock(return_value=[])
    repo.delete = MagicMock(return_value=True)
    return repo


@pytest.fixture
def mock_embedding_repo():
    repo = MagicMock()
    repo.add = MagicMock(return_value=[0, 1])
    repo.remove = MagicMock()
    repo.total_vectors = MagicMock(return_value=0)
    return repo


@pytest.fixture
def mock_mapping_repo():
    repo = MagicMock()
    repo.map_faiss_to_poi = MagicMock()
    repo.remove_mappings_for_poi = MagicMock()
    return repo


@pytest.mark.asyncio
async def test_create_poi_stores_metadata(mock_poi_repo, mock_embedding_repo, mock_mapping_repo):
    """create_poi saves POI metadata in the repository."""
    from backend.service.poi_service import POIService

    fake_embedding = np.random.randn(256).astype(np.float32)
    mock_model = MagicMock()
    mock_model.generate_from_bytes = MagicMock(return_value={"embedding": fake_embedding.tolist()})

    with patch("backend.service.poi_service.EmbeddingModelFactory.create", return_value=mock_model), \
         patch("backend.service.poi_service.UPLOAD_DIR") as mock_dir:
        mock_path = MagicMock()
        mock_path.__truediv__ = MagicMock(return_value=mock_path)
        mock_path.mkdir = MagicMock()
        mock_path.write_bytes = MagicMock()
        mock_dir.__truediv__ = MagicMock(return_value=mock_path)

        service = POIService(mock_poi_repo, mock_embedding_repo, mock_mapping_repo)
        result = await service.create_poi(images=[b"fake_image_bytes"], severity="high", notes="Test POI")

    assert "error" not in result
    assert mock_poi_repo.save.called
    saved_poi = mock_poi_repo.save.call_args[0][0]
    assert isinstance(saved_poi, POI)
    assert saved_poi.severity == Severity.HIGH


@pytest.mark.asyncio
async def test_create_poi_no_faces_returns_error(mock_poi_repo, mock_embedding_repo, mock_mapping_repo):
    """create_poi returns error dict when no faces are detected."""
    from backend.service.poi_service import POIService

    mock_model = MagicMock()
    mock_model.generate_from_bytes = MagicMock(return_value={"error": "no face detected"})

    with patch("backend.service.poi_service.EmbeddingModelFactory.create", return_value=mock_model):
        service = POIService(mock_poi_repo, mock_embedding_repo, mock_mapping_repo)
        result = await service.create_poi(images=[b"no_face_image"], severity="medium", notes="")

    assert "error" in result
    mock_poi_repo.save.assert_not_called()


def test_list_pois_returns_serialized_list(mock_poi_repo, mock_embedding_repo, mock_mapping_repo):
    """list_pois returns list of dicts from repository."""
    from backend.service.poi_service import POIService

    poi = POI(
        poi_id="poi-test-001",
        severity=Severity.LOW,
        notes="test",
        reference_images=[],
        status=POIStatus.ACTIVE,
        enrolled_by="system",
        created_at="2025-01-15T10:00:00Z",
        embedding_ids=[],
    )
    mock_poi_repo.list_all = MagicMock(return_value=[poi])

    service = POIService(mock_poi_repo, mock_embedding_repo, mock_mapping_repo)
    result = service.list_pois()

    assert len(result) == 1
    assert result[0]["poi_id"] == "poi-test-001"


def test_delete_poi_removes_from_all_repos(mock_poi_repo, mock_embedding_repo, mock_mapping_repo):
    """delete_poi removes vectors, mappings, and metadata."""
    from backend.service.poi_service import POIService

    service = POIService(mock_poi_repo, mock_embedding_repo, mock_mapping_repo)
    result = service.delete_poi("poi-test-001")

    assert result is True
    mock_embedding_repo.remove.assert_called_once_with("poi-test-001")
    mock_mapping_repo.remove_mappings_for_poi.assert_called_once_with("poi-test-001")
    mock_poi_repo.delete.assert_called_once_with("poi-test-001")
