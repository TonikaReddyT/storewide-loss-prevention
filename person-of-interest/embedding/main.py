# Main Entry Point — Starts all faceid services

import logging
import os
import signal
import sys
import threading

from src.config import API_HOST, API_PORT
from src.faiss_manager import FAISSManager
from src.redis_client import RedisClient
from src.embedding import EmbeddingService
from src.ingestion import IngestionService
from src.alert_service import AlertService
from src.search_api import create_app
from src.maintenance import MaintenanceService
from src.results_collector import ResultsCollector

logging.basicConfig(
    level=logging.DEBUG if os.environ.get("LOG_LEVEL", "").upper() == "DEBUG" else logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("faceid.main")


def main():
    log.info("=== Face-ID Service Starting ===")

    # Initialize components
    log.info("Initializing FAISS indexes...")
    faiss_mgr = FAISSManager()

    log.info("Connecting to Redis...")
    redis_client = RedisClient()
    redis_client.ping()
    log.info("Redis connected")

    log.info("Initializing embedding service...")
    embedding_svc = EmbeddingService()

    # Results collector
    log.info("Initializing results collector...")
    results = ResultsCollector()
    results.start()

    # Alert service (online mode)
    alert_svc = AlertService(redis_client)
    alert_svc.set_results_collector(results)

    # Ingestion service (MQTT → FAISS + Redis)
    ingestion = IngestionService(faiss_mgr, redis_client)
    ingestion.set_alert_callback(alert_svc.on_watchlist_match)
    ingestion.set_results_collector(results)
    ingestion.start()

    # Maintenance service (periodic save + cleanup)
    maintenance = MaintenanceService(faiss_mgr, redis_client)
    maintenance.start()

    # Flask API (enrollment + search)
    app = create_app(faiss_mgr, redis_client, embedding_svc)

    # Graceful shutdown
    def shutdown(sig, frame):
        log.info("Shutting down...")
        ingestion.stop()
        maintenance.stop()
        results.stop()
        faiss_mgr.save_periodic()
        log.info("Shutdown complete")
        sys.exit(0)

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)

    stats = faiss_mgr.get_stats()
    log.info(f"FAISS stats: {stats}")
    log.info(f"Watchlist entries: {len(redis_client.get_watchlist())}")
    log.info(f"=== Face-ID Service Ready on {API_HOST}:{API_PORT} ===")

    # Start Flask (blocking)
    app.run(host=API_HOST, port=API_PORT, threaded=True)


if __name__ == "__main__":
    main()
