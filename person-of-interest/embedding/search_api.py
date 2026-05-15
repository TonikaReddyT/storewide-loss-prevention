# Search & Enrollment API — Flask REST endpoints

import collections
import io
import logging
import uuid as uuid_lib

from flask import Flask, request, jsonify

from src.config import SEARCH_DISTANCE_THRESHOLD, SEARCH_TOP_K

log = logging.getLogger("faceid.api")


def create_app(faiss_mgr, redis_client, embedding_svc):
    """Create Flask app with all API routes."""
    app = Flask(__name__)

    # ── Enrollment (Online Mode) ─────────────────────────────

    @app.route("/api/enroll", methods=["POST"])
    def enroll():
        """Enroll a person into the watchlist from a photo.

        Accepts:
            multipart/form-data with:
                - photo: image file
                - name: person's name
                - threat_level: (optional) high/medium/low
        """
        if "photo" not in request.files:
            return jsonify({"error": "No photo provided"}), 400

        photo = request.files["photo"]
        name = request.form.get("name", "")
        threat_level = request.form.get("threat_level", "high")

        if not name:
            return jsonify({"error": "Name is required"}), 400

        # Generate embedding
        image_bytes = photo.read()
        result = embedding_svc.generate_from_bytes(image_bytes)
        if "error" in result:
            return jsonify(result), 400

        embedding = result["embedding"]

        # Assign UUID
        person_uuid = str(uuid_lib.uuid4())

        # Add to FAISS watchlist
        faiss_ids = faiss_mgr.add_to_watchlist(person_uuid, [embedding])

        # Store in Redis
        redis_client.add_to_watchlist(person_uuid)
        redis_client.store_identity(person_uuid, name=name, threat_level=threat_level)
        redis_client.map_faiss_to_uuid(faiss_ids, person_uuid)

        log.info(f"Enrolled {name} as {person_uuid} (threat={threat_level})")
        return jsonify({
            "uuid": person_uuid,
            "name": name,
            "threat_level": threat_level,
            "face_bbox": result.get("face_bbox"),
            "confidence": result.get("confidence"),
        }), 201

    @app.route("/api/enroll/<person_uuid>", methods=["DELETE"])
    def unenroll(person_uuid):
        """Remove a person from the watchlist."""
        faiss_mgr.remove_from_watchlist(person_uuid)
        redis_client.remove_from_watchlist(person_uuid)
        log.info(f"Unenrolled {person_uuid}")
        return jsonify({"removed": person_uuid}), 200

    @app.route("/api/watchlist", methods=["GET"])
    def list_watchlist():
        """List all enrolled watchlist persons."""
        uuids = redis_client.get_watchlist()
        persons = []
        for uid in uuids:
            identity = redis_client.get_identity(uid)
            persons.append({
                "uuid": uid,
                "name": identity.get("name", ""),
                "threat_level": identity.get("threat_level", ""),
                "enrolled": identity.get("first_seen", ""),
            })
        return jsonify({"watchlist": persons, "count": len(persons)})

    # ── Search (Offline Mode) ────────────────────────────────

    @app.route("/api/search", methods=["POST"])
    def search():
        """Search for a person in the 7-day history by photo.

        Accepts:
            multipart/form-data with:
                - photo: image file
                - limit: (optional) max results, default 50
        """
        if "photo" not in request.files:
            return jsonify({"error": "No photo provided"}), 400

        photo = request.files["photo"]
        limit = int(request.form.get("limit", SEARCH_TOP_K))

        # Generate embedding
        image_bytes = photo.read()
        result = embedding_svc.generate_from_bytes(image_bytes)
        if "error" in result:
            return jsonify(result), 400

        embedding = result["embedding"]

        # Search history FAISS
        matches = faiss_mgr.search_history(embedding, k=limit)

        if not matches:
            return jsonify({
                "found": False,
                "message": "No matches found in 7-day history",
                "face_bbox": result.get("face_bbox"),
            })

        # Filter by threshold and group by UUID
        uuid_distances = {}
        for uid, dist in matches:
            if dist < SEARCH_DISTANCE_THRESHOLD:
                if uid not in uuid_distances or dist < uuid_distances[uid]:
                    uuid_distances[uid] = dist

        if not uuid_distances:
            return jsonify({
                "found": False,
                "message": "No confident matches (all distances above threshold)",
                "best_distance": matches[0][1] if matches else None,
                "threshold": SEARCH_DISTANCE_THRESHOLD,
            })

        # Find best match (majority + lowest distance)
        counter = collections.Counter(uid for uid, _ in matches if uid in uuid_distances)
        best_uuid = counter.most_common(1)[0][0]
        best_distance = uuid_distances[best_uuid]

        # Get appearances from Redis
        appearances = redis_client.get_appearances(best_uuid)
        identity = redis_client.get_identity(best_uuid)

        # Aggregate by day
        by_day = {}
        for app in appearances:
            ts = app.get("timestamp", "")
            day = ts[:10] if len(ts) >= 10 else "unknown"
            if day not in by_day:
                by_day[day] = []
            by_day[day].append({
                "time": ts[11:19] if len(ts) >= 19 else ts,
                "camera": app.get("camera_id", ""),
                "aisle": app.get("aisle", ""),
                "duration_s": int(app.get("duration_s", 0)),
            })

        return jsonify({
            "found": True,
            "person_uuid": best_uuid,
            "match_distance": round(best_distance, 4),
            "name": identity.get("name", ""),
            "total_appearances": int(identity.get("total_appearances", 0)),
            "first_seen": identity.get("first_seen", ""),
            "last_seen": identity.get("last_seen", ""),
            "appearances_by_day": by_day,
            "total_records": len(appearances),
            "face_bbox": result.get("face_bbox"),
            "confidence": result.get("confidence"),
        })

    # ── Status ───────────────────────────────────────────────

    @app.route("/api/status", methods=["GET"])
    def status():
        """Get system status."""
        stats = faiss_mgr.get_stats()
        watchlist = redis_client.get_watchlist()
        return jsonify({
            "status": "running",
            "faiss": stats,
            "watchlist_count": len(watchlist),
            "redis": "connected" if redis_client.ping() else "disconnected",
        })

    @app.route("/api/alerts", methods=["GET"])
    def get_alerts():
        """Get recent alerts."""
        count = int(request.args.get("count", 50))
        alerts = redis_client.get_recent_alerts(count)
        return jsonify({"alerts": alerts, "count": len(alerts)})

    return app
