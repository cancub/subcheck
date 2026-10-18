"""Tests for `subcheck serve` (`subcheck/server.py`)."""

from __future__ import annotations

import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

from subcheck.autosub import AutoSubResult, Candidate
from subcheck.server import (
    ServerConfig,
    SubcheckWorker,
    WebhookSpec,
    apply_path_map,
    auto_register_arr_webhook,
    expand_video_targets,
    make_handler,
    notify_jellyfin,
    parse_jellyfin_webhook,
    parse_path_maps,
    parse_radarr_webhook,
    parse_sonarr_webhook,
)


def test_path_mapping():
    maps = parse_path_maps("/tv:/data/media/tv, /movies:/data/media/movies")
    assert apply_path_map("/tv/Show/S01E01.mkv", maps) == "/data/media/tv/Show/S01E01.mkv"
    assert apply_path_map("/movies/Film/movie.mp4", maps) == "/data/media/movies/Film/movie.mp4"
    assert apply_path_map("/other/path.mkv", maps) == "/other/path.mkv"


def test_expand_video_targets(tmp_path: Path):
    ep1 = tmp_path / "S01E01.mkv"
    ep2 = tmp_path / "sub" / "S01E02.mp4"
    ignored1 = tmp_path / ".subcheck-remux-123.mkv"
    ignored2 = tmp_path / "S01E01.autosub.mkv"
    ep2.parent.mkdir()
    for p in (ep1, ep2, ignored1, ignored2):
        p.write_bytes(b"video")

    targets = expand_video_targets(str(tmp_path))
    assert targets == [str(ep1), str(ep2)]


def test_parse_radarr_webhook():
    event, specs = parse_radarr_webhook({"eventType": "Test"})
    assert event == "Test" and specs == []

    payload = {
        "eventType": "Download",
        "movie": {"title": "Alien: Romulus", "year": 2024, "folderPath": "/data/media/movies/Alien Romulus (2024)"},
        "movieFile": {"path": "/data/media/movies/Alien Romulus (2024)/Alien.Romulus.2024.mkv"},
    }
    event, specs = parse_radarr_webhook(payload)
    assert event == "Download"
    assert len(specs) == 1
    assert specs[0].source == "radarr"
    assert specs[0].title == "Alien: Romulus"
    assert specs[0].video == "/data/media/movies/Alien Romulus (2024)/Alien.Romulus.2024.mkv"


def test_parse_sonarr_webhook_single_and_season_pack():
    event, specs = parse_sonarr_webhook({"eventType": "Test"})
    assert event == "Test" and specs == []

    # Single episode
    single_payload = {
        "eventType": "Download",
        "series": {"title": "Band of Brothers", "path": "/data/media/tv/Band of Brothers"},
        "episodes": [{"id": 10, "episodeFileId": 100, "seasonNumber": 1, "episodeNumber": 2}],
        "episodeFile": {"id": 100, "path": "/data/media/tv/Band of Brothers/S01E02 Day of Days.mkv"},
    }
    event, specs = parse_sonarr_webhook(single_payload)
    assert event == "Download"
    assert len(specs) == 1
    assert specs[0].title == "Band of Brothers"
    assert specs[0].season == 1 and specs[0].episode == 2

    # Multi-episode season pack
    pack_payload = {
        "eventType": "Download",
        "series": {"title": "Wayne", "path": "/data/media/tv/Wayne"},
        "episodes": [
            {"id": 1, "episodeFileId": 101, "seasonNumber": 1, "episodeNumber": 1},
            {"id": 2, "episodeFileId": 102, "seasonNumber": 1, "episodeNumber": 2},
        ],
        "episodeFiles": [
            {"id": 101, "path": "/data/media/tv/Wayne/S01E01.mkv"},
            {"id": 102, "path": "/data/media/tv/Wayne/S01E02.mkv"},
        ],
    }
    event, specs = parse_sonarr_webhook(pack_payload)
    assert len(specs) == 2
    assert (specs[0].season, specs[0].episode) == (1, 1)
    assert (specs[1].season, specs[1].episode) == (1, 2)


def test_parse_jellyfin_webhook():
    event, specs = parse_jellyfin_webhook(
        {
            "NotificationType": "ItemAdded",
            "ItemType": "Episode",
            "SeriesName": "The Rehearsal",
            "SeasonNumber": 1,
            "EpisodeNumber": 3,
            "Path": "/media/media/tv/The Rehearsal/S01E03.mp4",
        }
    )
    assert event == "ItemAdded"
    assert len(specs) == 1
    assert specs[0].title == "The Rehearsal"
    assert specs[0].season == 1 and specs[0].episode == 3


def test_worker_queue_and_http_endpoints(tmp_path: Path):
    video = tmp_path / "Test.Movie.2024.mkv"
    video.write_bytes(b"dummy")
    history_file = tmp_path / "history.json"

    def fake_autosub(video_path: str, **kwargs):
        cand = Candidate(
            file_id=0,
            release="embedded stream #2 (subrip, eng)",
            file_name="embedded_2.srt",
            path=video_path,
            from_cache=True,
            composite=62.0,
            composite_after=94.5,
            best_mode="shift",
            best_by=-1.25,
            best_scale=1.0,
            rank=1,
        )
        return AutoSubResult(
            video=video_path,
            meta={"title": kwargs.get("title") or "Test Movie", "season": None, "episode": None},
            language=kwargs.get("language", "en"),
            limit=3,
            candidates=[cand],
            windows_count=10,
            chosen=cand,
            embed_status="ok",
            embed_out=video_path,
            notes=["shifted embedded stream by -1.25s"],
        )

    cfg = ServerConfig(
        host="127.0.0.1",
        port=0,
        history_file=str(history_file),
        browse_root=str(tmp_path),
    )
    worker = SubcheckWorker(cfg, autosub_fn=fake_autosub, start_thread=False)

    # Deduplication check: enqueueing same file twice while queued returns same job
    j1 = worker.enqueue(WebhookSpec(source="radarr", video=str(video), title="Test Movie"))
    j2 = worker.enqueue(WebhookSpec(source="radarr", video=str(video), title="Test Movie"))
    assert j1.id == j2.id
    assert worker.status_snapshot()["queue_size"] == 1

    # Process job synchronously
    processed = worker.process_next_sync()
    assert processed is not None
    assert processed.status == "completed"
    assert processed.score_before == 62.0
    assert processed.score_after == 94.5
    assert processed.fix_mode == "shift"
    assert processed.fix_by == -1.25
    assert history_file.is_file()

    # Spin up HTTP server on ephemeral port
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(worker))
    port = httpd.server_address[1]
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    base = f"http://127.0.0.1:{port}"
    try:
        # 1. GET /healthz
        with urllib.request.urlopen(f"{base}/healthz") as r:
            data = json.loads(r.read().decode())
            assert data["status"] == "ok"

        # 2. GET / (Dashboard HTML)
        with urllib.request.urlopen(f"{base}/") as r:
            html_body = r.read().decode()
            assert "subcheck" in html_body

        # 3. GET /api/browse
        with urllib.request.urlopen(f"{base}/api/browse?path={tmp_path}") as r:
            bdata = json.loads(r.read().decode())
            assert any(e["name"] == "Test.Movie.2024.mkv" for e in bdata["entries"])

        # 4. POST /webhook/radarr Test event
        req = urllib.request.Request(
            f"{base}/webhook/radarr",
            data=json.dumps({"eventType": "Test"}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req) as r:
            assert r.status == 200

        # 5. POST /api/autosub
        req2 = urllib.request.Request(
            f"{base}/api/autosub",
            data=json.dumps({"video": str(video), "title": "Manual Movie"}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req2) as r:
            assert r.status == 202
            resp_data = json.loads(r.read().decode())
            assert len(resp_data["queued"]) == 1
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_notify_jellyfin_and_auto_register(monkeypatch):
    calls = []

    class FakeResp:
        def __init__(self, status=200, body=b"[]"):
            self.status = status
            self._body = body

        def read(self):
            return self._body

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    def fake_urlopen(req, timeout=10.0):
        body = json.loads(req.data.decode("utf-8")) if req.data else None
        calls.append((req.method, req.full_url, body))
        if req.method == "GET" and req.full_url.endswith("/api/v3/notification"):
            return FakeResp(200, b"[]")
        return FakeResp(201, b'{"id": 1}')

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    ok, msg = notify_jellyfin(
        "http://jellyfin:8096",
        "secret-token",
        "/data/media/movies/Film.mkv",
        jellyfin_path_maps=[("/data", "/media")],
    )
    assert ok is True
    assert calls[-1][1] == "http://jellyfin:8096/Library/Media/Updated"
    assert calls[-1][2]["Updates"][0]["Path"] == "/media/media/movies/Film.mkv"

    ok2, msg2 = auto_register_arr_webhook(
        "Sonarr",
        "http://sonarr:8989",
        "sonarr-key",
        "http://subcheck:8765/webhook/sonarr",
    )
    assert ok2 is True
    assert "created webhook" in msg2
