"""HTTP webhook server, sequential GPU worker queue, and Web UI (`subcheck serve`).

Designed for *arr + Jellyfin/Plex media stacks:
- Receives Sonarr (`POST /webhook/sonarr`), Radarr (`POST /webhook/radarr`),
  and Jellyfin (`POST /webhook/jellyfin`) import webhooks.
- Exposes a REST API (`POST /api/autosub`, `GET /api/status`, `GET /api/browse`,
  `GET /healthz`) and a dark-mode single-page dashboard (`GET /`).
- Processes video files sequentially on a background worker thread so GPU
  memory and OpenSubtitles rate limits are respected even on season-pack imports.
- Updates library files in-place via atomic replacement (`os.replace`),
  preserving source torrent hardlinks/hashes in the download directory.
- Optionally auto-registers its own webhook in Sonarr/Radarr on startup and
  notifies Jellyfin (`POST /Library/Media/Updated`) when a file is updated.
"""

from __future__ import annotations

import datetime
import html
import json
import logging
import os
import queue
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import asdict, dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Dict, List, Optional, Tuple

from . import __version__
from .autosub import AutoSubResult, parse_video_name, run_autosub

logger = logging.getLogger("subcheck.server")

VIDEO_EXTENSIONS = {".mkv", ".mp4", ".webm", ".mov", ".m4v"}


def _utc_now_iso() -> str:
    return (
        datetime.datetime.now(datetime.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def parse_path_maps(raw: Optional[str]) -> List[Tuple[str, str]]:
    """Parse 'FROM:TO,FROM2:TO2' into a list of (from_prefix, to_prefix) pairs."""
    if not raw:
        return []
    out: List[Tuple[str, str]] = []
    for item in raw.split(","):
        item = item.strip()
        if not item or ":" not in item:
            continue
        src, dst = item.split(":", 1)
        src = src.strip().rstrip("/")
        dst = dst.strip().rstrip("/")
        if src and dst:
            out.append((src, dst))
    return out


def apply_path_map(path: str, mappings: List[Tuple[str, str]]) -> str:
    """Translate *path* using the first matching prefix in *mappings*."""
    if not path or not mappings:
        return path
    for src, dst in mappings:
        if path == src:
            return dst
        if path.startswith(src + "/"):
            return dst + path[len(src) :]
    return path


def expand_video_targets(target_path: str) -> List[str]:
    """Return [target_path] for a single file, or all video files recursively
    inside *target_path* when it is a directory."""
    if not os.path.exists(target_path):
        return [target_path]
    if os.path.isfile(target_path):
        return [target_path]
    videos: List[str] = []
    for root, dirs, files in os.walk(target_path):
        dirs.sort()
        for f in sorted(files):
            if f.startswith(".subcheck-remux-"):
                continue
            ext = os.path.splitext(f)[1].lower()
            if ext not in VIDEO_EXTENSIONS:
                continue
            stem = os.path.splitext(f)[0]
            if stem.endswith(".autosub"):
                continue
            videos.append(os.path.join(root, f))
    return videos


# --------------------------------------------------------------------------- #
# Webhook Payload Parsers (Radarr / Sonarr / Jellyfin)


@dataclass
class WebhookSpec:
    """Normalized job request extracted from a webhook or API call."""

    source: str
    video: str
    title: Optional[str] = None
    season: Optional[int] = None
    episode: Optional[int] = None
    language: Optional[str] = None
    refresh: bool = False
    skip_embedded: bool = False
    in_place: Optional[bool] = None


def parse_radarr_webhook(payload: Dict[str, Any]) -> Tuple[str, List[WebhookSpec]]:
    """Parse a Radarr webhook payload.
    Returns (event_type, list_of_job_specs)."""
    event_type = str(payload.get("eventType") or "").strip()
    if event_type.lower() == "test":
        return "Test", []
    if event_type not in ("Download", "MovieFileRename"):
        return event_type or "Unknown", []

    movie = payload.get("movie") or {}
    movie_file = payload.get("movieFile") or {}
    title = movie.get("title") or None

    video_path = movie_file.get("path") or ""
    if not video_path and movie.get("folderPath") and movie_file.get("relativePath"):
        video_path = os.path.join(
            str(movie["folderPath"]), str(movie_file["relativePath"])
        )
    if not video_path:
        return event_type, []

    return event_type, [
        WebhookSpec(
            source="radarr",
            video=str(video_path),
            title=str(title) if title else None,
        )
    ]


def parse_sonarr_webhook(payload: Dict[str, Any]) -> Tuple[str, List[WebhookSpec]]:
    """Parse a Sonarr webhook payload (single episode or multi-episode / season pack).
    Returns (event_type, list_of_job_specs)."""
    event_type = str(payload.get("eventType") or "").strip()
    if event_type.lower() == "test":
        return "Test", []
    if event_type not in ("Download", "Rename"):
        return event_type or "Unknown", []

    series = payload.get("series") or {}
    series_title = series.get("title") or None
    series_path = series.get("path") or ""
    episodes = payload.get("episodes") or []

    # Map episodeFileId -> (seasonNumber, episodeNumber)
    ep_by_file_id: Dict[Any, Tuple[Optional[int], Optional[int]]] = {}
    for ep in episodes:
        fid = ep.get("episodeFileId") or ep.get("id")
        s_num = ep.get("seasonNumber")
        e_num = ep.get("episodeNumber")
        if fid is not None and fid not in ep_by_file_id:
            ep_by_file_id[fid] = (
                int(s_num) if s_num is not None else None,
                int(e_num) if e_num is not None else None,
            )

    default_season: Optional[int] = None
    default_episode: Optional[int] = None
    if episodes:
        s0 = episodes[0].get("seasonNumber")
        e0 = episodes[0].get("episodeNumber")
        default_season = int(s0) if s0 is not None else None
        default_episode = int(e0) if e0 is not None else None

    specs: List[WebhookSpec] = []
    episode_files = payload.get("episodeFiles") or []
    single_file = payload.get("episodeFile")
    if single_file and not episode_files:
        episode_files = [single_file]

    for ef in episode_files:
        video_path = ef.get("path") or ""
        if not video_path and series_path and ef.get("relativePath"):
            video_path = os.path.join(str(series_path), str(ef["relativePath"]))
        if not video_path:
            continue

        ef_id = ef.get("id")
        season, episode = ep_by_file_id.get(ef_id, (None, None))
        if season is None or episode is None:
            parsed = parse_video_name(video_path)
            season = season if season is not None else parsed.get("season")
            episode = episode if episode is not None else parsed.get("episode")
        if len(episode_files) == 1:
            if season is None:
                season = default_season
            if episode is None:
                episode = default_episode

        specs.append(
            WebhookSpec(
                source="sonarr",
                video=str(video_path),
                title=str(series_title) if series_title else None,
                season=int(season) if season is not None else None,
                episode=int(episode) if episode is not None else None,
            )
        )

    return event_type, specs


def parse_jellyfin_webhook(payload: Dict[str, Any]) -> Tuple[str, List[WebhookSpec]]:
    """Parse a Jellyfin webhook payload."""
    notif_type = str(
        payload.get("NotificationType") or payload.get("eventType") or ""
    ).strip()
    if notif_type.lower() == "test":
        return "Test", []

    video_path = str(payload.get("Path") or payload.get("path") or "").strip()
    if not video_path:
        return notif_type or "Unknown", []

    item_type = str(payload.get("ItemType") or "").lower()
    if item_type == "episode" or payload.get("SeriesName"):
        title = payload.get("SeriesName") or payload.get("Name")
        s_num = payload.get("SeasonNumber")
        e_num = payload.get("EpisodeNumber")
        return notif_type or "ItemAdded", [
            WebhookSpec(
                source="jellyfin",
                video=video_path,
                title=str(title) if title else None,
                season=int(s_num) if s_num is not None else None,
                episode=int(e_num) if e_num is not None else None,
            )
        ]

    title = payload.get("Name") or payload.get("title")
    return notif_type or "ItemAdded", [
        WebhookSpec(
            source="jellyfin",
            video=video_path,
            title=str(title) if title else None,
        )
    ]


# --------------------------------------------------------------------------- #
# External Integrations (Jellyfin Refresh & Sonarr/Radarr Webhook Auto-Wire)


def notify_jellyfin(
    jellyfin_url: str,
    api_key: str,
    video_path: str,
    jellyfin_path_maps: Optional[List[Tuple[str, str]]] = None,
    timeout: float = 10.0,
) -> Tuple[bool, str]:
    """Notify Jellyfin that *video_path* was updated so it re-probes the subtitle streams."""
    if not jellyfin_url or not api_key:
        return False, "not configured"
    mapped = apply_path_map(video_path, jellyfin_path_maps or [])
    base = jellyfin_url.rstrip("/")
    url = f"{base}/Library/Media/Updated"
    body = json.dumps(
        {"Updates": [{"Path": mapped, "UpdateType": "Modified"}]}
    ).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f'MediaBrowser Token="{api_key}"',
            "X-Emby-Token": api_key,
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if 200 <= resp.status < 300:
                return True, f"notified ({mapped})"
            return False, f"HTTP {resp.status}"
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)


def auto_register_arr_webhook(
    arr_name: str,
    arr_url: str,
    api_key: str,
    webhook_url: str,
    timeout: float = 10.0,
) -> Tuple[bool, str]:
    """Ensure a 'Subcheck' webhook is registered in Sonarr or Radarr v3 API."""
    if not arr_url or not api_key or not webhook_url:
        return False, "not configured"
    base = arr_url.rstrip("/")
    headers = {
        "Content-Type": "application/json",
        "X-Api-Key": api_key,
    }
    list_req = urllib.request.Request(
        f"{base}/api/v3/notification", headers=headers, method="GET"
    )
    try:
        with urllib.request.urlopen(list_req, timeout=timeout) as resp:
            items = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        return False, f"failed to query {arr_name} notifications: {exc}"

    existing = next(
        (
            item
            for item in items
            if str(item.get("name", "")).lower() == "subcheck"
        ),
        None,
    )

    fields = [
        {"name": "url", "value": webhook_url},
        {"name": "method", "value": 1},  # 1 = POST
    ]
    payload: Dict[str, Any] = {
        "name": "Subcheck",
        "implementation": "Webhook",
        "implementationName": "Webhook",
        "configContract": "WebhookSettings",
        "onGrab": False,
        "onDownload": True,
        "onUpgrade": True,
        "onRename": False,
        "onHealthIssue": False,
        "onApplicationUpdate": False,
        "fields": fields,
        "tags": [],
    }
    if existing:
        # Preserve any existing fields and update url/method
        existing_fields = {
            f.get("name"): f.get("value") for f in existing.get("fields", [])
        }
        if (
            existing_fields.get("url") == webhook_url
            and existing.get("onDownload") is True
            and existing.get("onUpgrade") is True
        ):
            return True, f"already registered in {arr_name} ({webhook_url})"
        merged_fields = []
        seen_keys = set()
        for f in existing.get("fields", []):
            k = f.get("name")
            if k == "url":
                merged_fields.append({"name": "url", "value": webhook_url})
                seen_keys.add("url")
            elif k == "method":
                merged_fields.append({"name": "method", "value": 1})
                seen_keys.add("method")
            else:
                merged_fields.append(f)
        if "url" not in seen_keys:
            merged_fields.append({"name": "url", "value": webhook_url})
        if "method" not in seen_keys:
            merged_fields.append({"name": "method", "value": 1})
        existing["fields"] = merged_fields
        existing["onDownload"] = True
        existing["onUpgrade"] = True
        nid = existing.get("id")
        put_req = urllib.request.Request(
            f"{base}/api/v3/notification/{nid}",
            data=json.dumps(existing).encode("utf-8"),
            headers=headers,
            method="PUT",
        )
        try:
            with urllib.request.urlopen(put_req, timeout=timeout) as resp:
                if 200 <= resp.status < 300:
                    return True, f"updated existing webhook in {arr_name} ({webhook_url})"
        except Exception as exc:  # noqa: BLE001
            return False, f"failed to update webhook in {arr_name}: {exc}"

    post_req = urllib.request.Request(
        f"{base}/api/v3/notification",
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(post_req, timeout=timeout) as resp:
            if 200 <= resp.status < 300:
                return True, f"created webhook in {arr_name} ({webhook_url})"
            return False, f"{arr_name} returned HTTP {resp.status}"
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode("utf-8", errors="replace")[:300]
        except Exception:  # noqa: BLE001
            pass
        return False, f"failed to create webhook in {arr_name}: HTTP {exc.code} {detail}"
    except Exception as exc:  # noqa: BLE001
        return False, f"failed to create webhook in {arr_name}: {exc}"


# --------------------------------------------------------------------------- #
# Server Config, Job Record & Sequential GPU Worker


@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8765
    language: str = "en"
    limit: int = 3
    snippets: int = 10
    duration: float = 10.0
    min_gap: float = 30.0
    model: str = "large-v3"
    device: str = "auto"
    compute_type: Optional[str] = None
    model_dir: Optional[str] = None
    beam_size: int = 5
    in_place: bool = True
    path_maps: List[Tuple[str, str]] = field(default_factory=list)
    jellyfin_url: str = ""
    jellyfin_api_key: str = ""
    jellyfin_path_maps: List[Tuple[str, str]] = field(default_factory=list)
    sonarr_url: str = ""
    sonarr_api_key: str = ""
    radarr_url: str = ""
    radarr_api_key: str = ""
    self_url: str = "http://subcheck.media-stack.svc.cluster.local:8765"
    auto_register: bool = False
    history_file: Optional[str] = None
    max_history: int = 200
    browse_root: str = "/data"


@dataclass
class JobRecord:
    id: str
    source: str
    video: str
    title: Optional[str] = None
    season: Optional[int] = None
    episode: Optional[int] = None
    language: str = "en"
    refresh: bool = False
    skip_embedded: bool = False
    in_place: bool = True
    status: str = "queued"  # queued | running | completed | failed
    created_at: str = field(default_factory=_utc_now_iso)
    started_at: str = ""
    finished_at: str = ""
    duration_s: float = 0.0
    chosen_source: str = ""
    score_before: Optional[float] = None
    score_after: Optional[float] = None
    fix_mode: str = ""
    fix_by: float = 0.0
    fix_scale: float = 1.0
    embed_status: str = ""
    embed_out: str = ""
    downloads_new: int = 0
    cache_hits: int = 0
    quota: Optional[dict] = None
    notes: List[str] = field(default_factory=list)
    error: str = ""
    jellyfin_status: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SubcheckWorker:
    """Manages a sequential job queue and history for `subcheck serve`."""

    def __init__(
        self,
        config: ServerConfig,
        autosub_fn: Optional[Callable[..., AutoSubResult]] = None,
        start_thread: bool = True,
    ) -> None:
        self.config = config
        self._autosub_fn = autosub_fn or run_autosub
        self._queue: "queue.Queue[Optional[JobRecord]]" = queue.Queue()
        self._lock = threading.Lock()
        self._active_job: Optional[JobRecord] = None
        self._queued_jobs: List[JobRecord] = []
        self._history: List[JobRecord] = []
        self._last_quota: Optional[dict] = None
        self._integration_status: Dict[str, str] = {}
        self._stop_event = threading.Event()
        self._load_history()
        self._thread: Optional[threading.Thread] = None
        if start_thread:
            self._thread = threading.Thread(
                target=self._run_loop, name="subcheck-worker", daemon=True
            )
            self._thread.start()

    def _load_history(self) -> None:
        hf = self.config.history_file
        if not hf or not os.path.isfile(hf):
            return
        try:
            raw = json.loads( open(hf, encoding="utf-8").read() )
            if isinstance(raw, list):
                for item in raw[: self.config.max_history]:
                    if isinstance(item, dict) and "id" in item and "video" in item:
                        self._history.append(
                            JobRecord(
                                **{
                                    k: v
                                    for k, v in item.items()
                                    if k in JobRecord.__dataclass_fields__
                                }
                            )
                        )
                        if not self._last_quota and item.get("quota"):
                            self._last_quota = item["quota"]
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to load history file %s: %s", hf, exc)

    def _save_history(self) -> None:
        hf = self.config.history_file
        if not hf:
            return
        try:
            os.makedirs(os.path.dirname(os.path.abspath(hf)), exist_ok=True)
            data = [j.to_dict() for j in self._history[: self.config.max_history]]
            tmp = hf + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2, ensure_ascii=False)
            os.replace(tmp, hf)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to save history file %s: %s", hf, exc)

    def enqueue(self, spec: WebhookSpec) -> JobRecord:
        mapped_video = apply_path_map(spec.video, self.config.path_maps)
        with self._lock:
            # Deduplicate if already queued or currently running for the same video
            if self._active_job and self._active_job.video == mapped_video:
                return self._active_job
            for qj in self._queued_jobs:
                if qj.video == mapped_video:
                    return qj

            job = JobRecord(
                id=uuid.uuid4().hex[:10],
                source=spec.source,
                video=mapped_video,
                title=spec.title,
                season=spec.season,
                episode=spec.episode,
                language=spec.language or self.config.language,
                refresh=spec.refresh,
                skip_embedded=spec.skip_embedded,
                in_place=(
                    spec.in_place
                    if spec.in_place is not None
                    else self.config.in_place
                ),
            )
            self._queued_jobs.append(job)
            self._queue.put(job)
            logger.info(
                "Enqueued job %s [%s] for %s", job.id, job.source, job.video
            )
            return job

    def process_next_sync(self, timeout: float = 0.1) -> Optional[JobRecord]:
        """Process a single queued job synchronously (useful for unit tests)."""
        try:
            job = self._queue.get(timeout=timeout)
        except queue.Empty:
            return None
        if job is None:
            return None
        self._execute_job(job)
        self._queue.task_done()
        return job

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                job = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if job is None:
                break
            try:
                self._execute_job(job)
            except Exception as exc:  # noqa: BLE001
                logger.exception("Unhandled error in worker loop: %s", exc)
            finally:
                self._queue.task_done()

    def _execute_job(self, job: JobRecord) -> None:
        with self._lock:
            if job in self._queued_jobs:
                self._queued_jobs.remove(job)
            self._active_job = job
            job.status = "running"
            job.started_at = _utc_now_iso()

        t0 = time.monotonic()
        logger.info("Starting subcheck job %s on %s", job.id, job.video)
        try:
            res: AutoSubResult = self._autosub_fn(
                job.video,
                language=job.language,
                limit=self.config.limit,
                refresh=job.refresh,
                skip_embedded=job.skip_embedded,
                title=job.title,
                season=job.season,
                episode=job.episode,
                snippets=self.config.snippets,
                duration=self.config.duration,
                min_gap=self.config.min_gap,
                model=self.config.model,
                device=self.config.device,
                compute_type=self.config.compute_type,
                download_root=self.config.model_dir,
                beam_size=self.config.beam_size,
                embed=True,
                overwrite=job.in_place,
            )
            meta = res.meta or {}
            if not job.title and meta.get("title"):
                job.title = str(meta["title"])
            if job.season is None and meta.get("season") is not None:
                job.season = int(meta["season"])
            if job.episode is None and meta.get("episode") is not None:
                job.episode = int(meta["episode"])

            job.embed_status = res.embed_status
            job.embed_out = res.embed_out
            job.downloads_new = res.downloads_new
            job.cache_hits = res.cache_hits
            job.quota = res.quota
            job.notes = list(res.notes)
            job.error = res.error

            if res.chosen is not None:
                c = res.chosen
                job.chosen_source = (
                    c.release or c.file_name or f"file #{c.file_id}"
                )
                job.score_before = c.composite
                job.score_after = c.composite_after
                job.fix_mode = c.best_mode
                job.fix_by = c.best_by
                job.fix_scale = c.best_scale

            if res.error:
                job.status = "failed"
                logger.warning(
                    "Job %s failed on %s: %s", job.id, job.video, res.error
                )
            else:
                job.status = "completed"
                logger.info(
                    "Job %s completed on %s: chosen=%s score=%.1f%%->%.1f%% embed=%s",
                    job.id,
                    job.video,
                    job.chosen_source,
                    job.score_before or 0.0,
                    job.score_after or 0.0,
                    job.embed_status,
                )
                if self.config.jellyfin_url and self.config.jellyfin_api_key:
                    ok, msg = notify_jellyfin(
                        self.config.jellyfin_url,
                        self.config.jellyfin_api_key,
                        job.embed_out or job.video,
                        self.config.jellyfin_path_maps,
                    )
                    job.jellyfin_status = msg
                    if not ok:
                        logger.warning(
                            "Jellyfin refresh notification failed for %s: %s",
                            job.video,
                            msg,
                        )
        except Exception as exc:  # noqa: BLE001
            job.status = "failed"
            job.error = str(exc)
            logger.exception("Job %s raised exception on %s", job.id, job.video)
        finally:
            job.duration_s = round(time.monotonic() - t0, 2)
            job.finished_at = _utc_now_iso()
            with self._lock:
                if job.quota:
                    self._last_quota = job.quota
                self._active_job = None
                self._history.insert(0, job)
                if len(self._history) > self.config.max_history:
                    self._history = self._history[: self.config.max_history]
                self._save_history()

    def auto_wire_integrations(self) -> Dict[str, str]:
        """Register webhooks in Sonarr and Radarr if configured."""
        status: Dict[str, str] = {}
        base_self = self.config.self_url.rstrip("/")
        if self.config.sonarr_url and self.config.sonarr_api_key:
            _, msg = auto_register_arr_webhook(
                "Sonarr",
                self.config.sonarr_url,
                self.config.sonarr_api_key,
                f"{base_self}/webhook/sonarr",
            )
            status["sonarr"] = msg
            logger.info("Sonarr integration: %s", msg)
        if self.config.radarr_url and self.config.radarr_api_key:
            _, msg = auto_register_arr_webhook(
                "Radarr",
                self.config.radarr_url,
                self.config.radarr_api_key,
                f"{base_self}/webhook/radarr",
            )
            status["radarr"] = msg
            logger.info("Radarr integration: %s", msg)
        if self.config.jellyfin_url and self.config.jellyfin_api_key:
            status["jellyfin"] = f"configured ({self.config.jellyfin_url})"
        with self._lock:
            self._integration_status = status
        return status

    def status_snapshot(self) -> Dict[str, Any]:
        with self._lock:
            completed = sum(1 for j in self._history if j.status == "completed")
            failed = sum(1 for j in self._history if j.status == "failed")
            return {
                "status": "ok",
                "version": __version__,
                "config": {
                    "language": self.config.language,
                    "model": self.config.model,
                    "device": self.config.device,
                    "snippets": self.config.snippets,
                    "limit": self.config.limit,
                    "in_place": self.config.in_place,
                    "browse_root": self.config.browse_root,
                },
                "integrations": dict(self._integration_status),
                "quota": self._last_quota,
                "queue_size": len(self._queued_jobs),
                "active_job": (
                    self._active_job.to_dict() if self._active_job else None
                ),
                "queued_jobs": [j.to_dict() for j in self._queued_jobs],
                "stats": {
                    "completed": completed,
                    "failed": failed,
                    "total": len(self._history),
                },
                "history": [j.to_dict() for j in self._history[:50]],
            }

    def stop(self) -> None:
        self._stop_event.set()
        self._queue.put(None)
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)


# --------------------------------------------------------------------------- #
# HTTP Handler & Dashboard UI


DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>subcheck — Subtitle Verification &amp; Auto-Fix</title>
<style>
  :root {
    --bg: #0f172a;
    --panel: #1e293b;
    --panel-alt: #334155;
    --text: #f8fafc;
    --muted: #94a3b8;
    --accent: #38bdf8;
    --ok: #22c55e;
    --warn: #f59e0b;
    --err: #ef4444;
    --border: #334155;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; padding: 24px;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: var(--bg); color: var(--text);
    line-height: 1.45;
  }
  .container { max-width: 1240px; margin: 0 auto; }
  header {
    display: flex; flex-wrap: wrap; justify-content: space-between;
    align-items: center; gap: 12px; margin-bottom: 20px;
    padding-bottom: 16px; border-bottom: 1px solid var(--border);
  }
  h1 { margin: 0; font-size: 1.4rem; display: flex; align-items: center; gap: 10px; }
  .version { font-size: 0.8rem; color: var(--accent); background: rgba(56,189,248,0.12); padding: 2px 8px; border-radius: 999px; }
  .cards {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
    gap: 12px; margin-bottom: 20px;
  }
  .card {
    background: var(--panel); border: 1px solid var(--border);
    border-radius: 10px; padding: 14px 16px;
  }
  .card-label { font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; color: var(--muted); }
  .card-val { font-size: 1.25rem; font-weight: 600; margin-top: 4px; }
  .card-sub { font-size: 0.8rem; color: var(--muted); margin-top: 2px; }
  .panel {
    background: var(--panel); border: 1px solid var(--border);
    border-radius: 10px; padding: 18px; margin-bottom: 20px;
  }
  .panel h2 { margin: 0 0 12px 0; font-size: 1.05rem; }
  .form-row {
    display: flex; flex-wrap: wrap; gap: 10px; align-items: center;
  }
  input[type="text"], input[type="number"], select {
    background: var(--bg); color: var(--text); border: 1px solid var(--border);
    border-radius: 6px; padding: 8px 12px; font-size: 0.9rem;
  }
  .input-path { flex: 1 1 380px; font-family: monospace; }
  button {
    background: var(--accent); color: #0f172a; font-weight: 600;
    border: none; border-radius: 6px; padding: 8px 16px;
    cursor: pointer; font-size: 0.9rem;
  }
  button.secondary {
    background: var(--panel-alt); color: var(--text);
  }
  button:hover { opacity: 0.92; }
  table { width: 100%; border-collapse: collapse; font-size: 0.88rem; }
  th, td { text-align: left; padding: 10px 8px; border-bottom: 1px solid var(--border); vertical-align: top; }
  th { color: var(--muted); font-weight: 600; font-size: 0.78rem; text-transform: uppercase; }
  .badge {
    display: inline-block; padding: 2px 8px; border-radius: 999px;
    font-size: 0.75rem; font-weight: 600;
  }
  .b-ok { background: rgba(34,197,94,0.15); color: #4ade80; }
  .b-run { background: rgba(56,189,248,0.18); color: #38bdf8; }
  .b-queue { background: rgba(245,158,11,0.18); color: #fbbf24; }
  .b-err { background: rgba(239,68,68,0.18); color: #f87171; }
  .mono { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 0.82rem; }
  .muted { color: var(--muted); }
  #browse-box {
    margin-top: 10px; max-height: 220px; overflow-y: auto;
    background: var(--bg); border: 1px solid var(--border); border-radius: 6px;
    padding: 8px; display: none;
  }
  .browse-item {
    padding: 4px 8px; cursor: pointer; border-radius: 4px;
    font-family: monospace; font-size: 0.83rem;
  }
  .browse-item:hover { background: var(--panel-alt); }
  #toast {
    margin-top: 10px; font-size: 0.85rem; color: var(--accent);
  }
</style>
</head>
<body>
<div class="container">
  <header>
    <h1>subcheck <span class="version" id="ver">v__VERSION__</span></h1>
    <div style="display:flex; gap:8px; align-items:center;">
      <button class="secondary" onclick="rewireArr()">Verify *arr Webhooks</button>
      <button class="secondary" onclick="refreshStatus()">Refresh</button>
    </div>
  </header>

  <div class="cards">
    <div class="card">
      <div class="card-label">Worker &amp; Queue</div>
      <div class="card-val" id="c-worker">Idle</div>
      <div class="card-sub" id="c-queue">0 queued</div>
    </div>
    <div class="card">
      <div class="card-label">Jobs Processed</div>
      <div class="card-val" id="c-stats">0 ok / 0 failed</div>
      <div class="card-sub" id="c-model">model: -</div>
    </div>
    <div class="card">
      <div class="card-label">OpenSubtitles Quota</div>
      <div class="card-val" id="c-quota">No API session yet</div>
      <div class="card-sub" id="c-quota-sub">Embedded tracks use 0 quota</div>
    </div>
    <div class="card">
      <div class="card-label">Integrations</div>
      <div class="card-val" id="c-integ">Ready</div>
      <div class="card-sub" id="c-integ-sub">/webhook/sonarr &bull; /webhook/radarr</div>
    </div>
  </div>

  <div class="panel">
    <h2>Verify / Fix Media File or Folder</h2>
    <div class="form-row">
      <input type="text" id="f-path" class="input-path" placeholder="/data/media/movies/... or /data/media/tv/..." value="/data/media">
      <button class="secondary" type="button" onclick="toggleBrowse()">Browse</button>
      <input type="text" id="f-lang" style="width:72px" value="en" title="Language code">
      <input type="text" id="f-title" style="width:170px" placeholder="Title override (opt)">
      <label style="font-size:0.85rem; display:flex; align-items:center; gap:5px;">
        <input type="checkbox" id="f-refresh"> Force OpenSubtitles
      </label>
      <button type="button" onclick="submitJob()">Queue Subcheck</button>
    </div>
    <div id="browse-box"></div>
    <div id="toast"></div>
  </div>

  <div class="panel">
    <h2>Recent &amp; Queued Jobs</h2>
    <table>
      <thead>
        <tr>
          <th>Status</th>
          <th>Source</th>
          <th>Media</th>
          <th>Chosen Track</th>
          <th>Score (Before &rarr; After)</th>
          <th>Fix / Action</th>
          <th>Time</th>
        </tr>
      </thead>
      <tbody id="jobs-body">
        <tr><td colspan="7" class="muted">Loading...</td></tr>
      </tbody>
    </table>
  </div>
</div>

<script>
let browseOpen = false;

function esc(s) {
  if (s === null || s === undefined) return '';
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

function badge(status) {
  if (status === 'completed') return '<span class="badge b-ok">completed</span>';
  if (status === 'running') return '<span class="badge b-run">running</span>';
  if (status === 'queued') return '<span class="badge b-queue">queued</span>';
  return '<span class="badge b-err">' + esc(status) + '</span>';
}

async function refreshStatus() {
  try {
    const r = await fetch('/api/status');
    const d = await r.json();
    document.getElementById('ver').textContent = 'v' + d.version;
    const active = d.active_job;
    document.getElementById('c-worker').textContent = active ? ('Running: ' + active.video.split('/').pop()) : 'Idle';
    document.getElementById('c-queue').textContent = d.queue_size + ' job(s) in queue';
    document.getElementById('c-stats').textContent = d.stats.completed + ' ok / ' + d.stats.failed + ' failed';
    document.getElementById('c-model').textContent = d.config.model + ' (' + d.config.device + ', ' + d.config.snippets + 'x10s)';

    if (d.quota && d.quota.remaining_downloads !== undefined) {
      document.getElementById('c-quota').textContent = d.quota.remaining_downloads + ' / ' + d.quota.allowed_downloads + ' left';
      document.getElementById('c-quota-sub').textContent = (d.quota.username || 'user') + ' (' + (d.quota.level || 'User') + ')';
    }

    const integKeys = Object.keys(d.integrations || {});
    if (integKeys.length > 0) {
      document.getElementById('c-integ').textContent = integKeys.join(', ');
      document.getElementById('c-integ-sub').textContent = integKeys.map(k => k + ': ' + d.integrations[k]).join(' | ');
    }

    const allJobs = [];
    if (d.active_job) allJobs.push(d.active_job);
    (d.queued_jobs || []).forEach(j => allJobs.push(j));
    (d.history || []).forEach(j => allJobs.push(j));

    const tbody = document.getElementById('jobs-body');
    if (allJobs.length === 0) {
      tbody.innerHTML = '<tr><td colspan="7" class="muted">No jobs recorded yet. Import media via Sonarr/Radarr or queue a path above.</td></tr>';
      return;
    }

    tbody.innerHTML = allJobs.map(j => {
      const fileName = j.video.split('/').slice(-2).join('/');
      let metaStr = j.title || '';
      if (j.season !== null && j.season !== undefined && j.episode !== null && j.episode !== undefined) {
        metaStr += ' S' + String(j.season).padStart(2, '0') + 'E' + String(j.episode).padStart(2, '0');
      }
      const scoreB = (j.score_before !== null && j.score_before !== undefined) ? j.score_before.toFixed(1) + '%' : '-';
      const scoreA = (j.score_after !== null && j.score_after !== undefined) ? j.score_after.toFixed(1) + '%' : '-';
      let fixStr = j.fix_mode || '-';
      if (j.fix_mode && j.fix_mode !== 'as-is') {
        fixStr += ' (' + (j.fix_by >= 0 ? '+' : '') + j.fix_by.toFixed(2) + 's)';
      }
      const embedInfo = j.error ? ('<span style="color:#f87171">' + esc(j.error) + '</span>') : esc(j.embed_status || '');
      const dur = j.duration_s ? (j.duration_s.toFixed(1) + 's') : '-';
      return '<tr>' +
        '<td>' + badge(j.status) + '</td>' +
        '<td>' + esc(j.source) + '</td>' +
        '<td><div class="mono">' + esc(fileName) + '</div><div class="muted" style="font-size:0.78rem">' + esc(metaStr) + '</div></td>' +
        '<td>' + esc(j.chosen_source || '-') + '</td>' +
        '<td class="mono">' + scoreB + ' &rarr; <strong>' + scoreA + '</strong></td>' +
        '<td><div>' + esc(fixStr) + '</div><div class="muted" style="font-size:0.78rem">' + embedInfo + '</div></td>' +
        '<td class="mono">' + dur + '</td>' +
      '</tr>';
    }).join('');
  } catch (e) {
    console.error(e);
  }
}

async function submitJob() {
  const path = document.getElementById('f-path').value.trim();
  const language = document.getElementById('f-lang').value.trim() || 'en';
  const title = document.getElementById('f-title').value.trim() || null;
  const refresh = document.getElementById('f-refresh').checked;
  if (!path) return;
  const toast = document.getElementById('toast');
  toast.textContent = 'Queueing...';
  const r = await fetch('/api/autosub', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({video: path, language, title, refresh})
  });
  const d = await r.json();
  if (d.error) {
    toast.textContent = 'Error: ' + d.error;
  } else {
    toast.textContent = 'Queued ' + (d.queued ? d.queued.length : 1) + ' file(s).';
    refreshStatus();
  }
}

async function rewireArr() {
  const toast = document.getElementById('toast');
  toast.textContent = 'Checking Sonarr/Radarr webhook registration...';
  const r = await fetch('/api/rewire', {method: 'POST'});
  const d = await r.json();
  toast.textContent = JSON.stringify(d.integrations || d);
  refreshStatus();
}

async function toggleBrowse() {
  const box = document.getElementById('browse-box');
  if (browseOpen) {
    box.style.display = 'none';
    browseOpen = false;
    return;
  }
  browseOpen = true;
  box.style.display = 'block';
  await loadBrowse(document.getElementById('f-path').value.trim() || '/data/media');
}

async function loadBrowse(target) {
  const box = document.getElementById('browse-box');
  const r = await fetch('/api/browse?path=' + encodeURIComponent(target));
  const d = await r.json();
  if (d.error) {
    box.innerHTML = '<div class="muted">' + esc(d.error) + '</div>';
    return;
  }
  document.getElementById('f-path').value = d.current;
  let html = '';
  if (d.parent) {
    html += '<div class="browse-item" onclick="loadBrowse(\\'' + d.parent.replace(/'/g, "\\\\'") + '\\')">&#128193; .. (parent directory)</div>';
  }
  (d.entries || []).forEach(e => {
    const icon = e.is_dir ? '&#128193;' : '&#127916;';
    const p = e.path.replace(/'/g, "\\\\'");
    if (e.is_dir) {
      html += '<div class="browse-item" onclick="loadBrowse(\\'' + p + '\\')">' + icon + ' ' + esc(e.name) + '/</div>';
    } else {
      html += '<div class="browse-item" onclick="selectFile(\\'' + p + '\\')">' + icon + ' ' + esc(e.name) + '</div>';
    }
  });
  box.innerHTML = html || '<div class="muted">(empty)</div>';
}

function selectFile(p) {
  document.getElementById('f-path').value = p;
  document.getElementById('browse-box').style.display = 'none';
  browseOpen = false;
}

refreshStatus();
setInterval(refreshStatus, 4000);
</script>
</body>
</html>
"""


def make_handler(worker: SubcheckWorker):
    """Build a BaseHTTPRequestHandler class bound to *worker*."""

    class SubcheckHTTPHandler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:
            logger.info("%s - %s", self.address_string(), fmt % args)

        def _send_json(self, status: int, data: Dict[str, Any]) -> None:
            raw = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def _send_html(self, status: int, content: str) -> None:
            raw = content.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def _read_json_body(self) -> Dict[str, Any]:
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0:
                return {}
            raw = self.rfile.read(length).decode("utf-8", errors="replace")
            return json.loads(raw) if raw.strip() else {}

        def do_GET(self) -> None:  # noqa: N802
            parsed = urllib.parse.urlparse(self.path)
            route = parsed.path.rstrip("/") or "/"

            if route == "/healthz":
                snap = worker.status_snapshot()
                self._send_json(
                    HTTPStatus.OK,
                    {
                        "status": "ok",
                        "version": snap["version"],
                        "queue_size": snap["queue_size"],
                        "active": snap["active_job"] is not None,
                    },
                )
                return

            if route == "/api/status":
                self._send_json(HTTPStatus.OK, worker.status_snapshot())
                return

            if route == "/api/browse":
                qs = urllib.parse.parse_qs(parsed.query)
                req_path = (qs.get("path") or [worker.config.browse_root])[0]
                target = os.path.abspath(req_path)
                if os.path.isfile(target):
                    target = os.path.dirname(target)
                if not os.path.isdir(target):
                    self._send_json(
                        HTTPStatus.BAD_REQUEST,
                        {"error": f"Directory not found: {target}"},
                    )
                    return
                entries = []
                try:
                    for name in sorted(os.listdir(target)):
                        if name.startswith("."):
                            continue
                        full = os.path.join(target, name)
                        is_dir = os.path.isdir(full)
                        ext = os.path.splitext(name)[1].lower()
                        if is_dir or ext in VIDEO_EXTENSIONS:
                            entries.append(
                                {"name": name, "path": full, "is_dir": is_dir}
                            )
                except OSError as exc:
                    self._send_json(
                        HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)}
                    )
                    return
                parent = os.path.dirname(target) if target != "/" else None
                self._send_json(
                    HTTPStatus.OK,
                    {"current": target, "parent": parent, "entries": entries},
                )
                return

            if route == "/":
                page = DASHBOARD_HTML.replace(
                    "__VERSION__", html.escape(__version__)
                )
                self._send_html(HTTPStatus.OK, page)
                return

            self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            parsed = urllib.parse.urlparse(self.path)
            route = parsed.path.rstrip("/")

            try:
                payload = self._read_json_body()
            except Exception as exc:  # noqa: BLE001
                self._send_json(
                    HTTPStatus.BAD_REQUEST,
                    {"error": f"invalid JSON payload: {exc}"},
                )
                return

            if route == "/webhook/radarr":
                event_type, specs = parse_radarr_webhook(payload)
                if event_type == "Test":
                    self._send_json(
                        HTTPStatus.OK,
                        {
                            "status": "ok",
                            "event": "Test",
                            "message": "Radarr test webhook received by subcheck",
                        },
                    )
                    return
                jobs = [worker.enqueue(s).to_dict() for s in specs]
                self._send_json(
                    HTTPStatus.ACCEPTED if jobs else HTTPStatus.OK,
                    {"status": "ok", "event": event_type, "queued": jobs},
                )
                return

            if route == "/webhook/sonarr":
                event_type, specs = parse_sonarr_webhook(payload)
                if event_type == "Test":
                    self._send_json(
                        HTTPStatus.OK,
                        {
                            "status": "ok",
                            "event": "Test",
                            "message": "Sonarr test webhook received by subcheck",
                        },
                    )
                    return
                jobs = [worker.enqueue(s).to_dict() for s in specs]
                self._send_json(
                    HTTPStatus.ACCEPTED if jobs else HTTPStatus.OK,
                    {"status": "ok", "event": event_type, "queued": jobs},
                )
                return

            if route == "/webhook/jellyfin":
                event_type, specs = parse_jellyfin_webhook(payload)
                if event_type == "Test":
                    self._send_json(
                        HTTPStatus.OK,
                        {
                            "status": "ok",
                            "event": "Test",
                            "message": "Jellyfin test webhook received by subcheck",
                        },
                    )
                    return
                jobs = [worker.enqueue(s).to_dict() for s in specs]
                self._send_json(
                    HTTPStatus.ACCEPTED if jobs else HTTPStatus.OK,
                    {"status": "ok", "event": event_type, "queued": jobs},
                )
                return

            if route == "/api/autosub":
                video_raw = str(payload.get("video") or payload.get("path") or "").strip()
                if not video_raw:
                    self._send_json(
                        HTTPStatus.BAD_REQUEST,
                        {"error": "missing 'video' path in request body"},
                    )
                    return
                mapped_target = apply_path_map(video_raw, worker.config.path_maps)
                targets = expand_video_targets(mapped_target)
                if not targets:
                    self._send_json(
                        HTTPStatus.BAD_REQUEST,
                        {"error": f"no video files found in {mapped_target!r}"},
                    )
                    return

                queued = []
                for t in targets:
                    spec = WebhookSpec(
                        source=str(payload.get("source") or "manual"),
                        video=t,
                        title=(
                            str(payload["title"])
                            if payload.get("title")
                            else None
                        ),
                        season=(
                            int(payload["season"])
                            if payload.get("season") is not None
                            else None
                        ),
                        episode=(
                            int(payload["episode"])
                            if payload.get("episode") is not None
                            else None
                        ),
                        language=(
                            str(payload["language"])
                            if payload.get("language")
                            else None
                        ),
                        refresh=bool(payload.get("refresh", False)),
                        skip_embedded=bool(payload.get("skip_embedded", False)),
                        in_place=(
                            bool(payload["in_place"])
                            if "in_place" in payload
                            else None
                        ),
                    )
                    queued.append(worker.enqueue(spec).to_dict())

                self._send_json(
                    HTTPStatus.ACCEPTED,
                    {"status": "queued", "queued": queued},
                )
                return

            if route == "/api/rewire":
                status = worker.auto_wire_integrations()
                self._send_json(
                    HTTPStatus.OK,
                    {"status": "ok", "integrations": status},
                )
                return

            self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})

    return SubcheckHTTPHandler


def run_server(config: ServerConfig) -> int:
    """Start the HTTP webhook & dashboard server and block until interrupted."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    worker = SubcheckWorker(config, start_thread=True)

    handler_cls = make_handler(worker)
    httpd = ThreadingHTTPServer((config.host, config.port), handler_cls)

    if config.auto_register:
        def _delayed_autowire() -> None:
            for delay in (2.0, 5.0, 10.0):
                if worker._stop_event.wait(delay):
                    return
                res = worker.auto_wire_integrations()
                if all("failed" not in v.lower() for v in res.values()):
                    break

        threading.Thread(
            target=_delayed_autowire, name="subcheck-autowire", daemon=True
        ).start()

    logger.info(
        "subcheck %s server listening on http://%s:%d (model=%s, device=%s, in_place=%s)",
        __version__,
        config.host,
        config.port,
        config.model,
        config.device,
        config.in_place,
    )
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down subcheck server...")
    finally:
        httpd.server_close()
        worker.stop()
    return 0
