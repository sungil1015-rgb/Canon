"""SQLite storage helpers for sequential video validation results."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.core.paths import DB_DIR


DEFAULT_DB_PATH = DB_DIR / "sequence_runs.sqlite3"


SCHEMA_SQL = """
PRAGMA journal_mode = WAL;

CREATE TABLE IF NOT EXISTS sequence_run (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_timestamp TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL,
    run_root TEXT,
    summary_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sequence_run_video (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL,
    video_name TEXT NOT NULL,
    video_path TEXT NOT NULL,
    output_dir TEXT NOT NULL,
    completed INTEGER NOT NULL,
    processed_frames INTEGER NOT NULL,
    confirmed_targets INTEGER NOT NULL,
    total_detections INTEGER NOT NULL,
    video_json TEXT NOT NULL,
    FOREIGN KEY(run_id) REFERENCES sequence_run(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS sequence_run_target (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id INTEGER NOT NULL,
    target_index INTEGER NOT NULL,
    target_name TEXT NOT NULL,
    completed INTEGER NOT NULL,
    start_frame INTEGER,
    confirmed_frame INTEGER,
    processed_frames INTEGER NOT NULL,
    detections_seen INTEGER NOT NULL,
    target_json TEXT NOT NULL,
    FOREIGN KEY(video_id) REFERENCES sequence_run_video(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_sequence_run_timestamp ON sequence_run(run_timestamp);
CREATE INDEX IF NOT EXISTS idx_sequence_run_video_run_id ON sequence_run_video(run_id);
CREATE INDEX IF NOT EXISTS idx_sequence_run_target_video_id ON sequence_run_target(video_id);
"""


def ensure_dir(path: Path) -> None:
	path.mkdir(parents=True, exist_ok=True)


def connect(db_path: Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
	ensure_dir(db_path.parent)
	connection = sqlite3.connect(db_path)
	connection.row_factory = sqlite3.Row
	connection.execute("PRAGMA foreign_keys = ON;")
	return connection


def initialize(db_path: Path = DEFAULT_DB_PATH) -> None:
	with connect(db_path) as connection:
		connection.executescript(SCHEMA_SQL)


def _json_dumps(payload: Any) -> str:
	return json.dumps(payload, ensure_ascii=False, indent=2)


def upsert_sequence_run(result: dict[str, object], db_path: Path = DEFAULT_DB_PATH) -> int:
	"""Store one full sequence run result and return its database id."""
	initialize(db_path)
	with connect(db_path) as connection:
		summary = result.get("summary", {})
		run_timestamp = str(result.get("run_timestamp") or summary.get("run_timestamp") or datetime.now(timezone.utc).isoformat())
		run_root = str(result.get("run_root") or "")
		connection.execute(
			"""
			INSERT INTO sequence_run (run_timestamp, created_at, run_root, summary_json)
			VALUES (?, ?, ?, ?)
			ON CONFLICT(run_timestamp) DO UPDATE SET
				created_at = excluded.created_at,
				run_root = excluded.run_root,
				summary_json = excluded.summary_json
			""",
			(run_timestamp, datetime.now(timezone.utc).isoformat(), run_root, _json_dumps(summary)),
		)
		run_id = connection.execute("SELECT id FROM sequence_run WHERE run_timestamp = ?", (run_timestamp,)).fetchone()["id"]
		connection.execute("DELETE FROM sequence_run_video WHERE run_id = ?", (run_id,))
		for video_index, video in enumerate(result.get("videos", [])):
			video_json = _json_dumps(video)
			cursor = connection.execute(
				"""
				INSERT INTO sequence_run_video (
					run_id, video_name, video_path, output_dir, completed,
					processed_frames, confirmed_targets, total_detections, video_json
				)
				VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
				""",
				(
					run_id,
					str(video.get("video_name", "")),
					str(video.get("video_path", "")),
					str(video.get("output_dir", "")),
					1 if bool(video.get("completed")) else 0,
					int(video.get("processed_frames", 0)),
					int(video.get("confirmed_targets", 0)),
					int(video.get("total_detections", 0)),
					video_json,
				),
			)
			video_id = int(cursor.lastrowid)
			connection.execute("DELETE FROM sequence_run_target WHERE video_id = ?", (video_id,))
			for target_index, target in enumerate(video.get("targets", [])):
				connection.execute(
					"""
					INSERT INTO sequence_run_target (
						video_id, target_index, target_name, completed, start_frame, confirmed_frame,
						processed_frames, detections_seen, target_json
					)
					VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
					""",
					(
						video_id,
						target_index,
						str(target.get("target_name", "")),
						1 if bool(target.get("completed")) else 0,
						target.get("start_frame"),
						target.get("confirmed_frame"),
						int(target.get("processed_frames", 0)),
						int(target.get("detections_seen", 0)),
						_json_dumps(target),
					),
				)
		connection.commit()
		return int(run_id)


@dataclass(slots=True)
class StoredSequenceRun:
	id: int
	run_timestamp: str
	run_root: str
	summary_json: str


__all__ = [
	"DEFAULT_DB_PATH",
	"SCHEMA_SQL",
	"StoredSequenceRun",
	"connect",
	"initialize",
	"upsert_sequence_run",
]