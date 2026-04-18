from __future__ import annotations

import argparse
import csv
import importlib
import io
import os
import smtplib
import sqlite3
import pymysql
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from email.message import EmailMessage
from functools import wraps
from typing import Any

import cv2
import numpy as np
from flask import (
    Flask,
    Response,
    abort,
    flash,
    g,
    jsonify,
    redirect,
    render_template,
    request,
    session,
    url_for,
)
from werkzeug.security import check_password_hash, generate_password_hash

sys.path.insert(0, os.path.dirname(__file__))
app_module = importlib.import_module("app")

try:
    api_module = importlib.import_module("api")
    InferenceSession = api_module.InferenceSession
    HAS_API = True
except Exception as exc:  # pragma: no cover - import depends on optional local env
    InferenceSession = None
    HAS_API = False
    API_IMPORT_ERROR = str(exc)

HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(HERE, "templates")
DB_PATH = os.environ.get("PLITHOS_DB_PATH", os.path.join(HERE, "plithos.db"))
RDS_HOST = "plithos-db.cu9pse8tlstt.us-east-1.rds.amazonaws.com"
RDS_PORT = 3306
RDS_DB = "plithos"
RDS_USER = "admin"
RDS_PASSWORD = "PUT_PASSWORD_HERE"
DB_BACKEND = os.environ.get("PLITHOS_DB_BACKEND", "auto").strip().lower()
ALLOW_SQLITE_FALLBACK = os.environ.get("PLITHOS_ALLOW_SQLITE_FALLBACK", "1").strip().lower() not in {"0", "false", "no", "off"}
_ACTIVE_DB_BACKEND = "sqlite"
JPEG_QUALITY = 64
ACTIVE_SLEEP = 0.01
IDLE_SLEEP = 0.12
EMAIL_RATE_LIMIT_SECONDS = 60
CAMERA_SCAN_MAX = int(os.environ.get("PLITHOS_CAMERA_SCAN_MAX", "6") or 6)
CAMERA_SCAN_CACHE_SECONDS = 8.0
PREVIEW_CACHE_SECONDS = 2.5
MAX_INFERENCE_FPS = 20.0
MAX_RECENT_ALERTS = 50
MAX_LOG_ROWS = 250
CAMERA_FRAME_WIDTH = 640
CAMERA_FRAME_HEIGHT = 360
TARGET_CAMERA_FPS = 30
INCIDENT_UPDATE_INTERVAL_SECONDS = 1.0
INCIDENT_END_GRACE_SECONDS = {"violence": 4.0, "fire": 6.0, "safety": 4.0}
IDLE_INFERENCE_INTERVAL_SECONDS = 1.0
MAX_LOG_EXPORT_ROWS = 5000

flask_app = Flask(__name__, template_folder=TEMPLATES_DIR, static_folder=HERE)
flask_app.secret_key = os.environ.get("PLITHOS_SECRET_KEY", "plithos-dev-secret")
flask_app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
)

registry_lock = threading.Lock()
camera_runtimes: dict[int, "CameraRuntime"] = {}
camera_scan_lock = threading.Lock()
camera_scan_cache: dict[str, Any] = {"checked_at": 0.0, "items": []}
preview_cache_lock = threading.Lock()
preview_frame_cache: dict[str, dict[str, Any]] = {}


def now_local() -> datetime:
    return datetime.now().astimezone()


def now_iso() -> str:
    return now_local().isoformat(timespec="seconds")


def parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def human_time(value: str | None) -> str:
    parsed = parse_iso(value)
    return parsed.strftime("%Y-%m-%d %H:%M:%S") if parsed else "--"


def incident_duration_between(start_value: str | None, end_value: str | None) -> int:
    start_at = parse_iso(start_value)
    end_at = parse_iso(end_value)
    if not start_at or not end_at:
        return 0
    return max(0, int((end_at - start_at).total_seconds()))


def _row_value(row: sqlite3.Row | dict[str, Any], key: str, default: Any = None) -> Any:
    if isinstance(row, sqlite3.Row):
        return row[key] if key in row.keys() else default
    return row.get(key, default)


def incident_duration_from_row(row: sqlite3.Row | dict[str, Any]) -> int:
    duration_value = _row_value(row, "duration_seconds")
    if duration_value:
        return int(duration_value)
    started_at = _row_value(row, "started_at")
    fallback_created_at = _row_value(row, "created_at")
    ended_at = _row_value(row, "ended_at")
    last_seen_at = _row_value(row, "last_seen_at")
    status = _row_value(row, "status")
    end_value = ended_at or last_seen_at or (now_iso() if status == "active" else fallback_created_at)
    return incident_duration_between(started_at or fallback_created_at, end_value)


def human_duration(seconds: int | float | None) -> str:
    total_seconds = max(0, int(seconds or 0))
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"



class CompatRow(dict):
    def __getitem__(self, key):
        if isinstance(key, int):
            return list(self.values())[key]
        return super().__getitem__(key)

    def keys(self):
        return super().keys()


class MySQLCursorCompat:
    def __init__(self, cursor=None, rows=None, lastrowid=None):
        self._cursor = cursor
        self._rows = rows
        self.lastrowid = lastrowid if lastrowid is not None else getattr(cursor, "lastrowid", None)

    def fetchone(self):
        if self._rows is not None:
            if not self._rows:
                return None
            return CompatRow(self._rows[0])
        row = self._cursor.fetchone() if self._cursor else None
        return CompatRow(row) if row else None

    def fetchall(self):
        if self._rows is not None:
            return [CompatRow(r) for r in self._rows]
        rows = self._cursor.fetchall() if self._cursor else []
        return [CompatRow(r) for r in rows]

    def close(self):
        if self._cursor:
            self._cursor.close()


class MySQLConnectionCompat:
    def __init__(self, conn):
        self._conn = conn
        self.row_factory = sqlite3.Row

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type:
            self._conn.rollback()
        else:
            self._conn.commit()
        self._conn.close()

    def commit(self):
        self._conn.commit()

    def close(self):
        self._conn.close()

    def executescript(self, script: str):
        statements = [s.strip() for s in script.split(";") if s.strip()]
        last = None
        for statement in statements:
            last = self.execute(statement)
        return last

    def execute(self, query: str, params=()):
        q = query.strip()
        upper = q.upper()

        if upper.startswith("PRAGMA FOREIGN_KEYS"):
            return MySQLCursorCompat(rows=[])
        if upper.startswith("PRAGMA JOURNAL_MODE"):
            return MySQLCursorCompat(rows=[])
        if upper.startswith("PRAGMA TABLE_INFO("):
            table_name = q[q.find("(") + 1:q.rfind(")")].strip().strip("'"`")
            sql = """
                SELECT COLUMN_NAME AS name
                FROM INFORMATION_SCHEMA.COLUMNS
                WHERE TABLE_SCHEMA = DATABASE()
                  AND TABLE_NAME = %s
                ORDER BY ORDINAL_POSITION
            """
            cur = self._conn.cursor(pymysql.cursors.DictCursor)
            cur.execute(sql, (table_name,))
            return MySQLCursorCompat(cur)

        if upper.startswith("SELECT LAST_INSERT_ROWID()"):
            cur = self._conn.cursor(pymysql.cursors.DictCursor)
            cur.execute("SELECT LAST_INSERT_ID() AS last_insert_rowid")
            return MySQLCursorCompat(cur)

        translated = query.replace("?", "%s")
        cur = self._conn.cursor(pymysql.cursors.DictCursor)
        cur.execute(translated, params)
        return MySQLCursorCompat(cur)


def _sqlite_db_connect() -> sqlite3.Connection:
    db_dir = os.path.dirname(os.path.abspath(DB_PATH))
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)
    journal_modes = ("WAL", "DELETE")
    last_error: Exception | None = None
    for journal_mode in journal_modes:
        conn = sqlite3.connect(DB_PATH, timeout=30, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            conn.execute(f"PRAGMA journal_mode = {journal_mode}")
            return conn
        except sqlite3.OperationalError as exc:
            last_error = exc
            conn.close()
    conn = sqlite3.connect(DB_PATH, timeout=30, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    if last_error:
        print(f"[WARN] SQLite journal mode fallback in use: {last_error}")
    return conn


def _mysql_db_connect():
    conn = pymysql.connect(
        host=RDS_HOST,
        port=RDS_PORT,
        user=RDS_USER,
        password=RDS_PASSWORD,
        database=RDS_DB,
        connect_timeout=8,
        read_timeout=8,
        write_timeout=8,
        autocommit=False,
        cursorclass=pymysql.cursors.DictCursor,
    )
    return MySQLConnectionCompat(conn)


def active_db_backend() -> str:
    return _ACTIVE_DB_BACKEND


def db_connect():
    global _ACTIVE_DB_BACKEND

    prefer_mysql = DB_BACKEND in {"auto", "mysql", "rds"}
    if prefer_mysql:
        try:
            conn = _mysql_db_connect()
            _ACTIVE_DB_BACKEND = "mysql"
            return conn
        except pymysql.MySQLError as exc:
            if DB_BACKEND in {"mysql", "rds"} and not ALLOW_SQLITE_FALLBACK:
                raise
            if not ALLOW_SQLITE_FALLBACK:
                raise
            print(f"[WARN] RDS unreachable ({exc}) — falling back to local SQLite database: {DB_PATH}")

    _ACTIVE_DB_BACKEND = "sqlite"
    return _sqlite_db_connect()


def init_db() -> None:
    with db_connect() as conn:

        if active_db_backend() == "mysql":
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    full_name TEXT NOT NULL,
                    email VARCHAR(255) NOT NULL UNIQUE,
                    password_hash TEXT NOT NULL,
                    theme VARCHAR(32) NOT NULL DEFAULT 'dark',
                    sound_enabled BOOLEAN NOT NULL DEFAULT 1,
                    auto_switch_alerts BOOLEAN NOT NULL DEFAULT 1,
                    notifications_enabled BOOLEAN NOT NULL DEFAULT 1,
                    violence_enabled BOOLEAN NOT NULL DEFAULT 1,
                    fire_enabled BOOLEAN NOT NULL DEFAULT 1,
                    safety_enabled BOOLEAN NOT NULL DEFAULT 1,
                    smtp_host TEXT DEFAULT '',
                    smtp_port INT NOT NULL DEFAULT 587,
                    smtp_username TEXT DEFAULT '',
                    smtp_password TEXT DEFAULT '',
                    smtp_sender TEXT DEFAULT '',
                    smtp_use_tls BOOLEAN NOT NULL DEFAULT 1,
                    report_email TEXT DEFAULT '',
                    alert_subject TEXT DEFAULT '',
                    alert_message TEXT DEFAULT '',
                    last_email_sent_at TEXT,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS cameras (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    user_id INT NOT NULL,
                    name TEXT NOT NULL,
                    source TEXT NOT NULL,
                    enabled BOOLEAN NOT NULL DEFAULT 1,
                    sort_order INT NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS alerts (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    user_id INT NOT NULL,
                    camera_id INT NOT NULL,
                    camera_name TEXT NOT NULL,
                    event_type VARCHAR(64) NOT NULL,
                    severity VARCHAR(64) NOT NULL,
                    detail TEXT NOT NULL,
                    confidence FLOAT,
                    max_confidence FLOAT,
                    persons INT,
                    status VARCHAR(64) NOT NULL DEFAULT 'resolved',
                    started_at TEXT,
                    ended_at TEXT,
                    last_seen_at TEXT,
                    duration_seconds INT NOT NULL DEFAULT 0,
                    event_count INT NOT NULL DEFAULT 1,
                    evidence_url TEXT DEFAULT '',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                    FOREIGN KEY (camera_id) REFERENCES cameras(id) ON DELETE CASCADE
                );
                """
            )
        else:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    full_name TEXT NOT NULL,
                    email TEXT NOT NULL UNIQUE,
                    password_hash TEXT NOT NULL,
                    theme TEXT NOT NULL DEFAULT 'dark',
                    sound_enabled INTEGER NOT NULL DEFAULT 1,
                    auto_switch_alerts INTEGER NOT NULL DEFAULT 1,
                    notifications_enabled INTEGER NOT NULL DEFAULT 1,
                    violence_enabled INTEGER NOT NULL DEFAULT 1,
                    fire_enabled INTEGER NOT NULL DEFAULT 1,
                    safety_enabled INTEGER NOT NULL DEFAULT 1,
                    smtp_host TEXT DEFAULT '',
                    smtp_port INTEGER NOT NULL DEFAULT 587,
                    smtp_username TEXT DEFAULT '',
                    smtp_password TEXT DEFAULT '',
                    smtp_sender TEXT DEFAULT '',
                    smtp_use_tls INTEGER NOT NULL DEFAULT 1,
                    report_email TEXT DEFAULT '',
                    alert_subject TEXT DEFAULT '',
                    alert_message TEXT DEFAULT '',
                    last_email_sent_at TEXT,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS cameras (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    name TEXT NOT NULL,
                    source TEXT NOT NULL,
                    enabled INTEGER NOT NULL DEFAULT 1,
                    sort_order INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    camera_id INTEGER NOT NULL REFERENCES cameras(id) ON DELETE CASCADE,
                    camera_name TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    detail TEXT NOT NULL,
                    confidence REAL,
                    max_confidence REAL,
                    persons INTEGER,
                    status TEXT NOT NULL DEFAULT 'resolved',
                    started_at TEXT,
                    ended_at TEXT,
                    last_seen_at TEXT,
                    duration_seconds INTEGER NOT NULL DEFAULT 0,
                    event_count INTEGER NOT NULL DEFAULT 1,
                    evidence_url TEXT DEFAULT '',
                    created_at TEXT NOT NULL
                );
                """
            )
        existing_columns = {str(row["name"]) for row in conn.execute("PRAGMA table_info(users)").fetchall()}
        required_columns = {
            "report_email": "TEXT DEFAULT ''",
            "alert_subject": "TEXT DEFAULT ''",
            "alert_message": "TEXT DEFAULT ''",
            "violence_enabled": "INTEGER NOT NULL DEFAULT 1",
            "fire_enabled": "INTEGER NOT NULL DEFAULT 1",
            "safety_enabled": "INTEGER NOT NULL DEFAULT 1",
        }
        for column, ddl in required_columns.items():
            if column not in existing_columns:
                conn.execute(f"ALTER TABLE users ADD COLUMN {column} {ddl}")
        existing_alert_columns = {str(row["name"]) for row in conn.execute("PRAGMA table_info(alerts)").fetchall()}
        required_alert_columns = {
            "max_confidence": "REAL",
            "status": "TEXT NOT NULL DEFAULT 'resolved'",
            "started_at": "TEXT",
            "ended_at": "TEXT",
            "last_seen_at": "TEXT",
            "duration_seconds": "INTEGER NOT NULL DEFAULT 0",
            "event_count": "INTEGER NOT NULL DEFAULT 1",
            "evidence_url": "TEXT DEFAULT ''",
        }
        for column, ddl in required_alert_columns.items():
            if column not in existing_alert_columns:
                conn.execute(f"ALTER TABLE alerts ADD COLUMN {column} {ddl}")
        conn.execute(
            """
            UPDATE users
            SET report_email = COALESCE(NULLIF(report_email, ''), email),
                alert_subject = COALESCE(alert_subject, ''),
                alert_message = COALESCE(alert_message, ''),
                violence_enabled = COALESCE(violence_enabled, 1),
                fire_enabled = COALESCE(fire_enabled, 1),
                safety_enabled = COALESCE(safety_enabled, 1)
            """
        )
        conn.execute("UPDATE alerts SET started_at = COALESCE(started_at, created_at)")
        conn.execute("UPDATE alerts SET last_seen_at = COALESCE(last_seen_at, ended_at, started_at, created_at)")
        conn.execute("UPDATE alerts SET max_confidence = COALESCE(max_confidence, confidence)")
        conn.execute("UPDATE alerts SET event_count = COALESCE(NULLIF(event_count, 0), 1)")
        conn.execute("UPDATE alerts SET status = CASE WHEN COALESCE(status, '') = '' THEN 'resolved' ELSE status END")
        conn.execute("UPDATE alerts SET evidence_url = COALESCE(evidence_url, '')")
        conn.execute(
            """
            UPDATE alerts
            SET ended_at = COALESCE(ended_at, last_seen_at, started_at, created_at)
            WHERE status != 'active'
            """
        )
        active_rows = conn.execute(
            "SELECT id, started_at, last_seen_at, created_at FROM alerts WHERE status = 'active'"
        ).fetchall()
        for row in active_rows:
            ended_at = str(row["last_seen_at"] or row["started_at"] or row["created_at"] or now_iso())
            started_at = str(row["started_at"] or row["created_at"] or ended_at)
            conn.execute(
                """
                UPDATE alerts
                SET status = 'resolved',
                    ended_at = ?,
                    duration_seconds = ?
                WHERE id = ?
                """,
                (ended_at, incident_duration_between(started_at, ended_at), int(row["id"])),
            )
        conn.commit()


def fetch_one(query: str, params: tuple[Any, ...] = ()) -> sqlite3.Row | None:
    with db_connect() as conn:
        return conn.execute(query, params).fetchone()


def fetch_all(query: str, params: tuple[Any, ...] = ()) -> list[sqlite3.Row]:
    with db_connect() as conn:
        return conn.execute(query, params).fetchall()


def execute_db(query: str, params: tuple[Any, ...] = ()) -> int:
    with db_connect() as conn:
        cur = conn.execute(query, params)
        conn.commit()
        return cur.lastrowid


def count_users() -> int:
    row = fetch_one("SELECT COUNT(*) AS count FROM users")
    return int(row["count"]) if row else 0


def get_user_by_id(user_id: int | None) -> sqlite3.Row | None:
    if not user_id:
        return None
    return fetch_one("SELECT * FROM users WHERE id = ?", (user_id,))


def get_user_by_email(email: str) -> sqlite3.Row | None:
    return fetch_one("SELECT * FROM users WHERE lower(email) = lower(?)", (email.strip(),))


def cameras_for_user(user_id: int) -> list[sqlite3.Row]:
    return fetch_all(
        "SELECT * FROM cameras WHERE user_id = ? AND enabled = 1 ORDER BY sort_order, id",
        (user_id,),
    )


def user_has_cameras(user_id: int) -> bool:
    row = fetch_one("SELECT COUNT(*) AS count FROM cameras WHERE user_id = ? AND enabled = 1", (user_id,))
    return bool(row and row["count"])


def alerts_for_user(
    user_id: int,
    *,
    limit: int | None = MAX_RECENT_ALERTS,
    camera_id: int | None = None,
    event_type: str | None = None,
    search: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
) -> list[sqlite3.Row]:
    clauses = ["user_id = ?", "event_type IN ('violence', 'fire', 'safety')"]
    params: list[Any] = [user_id]
    sort_field = "COALESCE(started_at, created_at)"
    if camera_id:
        clauses.append("camera_id = ?")
        params.append(camera_id)
    if event_type and event_type != "all":
        clauses.append("event_type = ?")
        params.append(event_type)
    if search:
        clauses.append("(camera_name LIKE ? OR detail LIKE ?)")
        token = f"%{search.strip()}%"
        params.extend([token, token])
    if date_from:
        clauses.append(f"{sort_field} >= ?")
        params.append(f"{date_from}T00:00:00")
    if date_to:
        clauses.append(f"{sort_field} <= ?")
        params.append(f"{date_to}T23:59:59")
    query = f"""
        SELECT * FROM alerts
        WHERE {' AND '.join(clauses)}
        ORDER BY {sort_field} DESC
    """
    if limit is not None:
        query += "\n        LIMIT ?"
        params.append(limit)
    return fetch_all(query, tuple(params))


def total_alerts_for_user(user_id: int) -> int:
    row = fetch_one(
        "SELECT COUNT(*) AS count FROM alerts WHERE user_id = ? AND event_type IN ('violence', 'fire', 'safety')",
        (user_id,),
    )
    return int(row["count"]) if row else 0


def clear_alerts_for_user(user_id: int) -> None:
    with db_connect() as conn:
        conn.execute("DELETE FROM alerts WHERE user_id = ?", (user_id,))
        conn.commit()


def validate_email(value: str) -> bool:
    return bool(value and "@" in value and "." in value)


def env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def default_alert_subject() -> str:
    return "Plithos alert"


def default_alert_message() -> str:
    return "Plithos detected an incident. A report is attached."


def report_recipient(user: sqlite3.Row) -> str:
    return str(user["report_email"] or user["email"] or "").strip().lower()


def mail_delivery_config(user: sqlite3.Row | None = None) -> dict[str, Any]:
    host = os.environ.get("PLITHOS_EMAIL_HOST", "").strip()
    port = os.environ.get("PLITHOS_EMAIL_PORT", "").strip() or "587"
    username = os.environ.get("PLITHOS_EMAIL_USERNAME", "").strip()
    password = os.environ.get("PLITHOS_EMAIL_PASSWORD", "")
    sender = os.environ.get("PLITHOS_EMAIL_SENDER", "").strip()
    if host and sender:
        return {
            "host": host,
            "port": int(port or 587),
            "username": username,
            "password": password,
            "sender": sender,
            "use_tls": env_flag("PLITHOS_EMAIL_USE_TLS", True),
        }
    if user and user["smtp_host"] and user["smtp_sender"]:
        return {
            "host": str(user["smtp_host"]).strip(),
            "port": int(user["smtp_port"] or 587),
            "username": str(user["smtp_username"] or "").strip(),
            "password": str(user["smtp_password"] or ""),
            "sender": str(user["smtp_sender"]).strip(),
            "use_tls": bool(user["smtp_use_tls"]),
        }
    return {"host": "", "port": 587, "username": "", "password": "", "sender": "", "use_tls": True}


def email_sender_ready(user: sqlite3.Row | None = None) -> bool:
    config = mail_delivery_config(user)
    return bool(config["host"] and config["sender"])


def send_email_message(
    *,
    to_address: str,
    subject: str,
    body: str,
    user: sqlite3.Row | None = None,
    attachment_name: str | None = None,
    attachment_bytes: bytes | None = None,
) -> tuple[bool, str]:
    config = mail_delivery_config(user)
    if not config["host"] or not config["sender"] or not to_address:
        return False, "Email delivery is not configured."

    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = config["sender"]
    message["To"] = to_address
    message.set_content(body)
    if attachment_name and attachment_bytes is not None:
        message.add_attachment(
            attachment_bytes,
            maintype="text",
            subtype="csv",
            filename=attachment_name,
        )

    try:
        with smtplib.SMTP(config["host"], int(config["port"] or 587), timeout=20) as smtp:
            if config["use_tls"]:
                smtp.starttls()
            if config["username"]:
                smtp.login(config["username"], config["password"] or "")
            smtp.send_message(message)
        return True, ""
    except Exception as exc:  # pragma: no cover - depends on local mail configuration
        return False, str(exc)


def create_user(full_name: str, email: str, password: str) -> int:
    return execute_db(
        """
        INSERT INTO users (full_name, email, password_hash, report_email, alert_subject, alert_message, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            full_name.strip(),
            email.strip().lower(),
            generate_password_hash(password),
            email.strip().lower(),
            default_alert_subject(),
            default_alert_message(),
            now_iso(),
        ),
    )


def update_user_settings(user_id: int, payload: dict[str, Any]) -> None:
    fields = ", ".join(f"{key} = ?" for key in payload.keys())
    execute_params = tuple(payload.values()) + (user_id,)
    with db_connect() as conn:
        conn.execute(f"UPDATE users SET {fields} WHERE id = ?", execute_params)
        conn.commit()


def default_model_settings() -> dict[str, bool]:
    return {
        "violence": True,
        "fire": True,
        "safety": True,
    }


def model_settings_from_record(record: Any | None) -> dict[str, bool]:
    settings = default_model_settings()
    if record is None:
        return settings
    keys = set(record.keys()) if hasattr(record, "keys") else set(record.keys()) if isinstance(record, dict) else set()
    for key in settings:
        field_name = f"{key}_enabled"
        if key in keys:
            settings[key] = bool(record[key])
        elif field_name in keys:
            settings[key] = bool(record[field_name])
    return settings


def model_settings_signature(settings: dict[str, bool]) -> tuple[tuple[str, bool], ...]:
    return tuple(sorted((key, bool(value)) for key, value in settings.items()))


def replace_cameras(user_id: int, cameras: list[dict[str, str]]) -> None:
    with db_connect() as conn:
        existing_rows = conn.execute(
            "SELECT id, source, enabled FROM cameras WHERE user_id = ? ORDER BY id",
            (user_id,),
        ).fetchall()
        existing_by_source = {str(row["source"]): row for row in existing_rows}
        used_ids: set[int] = set()
        created_at = now_iso()
        for index, camera in enumerate(cameras):
            source = camera["source"].strip()
            camera_name = camera["name"].strip()
            existing = existing_by_source.get(source)
            if existing and int(existing["id"]) not in used_ids:
                conn.execute(
                    """
                    UPDATE cameras
                    SET name = ?, source = ?, enabled = 1, sort_order = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (
                        camera_name,
                        source,
                        index,
                        created_at,
                        int(existing["id"]),
                    ),
                )
                used_ids.add(int(existing["id"]))
            else:
                conn.execute(
                    """
                    INSERT INTO cameras (user_id, name, source, enabled, sort_order, created_at, updated_at)
                    VALUES (?, ?, ?, 1, ?, ?, ?)
                    """,
                    (
                        user_id,
                        camera_name,
                        source,
                        index,
                        created_at,
                        created_at,
                    ),
                )
                used_ids.add(int(conn.execute("SELECT last_insert_rowid()").fetchone()[0]))
        for offset, row in enumerate(existing_rows, start=0):
            if int(row["id"]) in used_ids:
                continue
            conn.execute(
                "UPDATE cameras SET enabled = 0, sort_order = ?, updated_at = ? WHERE id = ?",
                (max(len(cameras), offset), created_at, int(row["id"])),
            )
        conn.commit()


def alert_row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    started_at = str(row["started_at"] or row["created_at"])
    ended_at = str(row["ended_at"] or "") or None
    last_seen_at = str(row["last_seen_at"] or ended_at or started_at)
    status = str(row["status"] or ("active" if not ended_at else "resolved"))
    confidence = row["max_confidence"] if row["max_confidence"] is not None else row["confidence"]
    duration_seconds = incident_duration_from_row(row)
    return {
        "id": row["id"],
        "camera_id": row["camera_id"],
        "camera_name": row["camera_name"],
        "type": row["event_type"],
        "severity": row["severity"],
        "detail": row["detail"],
        "confidence": confidence,
        "persons": row["persons"],
        "time": started_at,
        "started_at": started_at,
        "ended_at": ended_at,
        "last_seen_at": last_seen_at,
        "duration_seconds": duration_seconds,
        "duration_label": human_duration(duration_seconds),
        "status": status,
        "event_count": int(row["event_count"] or 1),
        "evidence_url": str(row["evidence_url"] or ""),
    }


def record_alert(
    *,
    user_id: int,
    camera_id: int,
    camera_name: str,
    event_type: str,
    severity: str,
    detail: str,
    confidence: float | None,
    persons: int | None,
) -> dict[str, Any]:
    created_at = now_iso()
    alert_id = execute_db(
        """
        INSERT INTO alerts (
            user_id, camera_id, camera_name, event_type, severity, detail,
            confidence, max_confidence, persons, status, started_at, last_seen_at,
            duration_seconds, event_count, evidence_url, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'active', ?, ?, 0, 1, '', ?)
        """,
        (
            user_id,
            camera_id,
            camera_name,
            event_type,
            severity,
            detail,
            confidence,
            confidence,
            persons,
            created_at,
            created_at,
            created_at,
        ),
    )
    alert = alert_row_to_dict(fetch_one("SELECT * FROM alerts WHERE id = ?", (alert_id,)))
    alert["user_id"] = user_id
    threading.Thread(target=send_alert_email_if_due, args=(alert,), daemon=True).start()
    return alert


def update_alert_incident(
    alert_id: int,
    *,
    severity: str,
    detail: str,
    confidence: float | None,
    persons: int | None,
    last_seen_at: str | None = None,
) -> dict[str, Any] | None:
    row = fetch_one("SELECT * FROM alerts WHERE id = ?", (alert_id,))
    if not row:
        return None
    seen_at = last_seen_at or now_iso()
    previous_confidence = row["max_confidence"] if row["max_confidence"] is not None else row["confidence"]
    max_confidence = previous_confidence
    if confidence is not None:
        max_confidence = confidence if max_confidence is None else max(float(max_confidence), float(confidence))
    max_persons = persons
    if row["persons"] is not None and persons is not None:
        max_persons = max(int(row["persons"]), int(persons))
    elif row["persons"] is not None:
        max_persons = int(row["persons"])
    event_count = int(row["event_count"] or 1) + 1
    with db_connect() as conn:
        conn.execute(
            """
            UPDATE alerts
            SET severity = ?,
                detail = ?,
                confidence = ?,
                max_confidence = ?,
                persons = ?,
                last_seen_at = ?,
                event_count = ?,
                status = 'active'
            WHERE id = ?
            """,
            (severity, detail, confidence, max_confidence, max_persons, seen_at, event_count, alert_id),
        )
        conn.commit()
    updated = fetch_one("SELECT * FROM alerts WHERE id = ?", (alert_id,))
    return alert_row_to_dict(updated) if updated else None


def resolve_alert_incident(alert_id: int, *, ended_at: str | None = None) -> dict[str, Any] | None:
    row = fetch_one("SELECT * FROM alerts WHERE id = ?", (alert_id,))
    if not row:
        return None
    started_at = str(row["started_at"] or row["created_at"] or now_iso())
    final_seen_at = str(row["last_seen_at"] or row["created_at"] or started_at)
    closed_at = ended_at or final_seen_at or now_iso()
    duration_seconds = incident_duration_between(started_at, closed_at)
    with db_connect() as conn:
        conn.execute(
            """
            UPDATE alerts
            SET status = 'resolved',
                ended_at = ?,
                last_seen_at = ?,
                duration_seconds = ?
            WHERE id = ?
            """,
            (closed_at, final_seen_at, duration_seconds, alert_id),
        )
        conn.commit()
    resolved = fetch_one("SELECT * FROM alerts WHERE id = ?", (alert_id,))
    return alert_row_to_dict(resolved) if resolved else None


def build_alert_csv(alert: dict[str, Any]) -> bytes:
    return build_alerts_csv([alert])


def build_alerts_csv(alerts: list[dict[str, Any]]) -> bytes:
    stream = io.StringIO()
    writer = csv.writer(stream)
    writer.writerow(
        [
            "Started",
            "Ended",
            "Duration",
            "Camera",
            "Type",
            "Persons",
            "Footage",
        ]
    )
    for alert in alerts:
        writer.writerow(
            [
                human_time(alert.get("started_at") or alert.get("time")),
                human_time(alert.get("ended_at")),
                alert.get("duration_label") or human_duration(int(alert.get("duration_seconds") or 0)),
                alert["camera_name"],
                alert["type"],
                alert.get("persons") or "",
                alert.get("evidence_url") or "",
            ]
        )
    return stream.getvalue().encode("utf-8")


def send_alert_email_if_due(alert: dict[str, Any]) -> None:
    user = get_user_by_id(alert["user_id"])
    if not user or not user["notifications_enabled"]:
        return

    last_sent = parse_iso(user["last_email_sent_at"])
    if last_sent and (now_local() - last_sent) < timedelta(seconds=EMAIL_RATE_LIMIT_SECONDS):
        return

    recipient = report_recipient(user)
    if not recipient or not validate_email(recipient):
        return

    subject_base = str(user["alert_subject"] or "").strip() or default_alert_subject()
    subject = f"{subject_base} | {alert['camera_name']} | {alert['type'].title()}"
    confidence = ""
    if alert.get("confidence") is not None:
        confidence = f"{float(alert['confidence']) * 100:.1f}%" if alert["type"] == "violence" else f"{float(alert['confidence']):.2f}"
    intro = str(user["alert_message"] or "").strip() or default_alert_message()
    body = "\n".join(
        [
            intro,
            "",
            f"Time: {human_time(alert['time'])}",
            f"Camera: {alert['camera_name']}",
            f"Type: {alert['type'].title()}",
            f"Severity: {alert['severity']}",
            f"Detail: {alert['detail']}",
            f"Confidence: {confidence or '--'}",
            f"Persons: {alert.get('persons') or '--'}",
            "",
            "A CSV report is attached to this email.",
        ]
    )
    ok, error = send_email_message(
        to_address=recipient,
        subject=subject,
        body=body,
        user=user,
        attachment_name=f"plithos-alert-{alert['camera_id']}-{alert['id']}.csv",
        attachment_bytes=build_alert_csv(alert),
    )
    if ok:
        update_user_settings(alert["user_id"], {"last_email_sent_at": now_iso()})
    elif error:
        print(f"[WARN] Email send failed: {error}")


class SharedModels:
    def __init__(self) -> None:
        self.ready = False
        self.loading = False
        self.error = ""
        self.lock = threading.Lock()
        self.inference_lock = threading.Lock()
        self.v_model = None
        self.f_model = None
        self.p_model = None
        self.ppe_model = None
        self.pose_model = None

    def ensure_loaded(self) -> None:
        with self.lock:
            if self.ready or self.loading:
                return
            self.loading = True
        threading.Thread(target=self._load, daemon=True).start()

    def _load(self) -> None:
        try:
            run_v = app_module.MODE in ("both", "violence")
            self.v_model = app_module.load_violence_model(app_module.VIOLENCE_H5_PATH, app_module.VIOLENCE_ONNX_PATH) if run_v else None
            try:
                self.f_model = app_module.load_fire_model(app_module.FIRE_MODEL_PATH) if app_module.ENABLE_FIRE else None
            except FileNotFoundError:
                print("[WARN] Fire model not found - fire detection disabled")
                self.f_model = None
            self.p_model = app_module.load_person_model()
            try:
                self.ppe_model = app_module.load_ppe_model(app_module.PPE_MODEL_PATH) if app_module.ENABLE_PPE else None
            except FileNotFoundError:
                print("[WARN] PPE model not found - safety detection disabled")
                self.ppe_model = None
            self.pose_model = app_module.load_pose_model(app_module.POSE_MODEL_PATH) if run_v else None
            app_module.warmup(self.v_model, self.f_model, self.p_model, self.ppe_model, self.pose_model)
            self.ready = True
            self.error = ""
        except Exception as exc:  # pragma: no cover - model env specific
            self.error = str(exc)
            print(f"[ERROR] Model loading failed: {exc}")
        finally:
            self.loading = False


shared_models = SharedModels()


def base_camera_state(camera_id: int, name: str, source: str, user_id: int) -> dict[str, Any]:
    return {
        "camera_id": camera_id,
        "user_id": user_id,
        "name": name,
        "source": source,
        "camera_online": False,
        "models_ready": shared_models.ready,
        "frame_skipped": False,
        "is_violent": False,
        "v_conf": 0.0,
        "is_fire": False,
        "is_smoke": False,
        "fire_dets": [],
        "is_safety_missing": False,
        "ppe_dets": [],
        "ppe_people": [],
        "safety_missing_items": [],
        "safety_summary": "No missing safety equipment",
        "person_detections": [],
        "person_count": 0,
        "fps": 0.0,
        "frame": 0,
        "video_timestamp": "",
        "updated_at": "",
        "uptime_sec": 0,
        "active_alert": False,
        "alert_type": "",
        "last_alert": None,
        "last_alert_id": None,
        "last_detail": "",
    }


def open_capture(source: str):
    src: Any = int(source) if str(source).isdigit() else source
    if isinstance(src, int):
        cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(src)
    else:
        cap = cv2.VideoCapture(src)
    return cap


def configure_capture(cap) -> None:
    if cap is None:
        return
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass
    try:
        cap.set(cv2.CAP_PROP_FPS, TARGET_CAMERA_FPS)
    except Exception:
        pass
    try:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_FRAME_HEIGHT)
    except Exception:
        pass
    try:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    except Exception:
        pass


def placeholder_frame(name: str, message: str) -> bytes:
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(img, (26, 26), (614, 454), (44, 78, 112), 2)
    cv2.putText(img, name[:28], (52, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (236, 242, 247), 2, cv2.LINE_AA)
    cv2.putText(img, message, (52, 246), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (162, 182, 201), 2, cv2.LINE_AA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    return buf.tobytes() if ok else b""


def encode_jpeg(frame: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    return buf.tobytes() if ok else b""


def discover_local_camera_sources(force: bool = False) -> list[dict[str, str]]:
    now_tick = time.monotonic()
    with camera_scan_lock:
        cached_items = camera_scan_cache.get("items", [])
        checked_at = float(camera_scan_cache.get("checked_at", 0.0))
        if not force and cached_items and (now_tick - checked_at) < CAMERA_SCAN_CACHE_SECONDS:
            return [dict(item) for item in cached_items]

    discovered: list[dict[str, str]] = []
    for index in range(max(CAMERA_SCAN_MAX, 1)):
        cap = open_capture(str(index))
        try:
            if not cap.isOpened():
                continue
            configure_capture(cap)
            ok, _ = cap.read()
            if ok:
                discovered.append({"source": str(index), "label": f"Camera {len(discovered) + 1}"})
        finally:
            cap.release()

    with camera_scan_lock:
        camera_scan_cache["checked_at"] = now_tick
        camera_scan_cache["items"] = [dict(item) for item in discovered]
    return discovered


def preview_frame_for_source(source: str, title: str) -> bytes:
    cache_key = str(source)
    now_tick = time.monotonic()
    with preview_cache_lock:
        cached = preview_frame_cache.get(cache_key)
        if cached and (now_tick - float(cached.get("created_at", 0.0))) < PREVIEW_CACHE_SECONDS:
            return cached["payload"]

    cap = open_capture(source)
    payload = placeholder_frame(title, "Camera preview unavailable")
    try:
        if cap.isOpened():
            configure_capture(cap)
            ok, frame = cap.read()
            if ok and frame is not None:
                payload = encode_jpeg(frame)
    finally:
        cap.release()

    with preview_cache_lock:
        preview_frame_cache[cache_key] = {"created_at": now_tick, "payload": payload}
    return payload


def camera_setup_slots(user_id: int, *, include_detected: bool = True, force_scan: bool = False) -> list[dict[str, Any]]:
    saved_rows = cameras_for_user(user_id)
    saved_by_source = {str(row["source"]): row for row in saved_rows}
    slots: list[dict[str, Any]] = []

    if include_detected:
        for detected in discover_local_camera_sources(force=force_scan):
            source = str(detected["source"])
            saved = saved_by_source.pop(source, None)
            slots.append(
                {
                    "source": source,
                    "title": detected["label"],
                    "name": str(saved["name"]) if saved else "",
                    "camera_id": int(saved["id"]) if saved else None,
                    "connected": True,
                    "is_local": True,
                }
            )
    else:
        for offset, row in enumerate(saved_rows, start=1):
            slots.append(
                {
                    "source": str(row["source"]),
                    "title": f"Camera {offset}",
                    "name": str(row["name"]),
                    "camera_id": int(row["id"]),
                    "connected": True,
                    "is_local": str(row["source"]).isdigit(),
                }
            )
        return slots

    for offset, row in enumerate(saved_by_source.values(), start=len(slots) + 1):
        slots.append(
            {
                "source": str(row["source"]),
                "title": f"Saved camera {offset}",
                "name": str(row["name"]),
                "camera_id": int(row["id"]),
                "connected": False,
                "is_local": str(row["source"]).isdigit(),
            }
        )

    return slots


@dataclass
class CameraRuntime:
    camera_id: int
    user_id: int
    name: str
    source: str
    enabled_models: dict[str, bool] | None = None

    def __post_init__(self) -> None:
        self.enabled_models = model_settings_from_record(self.enabled_models)
        self.state_lock = threading.Lock()
        self.frame_lock = threading.Lock()
        self.result_lock = threading.Lock()
        self.pending_frame_lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.inference_thread: threading.Thread | None = None
        self.session = None
        self.latest_jpeg: bytes | None = None
        self.latest_jpeg_seq = 0
        self.pending_inference_frame: np.ndarray | None = None
        self.placeholder = placeholder_frame(self.name, "Waiting for the live feed")
        self.incident_gate = {
            "violence": {"incident_id": None, "last_seen_tick": 0.0, "last_persist_tick": 0.0, "last_detail": ""},
            "fire": {"incident_id": None, "last_seen_tick": 0.0, "last_persist_tick": 0.0, "last_detail": ""},
            "safety": {"incident_id": None, "last_seen_tick": 0.0, "last_persist_tick": 0.0, "last_detail": ""},
        }
        self.trace_history: dict[str, deque[tuple[int, int]]] = {}
        self.trace_sequence = {"person": 0, "fire": 0, "safety": 0}
        self.state = base_camera_state(self.camera_id, self.name, self.source, self.user_id)
        self._violence_sustain_gate = app_module.ViolenceSustainGate(app_module.VIOLENCE_SUSTAIN_SEC)
        self._prev_gray_motion: "np.ndarray | None" = None
        self.last_detection = {
            "is_violent": False,
            "v_conf": 0.0,
            "is_fire": False,
            "is_smoke": False,
            "fire_dets": [],
            "is_safety_missing": False,
            "ppe_dets": [],
            "ppe_people": [],
            "safety_missing_items": [],
            "safety_summary": "No missing safety equipment",
            "person_detections": [],
            "person_count": 0,
            "frame_skipped": False,
        }

    def start(self) -> None:
        if self.thread and self.thread.is_alive():
            return
        self.stop_event.clear()
        if not self.inference_thread or not self.inference_thread.is_alive():
            self.inference_thread = threading.Thread(target=self.run_inference_loop, daemon=True)
            self.inference_thread.start()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()

    def snapshot(self) -> dict[str, Any]:
        with self.state_lock:
            return {
                **self.state,
                "enabled_models": dict(self.enabled_models or default_model_settings()),
                "fire_dets": [dict(item) for item in self.state["fire_dets"]],
                "ppe_dets": [dict(item) for item in self.state["ppe_dets"]],
                "ppe_people": [dict(item) for item in self.state["ppe_people"]],
                "person_detections": [dict(item) for item in self.state["person_detections"]],
                "safety_missing_items": list(self.state["safety_missing_items"]),
                "last_alert": dict(self.state["last_alert"]) if self.state["last_alert"] else None,
            }

    def current_jpeg(self) -> bytes:
        with self.frame_lock:
            return self.latest_jpeg or self.placeholder

    def current_frame_packet(self) -> tuple[bytes, int]:
        with self.frame_lock:
            return self.latest_jpeg or self.placeholder, self.latest_jpeg_seq

    def update_trace_history(self, last: dict[str, Any]) -> dict[str, list[tuple[int, int]]]:
        def assign_points(prefix: str, items: list[dict[str, Any]], *, distance_limit: float = 140.0) -> dict[str, tuple[int, int]]:
            assigned: dict[str, tuple[int, int]] = {}
            existing_keys = [key for key in self.trace_history.keys() if key.startswith(f"{prefix}:")]
            used_keys: set[str] = set()
            for item in items:
                x1, y1, x2, y2 = item["bbox"]
                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                best_key = None
                best_distance = None
                for key in existing_keys:
                    if key in used_keys:
                        continue
                    trail = self.trace_history.get(key)
                    if not trail:
                        continue
                    px, py = trail[-1]
                    distance = ((center[0] - px) ** 2 + (center[1] - py) ** 2) ** 0.5
                    if best_distance is None or distance < best_distance:
                        best_distance = distance
                        best_key = key
                if best_key is not None and best_distance is not None and best_distance <= distance_limit:
                    assigned[best_key] = center
                    used_keys.add(best_key)
                    continue
                self.trace_sequence[prefix] += 1
                new_key = f"{prefix}:{self.trace_sequence[prefix]}"
                assigned[new_key] = center
                used_keys.add(new_key)
            return assigned

        active_points: dict[str, tuple[int, int]] = {}
        active_points.update(assign_points("person", list(last.get("person_detections", []))))
        active_points.update(assign_points("fire", list(last.get("fire_dets", []))))
        active_points.update(assign_points("safety", list(last.get("ppe_people", []))))

        stale_keys = [key for key in self.trace_history.keys() if key not in active_points]
        for key in stale_keys:
            self.trace_history.pop(key, None)

        for key, point in active_points.items():
            trail = self.trace_history.get(key)
            if trail is None:
                trail = deque(maxlen=app_module.PPE_TRACE_LENGTH)
                self.trace_history[key] = trail
            trail.append(point)

        return {key: list(points) for key, points in self.trace_history.items() if points}

    def update_config(self, *, user_id: int, name: str, source: str, enabled_models: dict[str, bool] | None = None) -> None:
        self.user_id = user_id
        self.name = name
        self.source = source
        if enabled_models is not None:
            self.enabled_models = model_settings_from_record(enabled_models)
        self.placeholder = placeholder_frame(self.name, "Waiting for the live feed")
        with self.state_lock:
            self.state["user_id"] = user_id
            self.state["name"] = name
            self.state["source"] = source

    def mark_models_state(self) -> None:
        with self.state_lock:
            self.state["models_ready"] = shared_models.ready
            if shared_models.error:
                self.state["last_detail"] = shared_models.error

    def queue_inference(self, frame: np.ndarray) -> None:
        with self.pending_frame_lock:
            self.pending_inference_frame = frame

    def pop_pending_inference(self) -> np.ndarray | None:
        with self.pending_frame_lock:
            frame = self.pending_inference_frame
            self.pending_inference_frame = None
            return frame

    def reset_incidents(self) -> None:
        for tracker in self.incident_gate.values():
            tracker["incident_id"] = None
            tracker["last_seen_tick"] = 0.0
            tracker["last_persist_tick"] = 0.0
            tracker["last_detail"] = ""
        self.trace_history.clear()
        self.trace_sequence = {"person": 0, "fire": 0, "safety": 0}
        with self.state_lock:
            self.state["last_alert"] = None
            self.state["last_alert_id"] = None

    def emit_alert(self, event_type: str, active: bool, detail: str, severity: str, confidence: float | None, persons: int) -> None:
        now_tick = time.monotonic()
        gate = self.incident_gate[event_type]
        if active:
            gate["last_seen_tick"] = now_tick
            should_persist = (
                gate["incident_id"] is None
                or detail != gate["last_detail"]
                or (now_tick - gate["last_persist_tick"]) >= INCIDENT_UPDATE_INTERVAL_SECONDS
            )
            gate["last_detail"] = detail
            if gate["incident_id"] is None:
                alert = record_alert(
                    user_id=self.user_id,
                    camera_id=self.camera_id,
                    camera_name=self.name,
                    event_type=event_type,
                    severity=severity,
                    detail=detail,
                    confidence=confidence,
                    persons=persons,
                )
                gate["incident_id"] = int(alert["id"])
                gate["last_persist_tick"] = now_tick
                with self.state_lock:
                    self.state["last_alert"] = alert
                    self.state["last_alert_id"] = alert["id"]
                return
            if not should_persist:
                return
            alert = update_alert_incident(
                int(gate["incident_id"]),
                severity=severity,
                detail=detail,
                confidence=confidence,
                persons=persons,
                last_seen_at=now_iso(),
            )
            gate["last_persist_tick"] = now_tick
            if alert:
                with self.state_lock:
                    self.state["last_alert"] = alert
                    self.state["last_alert_id"] = alert["id"]
            return
        incident_id = gate["incident_id"]
        if not incident_id:
            return
        if now_tick - gate["last_seen_tick"] < INCIDENT_END_GRACE_SECONDS[event_type]:
            return
        alert = resolve_alert_incident(int(incident_id))
        gate["incident_id"] = None
        gate["last_seen_tick"] = 0.0
        gate["last_persist_tick"] = 0.0
        gate["last_detail"] = ""
        if alert:
            with self.state_lock:
                self.state["last_alert"] = alert
                self.state["last_alert_id"] = alert["id"]

    def apply_inference_result(self, result: dict[str, Any]) -> None:
        with self.result_lock:
            last = dict(self.last_detection)
            last["frame_skipped"] = result.get("skipped", False)
            if not last["frame_skipped"]:
                persons = int(result.get("person_count", 0))
                violent = bool(result.get("is_violent", False))

                # ── Tier 2: Sustained-alert timer (server path) ───────────────
                sustained = self._violence_sustain_gate.update(violent)
                if not sustained:
                    violent = False

                last["is_violent"] = violent
                last["v_conf"] = float(result.get("violence_confidence", 0.0)) if violent else 0.0
                last["is_fire"] = bool(result.get("is_fire", False))
                last["is_smoke"] = bool(result.get("is_smoke", False))
                last["fire_dets"] = [
                    {"bbox": list(item["bbox"]), "conf": float(item["confidence"]), "label": str(item["label"])}
                    for item in result.get("fire_detections", [])
                ]
                last["is_safety_missing"] = bool(result.get("is_safety_missing", False))
                last["ppe_dets"] = [
                    {
                        "bbox": list(item["bbox"]),
                        "conf": float(item["confidence"]),
                        "label": str(item["label"]),
                        "missing_item": str(item.get("missing_item") or item["label"]),
                        "person_index": item.get("person_index"),
                    }
                    for item in result.get("ppe_detections", [])
                ]
                last["ppe_people"] = [
                    {
                        "bbox": list(item["bbox"]),
                        "conf": float(item.get("confidence", 0.0)),
                        "label": str(item["label"]),
                        "missing_items": list(item.get("missing_items", [])),
                        "person_index": item.get("person_index"),
                    }
                    for item in result.get("ppe_people", [])
                ]
                last["safety_missing_items"] = list(result.get("safety_missing_items", []))
                last["safety_summary"] = str(result.get("safety_summary") or "No missing safety equipment")
                last["person_detections"] = [
                    {"bbox": list(item["bbox"]), "conf": float(item["confidence"])}
                    for item in result.get("person_detections", [])
                ]
                last["person_count"] = persons
            self.last_detection = last

        fire_active = bool(last["is_fire"] or last["is_smoke"])
        safety_active = bool(last["is_safety_missing"] and last["ppe_people"])
        fire_confidence = max((float(item.get("conf", 0.0)) for item in last["fire_dets"]), default=0.0)
        safety_confidence = max((float(item.get("conf", 0.0)) for item in last["ppe_dets"]), default=0.0)
        v_pct = int(round(last["v_conf"] * 100))
        fire_label = (
            "Fire and smoke"
            if last["is_fire"] and last["is_smoke"]
            else "Fire"
            if last["is_fire"]
            else "Smoke"
            if last["is_smoke"]
            else "Clear"
        )
        self.emit_alert("violence", last["is_violent"], f"Confidence {v_pct}%", "HIGH", last["v_conf"], int(last["person_count"]))
        self.emit_alert("fire", fire_active, fire_label, "HIGH", fire_confidence, int(last["person_count"]))
        self.emit_alert(
            "safety",
            safety_active,
            str(last.get("safety_summary") or "Missing safety equipment"),
            "HIGH",
            safety_confidence,
            int(last["person_count"]),
        )

    def run_inference_loop(self) -> None:
        inference_idle_sleep = min(ACTIVE_SLEEP, 0.01)
        while not self.stop_event.is_set():
            if not shared_models.ready:
                time.sleep(inference_idle_sleep)
                continue

            if self.session is None and HAS_API:
                self.session = InferenceSession(
                    shared_models.v_model,
                    None,
                    shared_models.f_model,
                    shared_models.p_model,
                    shared_models.ppe_model,
                    enabled_models=self.enabled_models,
                    pose_model=shared_models.pose_model,
                )
            if self.session is None:
                time.sleep(inference_idle_sleep)
                continue

            frame = self.pop_pending_inference()
            if frame is None:
                time.sleep(inference_idle_sleep)
                continue

            with shared_models.inference_lock:
                result = self.session.infer(frame)
            self.apply_inference_result(result)

    def run(self) -> None:
        shared_models.ensure_loaded()
        fps_cap = max(int(app_module.FPS_CAP or 0), TARGET_CAMERA_FPS)
        frame_interval = (1.0 / fps_cap) if fps_cap > 0 else 0.0
        inference_interval = (1.0 / MAX_INFERENCE_FPS) if MAX_INFERENCE_FPS > 0 else 0.0
        prev_gray = None
        frame_count = 0
        fps_counter = 0
        fps_display = 0.0
        t_fps = time.time()
        t_last = 0.0
        last_inference_request_at = 0.0
        started_at = time.monotonic()

        while not self.stop_event.is_set():
            self.mark_models_state()
            if not shared_models.ready:
                with self.state_lock:
                    self.state["camera_online"] = False
                    self.state["updated_at"] = now_iso()
                    self.state["uptime_sec"] = int(time.monotonic() - started_at)
                time.sleep(0.5)
                continue

            cap = open_capture(self.source)
            if not cap.isOpened():
                with self.state_lock:
                    self.state["camera_online"] = False
                    self.state["updated_at"] = now_iso()
                    self.state["last_detail"] = "Camera not available"
                time.sleep(2.0)
                continue

            configure_capture(cap)

            while not self.stop_event.is_set():
                ok, frame = cap.read()
                if not ok:
                    with self.state_lock:
                        self.state["camera_online"] = False
                        self.state["updated_at"] = now_iso()
                        self.state["last_detail"] = "Connection lost"
                    break

                if frame_interval > 0:
                    now_tick = time.time()
                    wait = frame_interval - (now_tick - t_last)
                    if wait > 0:
                        time.sleep(wait)
                t_last = time.time()
                stamp = now_local()

                frame_count += 1
                fps_counter += 1
                if fps_counter >= 15:
                    elapsed = max(time.time() - t_fps, 1e-6)
                    fps_display = fps_counter / elapsed
                    fps_counter = 0
                    t_fps = time.time()

                curr_gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (160, 120))
                motion = prev_gray is None or app_module.has_motion(prev_gray, curr_gray)
                prev_gray = curr_gray

                if self.session is not None:
                    now_monotonic = time.monotonic()
                    min_interval = inference_interval if motion else max(inference_interval, IDLE_INFERENCE_INTERVAL_SECONDS)
                    infer_due = min_interval <= 0 or (now_monotonic - last_inference_request_at) >= min_interval
                    if infer_due:
                        self.queue_inference(frame.copy())
                        last_inference_request_at = now_monotonic

                with self.result_lock:
                    last = dict(self.last_detection)
                trace_points = self.update_trace_history(last)
                annotated = app_module.draw_live_annotations(
                    frame.copy(),
                    is_violent=bool(last.get("is_violent")),
                    v_conf=float(last.get("v_conf", 0.0)),
                    fire_detections=last.get("fire_dets", []),
                    person_detections=last.get("person_detections", []),
                    ppe_people=last.get("ppe_people", []),
                    ppe_detections=last.get("ppe_dets", []),
                    video_timestamp=stamp.strftime("%Y-%m-%d %H:%M:%S"),
                    trace_points=trace_points,
                    enabled_models=self.enabled_models,
                )
                payload = encode_jpeg(annotated)
                if payload:
                    with self.frame_lock:
                        self.latest_jpeg = payload
                        self.latest_jpeg_seq += 1

                fire_active = bool(last["is_fire"] or last["is_smoke"])
                safety_active = bool(last["is_safety_missing"] and last["ppe_people"])
                alert_type = ""
                if last["is_violent"]:
                    alert_type = "violence"
                elif fire_active:
                    alert_type = "fire"
                elif safety_active:
                    alert_type = "safety"

                v_pct = int(round(last["v_conf"] * 100))
                fire_label = "Fire and smoke" if last["is_fire"] and last["is_smoke"] else "Fire" if last["is_fire"] else "Smoke" if last["is_smoke"] else "Clear"
                safety_label = str(last.get("safety_summary") or "No missing safety equipment")

                with self.state_lock:
                    self.state.update(last)
                    self.state["camera_online"] = True
                    self.state["models_ready"] = shared_models.ready
                    self.state["fps"] = round(fps_display, 1)
                    self.state["frame"] = frame_count
                    self.state["video_timestamp"] = stamp.strftime("%Y-%m-%d %H:%M:%S")
                    self.state["updated_at"] = stamp.isoformat(timespec="seconds")
                    self.state["uptime_sec"] = int(time.monotonic() - started_at)
                    self.state["active_alert"] = bool(alert_type)
                    self.state["alert_type"] = alert_type
                    self.state["last_detail"] = (
                        fire_label
                        if fire_active
                        else safety_label
                        if safety_active
                        else f"{v_pct}% confidence"
                        if last["is_violent"]
                        else "Monitoring"
                    )
            cap.release()


def refresh_camera_runtimes() -> None:
    desired_rows = fetch_all(
        """
        SELECT
            cameras.*,
            users.violence_enabled,
            users.fire_enabled,
            users.safety_enabled
        FROM cameras
        JOIN users ON users.id = cameras.user_id
        WHERE cameras.enabled = 1
        ORDER BY cameras.sort_order, cameras.id
        """
    )
    desired = {int(row["id"]): row for row in desired_rows}
    with registry_lock:
        for camera_id, runtime in list(camera_runtimes.items()):
            row = desired.get(camera_id)
            if row is None:
                runtime.stop()
                del camera_runtimes[camera_id]
                continue
            desired_model_settings = model_settings_from_record(row)
            if (
                runtime.name != row["name"]
                or runtime.source != row["source"]
                or runtime.user_id != row["user_id"]
                or model_settings_signature(runtime.enabled_models or default_model_settings()) != model_settings_signature(desired_model_settings)
            ):
                runtime.stop()
                del camera_runtimes[camera_id]
        for camera_id, row in desired.items():
            if camera_id in camera_runtimes:
                continue
            runtime = CameraRuntime(
                camera_id=camera_id,
                user_id=int(row["user_id"]),
                name=str(row["name"]),
                source=str(row["source"]),
                enabled_models=model_settings_from_record(row),
            )
            camera_runtimes[camera_id] = runtime
            runtime.start()


def pick_focus_camera(states: list[dict[str, Any]]) -> int | None:
    if not states:
        return None
    priorities = {"violence": 3, "fire": 2, "safety": 1, "": 0}
    best = max(states, key=lambda item: (priorities.get(item.get("alert_type", ""), 0), item.get("updated_at", "")))
    return int(best["camera_id"]) if best.get("active_alert") else int(states[0]["camera_id"])


@flask_app.before_request
def load_user() -> None:
    g.user = get_user_by_id(session.get("user_id"))


@flask_app.context_processor
def shared_context() -> dict[str, Any]:
    return {
        "current_user": g.get("user"),
        "current_year": now_local().year,
    }


def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if not g.user:
            next_url = request.full_path[:-1] if request.full_path.endswith("?") else request.full_path
            return redirect(url_for("login", next=next_url))
        return view(*args, **kwargs)

    return wrapped


def setup_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if g.user and not user_has_cameras(int(g.user["id"])) and request.endpoint not in {"setup", "logout"}:
            return redirect(url_for("setup"))
        return view(*args, **kwargs)

    return wrapped


def current_camera_states(user_id: int) -> list[dict[str, Any]]:
    rows = cameras_for_user(user_id)
    states: list[dict[str, Any]] = []
    with registry_lock:
        for row in rows:
            runtime = camera_runtimes.get(int(row["id"]))
            if runtime:
                states.append(runtime.snapshot())
            else:
                states.append(base_camera_state(int(row["id"]), str(row["name"]), str(row["source"]), int(row["user_id"])))
    return states


def reset_incident_trackers_for_user(user_id: int) -> None:
    with registry_lock:
        for runtime in camera_runtimes.values():
            if int(runtime.user_id) == int(user_id):
                runtime.reset_incidents()


def selected_camera_id_for_request(user_id: int) -> int | None:
    requested = request.args.get("camera", type=int)
    states = current_camera_states(user_id)
    if requested and any(state["camera_id"] == requested for state in states):
        return requested
    return pick_focus_camera(states)


def read_camera_form(
    prefix_name: str = "camera_name",
    prefix_source: str = "camera_source",
    remove_name: str = "camera_remove",
) -> list[dict[str, str]]:
    names = request.form.getlist(prefix_name)
    sources = request.form.getlist(prefix_source)
    removed_sources = {value.strip() for value in request.form.getlist(remove_name) if value.strip()}
    cameras: list[dict[str, str]] = []
    seen_sources: set[str] = set()
    for index, name in enumerate(names):
        source = (sources[index] if index < len(sources) else str(index)).strip() or str(index)
        camera_name = name.strip()
        if source in removed_sources or not camera_name or source in seen_sources:
            continue
        cameras.append({"name": camera_name, "source": source})
        seen_sources.add(source)
    return cameras


@flask_app.after_request
def no_store(response):
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    return response


@flask_app.route("/")
def landing():
    if g.user:
        return redirect(url_for("dashboard" if user_has_cameras(int(g.user["id"])) else "setup"))
    return render_template("landing.html", page_title="Plithos", active_page="landing")


@flask_app.route("/login", methods=["GET", "POST"])
def login():
    if g.user:
        return redirect(url_for("dashboard" if user_has_cameras(int(g.user["id"])) else "setup"))
    first_user = count_users() == 0
    next_url = request.values.get("next", "").strip() or request.args.get("next", "").strip()
    if request.method == "POST":
        if first_user:
            full_name = request.form.get("full_name", "").strip()
            email = request.form.get("email", "").strip().lower()
            password = request.form.get("password", "")
            confirm = request.form.get("confirm_password", "")
            if not full_name:
                flash("Enter your name to create the first account.", "error")
            elif not validate_email(email):
                flash("Enter a valid email address.", "error")
            elif len(password) < 8:
                flash("Use at least 8 characters for the password.", "error")
            elif password != confirm:
                flash("Passwords do not match.", "error")
            else:
                user_id = create_user(full_name, email, password)
                session["user_id"] = user_id
                refresh_camera_runtimes()
                return redirect(url_for("setup"))
        else:
            email = request.form.get("email", "").strip().lower()
            password = request.form.get("password", "")
            user = get_user_by_email(email)
            if not user or not check_password_hash(user["password_hash"], password):
                flash("Incorrect email or password.", "error")
            else:
                session["user_id"] = int(user["id"])
                return redirect(next_url or url_for("dashboard" if user_has_cameras(int(user["id"])) else "setup"))
    return render_template("login.html", page_title="Login", active_page="login", first_user=first_user, next_url=next_url)


@flask_app.route("/logout", methods=["POST"])
@login_required
def logout():
    session.clear()
    return redirect(url_for("landing"))


@flask_app.route("/setup", methods=["GET", "POST"])
@login_required
def setup():
    user_id = int(g.user["id"])
    if request.method == "POST":
        cameras = read_camera_form()
        if not cameras:
            flash("Name at least one camera to continue.", "error")
        else:
            replace_cameras(user_id, cameras)
            refresh_camera_runtimes()
            flash("Cameras saved.", "success")
            return redirect(url_for("dashboard"))
    slots = camera_setup_slots(user_id, force_scan=request.args.get("refresh") == "1")
    return render_template(
        "setup.html",
        page_title="Setup",
        active_page="setup",
        slots=slots,
    )


@flask_app.route("/dashboard")
@login_required
@setup_required
def dashboard():
    cameras = cameras_for_user(int(g.user["id"]))
    selected_camera_id = selected_camera_id_for_request(int(g.user["id"]))
    return render_template(
        "dashboard.html",
        page_title="Dashboard",
        active_page="dashboard",
        cameras=cameras,
        selected_camera_id=selected_camera_id,
        preferences={
            "theme": g.user["theme"],
            "sound_enabled": bool(g.user["sound_enabled"]),
            "auto_switch_alerts": bool(g.user["auto_switch_alerts"]),
        },
    )


@flask_app.route("/cameras")
@login_required
@setup_required
def cameras_page():
    cameras = cameras_for_user(int(g.user["id"]))
    return render_template(
        "cameras.html",
        page_title="All Cameras",
        active_page="cameras",
        cameras=cameras,
        preferences={
            "theme": g.user["theme"],
            "sound_enabled": bool(g.user["sound_enabled"]),
        },
    )


@flask_app.route("/logs")
@login_required
@setup_required
def logs():
    user_id = int(g.user["id"])
    camera_id = request.args.get("camera", type=int)
    event_type = request.args.get("type", "all")
    search = request.args.get("search", "").strip()
    date_from = request.args.get("date_from", "")
    date_to = request.args.get("date_to", "")
    raw_rows = alerts_for_user(
        user_id,
        limit=MAX_LOG_ROWS,
        camera_id=camera_id,
        event_type=event_type,
        search=search,
        date_from=date_from,
        date_to=date_to,
    )
    rows = [alert_row_to_dict(row) for row in raw_rows]
    by_type = {"violence": 0, "fire": 0, "safety": 0}
    by_day: dict[str, int] = {}
    for row in rows:
        by_type[row["type"]] = by_type.get(row["type"], 0) + 1
        day_key = str(row["started_at"] or row["time"])[:10]
        by_day[day_key] = by_day.get(day_key, 0) + 1
    trend = [{"label": key, "count": by_day[key]} for key in sorted(by_day.keys())[-7:]]
    type_max = max(by_type.values()) if by_type else 0
    trend_max = max((item["count"] for item in trend), default=0)
    return render_template(
        "logs.html",
        page_title="Logs",
        active_page="logs",
        cameras=cameras_for_user(user_id),
        rows=rows,
        filters={
            "camera": camera_id,
            "type": event_type,
            "search": search,
            "date_from": date_from,
            "date_to": date_to,
        },
        type_summary=by_type,
        trend=trend,
        type_max=type_max,
        trend_max=trend_max,
    )


@flask_app.route("/logs/export")
@login_required
@setup_required
def logs_export():
    user_id = int(g.user["id"])
    camera_id = request.args.get("camera", type=int)
    event_type = request.args.get("type", "all")
    search = request.args.get("search", "").strip()
    date_from = request.args.get("date_from", "")
    date_to = request.args.get("date_to", "")
    rows = alerts_for_user(
        user_id,
        limit=MAX_LOG_EXPORT_ROWS,
        camera_id=camera_id,
        event_type=event_type,
        search=search,
        date_from=date_from,
        date_to=date_to,
    )
    payload = build_alerts_csv([alert_row_to_dict(row) for row in rows])
    filename = f"plithos-logs-{datetime.now().strftime('%Y%m%d-%H%M%S')}.csv"
    return Response(
        payload,
        mimetype="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@flask_app.route("/logs/clear", methods=["POST"])
@login_required
@setup_required
def logs_clear():
    user_id = int(g.user["id"])
    clear_alerts_for_user(user_id)
    reset_incident_trackers_for_user(user_id)
    flash("Logs cleared.", "success")
    return redirect(url_for("logs"))


@flask_app.route("/settings", methods=["GET", "POST"])
@login_required
@setup_required
def settings():
    user_id = int(g.user["id"])
    if request.method == "POST":
        action = request.form.get("action")
        if action == "profile":
            full_name = request.form.get("full_name", "").strip()
            email = request.form.get("email", "").strip().lower()
            existing = get_user_by_email(email) if email else None
            if not full_name:
                flash("Enter a name for the account.", "error")
            elif not validate_email(email):
                flash("Enter a valid email address.", "error")
            elif existing and int(existing["id"]) != user_id:
                flash("That email is already in use.", "error")
            else:
                payload: dict[str, Any] = {"full_name": full_name, "email": email}
                if not g.user["report_email"] or str(g.user["report_email"]).strip().lower() == str(g.user["email"]).strip().lower():
                    payload["report_email"] = email
                update_user_settings(user_id, payload)
                flash("Account details updated.", "success")
        elif action == "preferences":
            update_user_settings(
                user_id,
                {
                    "theme": request.form.get("theme", "dark"),
                    "sound_enabled": 1 if request.form.get("sound_enabled") else 0,
                    "auto_switch_alerts": 1 if request.form.get("auto_switch_alerts") else 0,
                    "notifications_enabled": 1 if request.form.get("notifications_enabled") else 0,
                    "violence_enabled": 1 if request.form.get("violence_enabled") else 0,
                    "fire_enabled": 1 if request.form.get("fire_enabled") else 0,
                    "safety_enabled": 1 if request.form.get("safety_enabled") else 0,
                },
            )
            refresh_camera_runtimes()
            flash("Preferences saved.", "success")
        elif action == "alerts":
            report_email = request.form.get("report_email", "").strip().lower()
            subject = request.form.get("alert_subject", "").strip()
            message = request.form.get("alert_message", "").strip()
            if not validate_email(report_email):
                flash("Enter a valid email address for reports.", "error")
            else:
                update_user_settings(
                    user_id,
                    {
                        "report_email": report_email,
                        "alert_subject": subject or default_alert_subject(),
                        "alert_message": message or default_alert_message(),
                    },
                )
                flash("Alert email settings updated.", "success")
        elif action == "password":
            current_password = request.form.get("current_password", "")
            new_password = request.form.get("new_password", "")
            confirm_password = request.form.get("confirm_password", "")
            if not check_password_hash(str(g.user["password_hash"]), current_password):
                flash("Enter your current password to make a change.", "error")
            elif len(new_password) < 8:
                flash("Use at least 8 characters for the new password.", "error")
            elif new_password != confirm_password:
                flash("The new passwords do not match.", "error")
            else:
                update_user_settings(user_id, {"password_hash": generate_password_hash(new_password)})
                flash("Password updated.", "success")
        elif action == "cameras":
            cameras = read_camera_form()
            replace_cameras(user_id, cameras)
            refresh_camera_runtimes()
            if cameras:
                flash("Camera settings updated.", "success")
            else:
                flash("All cameras were removed. Add a camera to continue monitoring.", "success")
        elif action == "send_test_email":
            subject = str(g.user["alert_subject"] or default_alert_subject()).strip()
            message = "\n".join(
                [
                    str(g.user["alert_message"] or default_alert_message()).strip(),
                    "",
                    "This is a test email from Plithos.",
                    f"Time: {human_time(now_iso())}",
                ]
            )
            ok, error = send_email_message(
                to_address=report_recipient(g.user),
                subject=subject,
                body=message,
                user=g.user,
            )
            if ok:
                flash("Test email sent.", "success")
            else:
                flash(error or "Email delivery is not ready yet.", "error")
        return redirect(url_for("settings"))

    return render_template(
        "settings.html",
        page_title="Settings",
        active_page="settings",
        cameras=cameras_for_user(user_id),
        camera_slots=camera_setup_slots(
            user_id,
            include_detected=request.args.get("refresh") == "1",
            force_scan=request.args.get("refresh") == "1",
        ),
    )


@flask_app.route("/api/state")
@login_required
@setup_required
def api_state():
    user_id = int(g.user["id"])
    states = current_camera_states(user_id)
    recent_alerts = [alert_row_to_dict(row) for row in alerts_for_user(user_id, limit=MAX_RECENT_ALERTS)]
    return jsonify(
        {
            "generated_at": now_iso(),
            "models_ready": shared_models.ready,
            "model_error": shared_models.error,
            "settings": {
                "theme": g.user["theme"],
                "sound_enabled": bool(g.user["sound_enabled"]),
                "auto_switch_alerts": bool(g.user["auto_switch_alerts"]),
                "notifications_enabled": bool(g.user["notifications_enabled"]),
                "models": model_settings_from_record(g.user),
            },
            "cameras": states,
            "recent_alerts": recent_alerts,
            "alerts_total": total_alerts_for_user(user_id),
            "active_alert_camera_id": pick_focus_camera(states),
        }
    )


def stream_camera(camera_id: int):
    placeholder = placeholder_frame("Plithos", "Camera not configured")
    last_seq = -1
    while True:
        with registry_lock:
            runtime = camera_runtimes.get(camera_id)
        if runtime:
            payload, seq = runtime.current_frame_packet()
        else:
            payload, seq = placeholder, -1
        if seq != last_seq or not runtime:
            last_seq = seq
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + payload + b"\r\n"
        time.sleep(ACTIVE_SLEEP if runtime else IDLE_SLEEP)


def snapshot_payload(camera_id: int) -> bytes:
    placeholder = placeholder_frame("Plithos", "Camera not configured")
    with registry_lock:
        runtime = camera_runtimes.get(camera_id)
    return runtime.current_jpeg() if runtime else placeholder


@flask_app.route("/video_feed/<int:camera_id>")
@login_required
@setup_required
def video_feed(camera_id: int):
    if not any(int(row["id"]) == camera_id for row in cameras_for_user(int(g.user["id"]))):
        abort(404)
    return Response(
        stream_camera(camera_id),
        mimetype="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
            "X-Accel-Buffering": "no",
        },
    )


@flask_app.route("/snapshot/<int:camera_id>")
@login_required
@setup_required
def snapshot_feed(camera_id: int):
    if not any(int(row["id"]) == camera_id for row in cameras_for_user(int(g.user["id"]))):
        abort(404)
    return Response(snapshot_payload(camera_id), mimetype="image/jpeg")


@flask_app.route("/setup_preview/<int:source_index>")
@login_required
def setup_preview(source_index: int):
    if source_index < 0 or source_index >= max(CAMERA_SCAN_MAX, 1):
        abort(404)
    return Response(preview_frame_for_source(str(source_index), f"Camera {source_index + 1}"), mimetype="image/jpeg")


@flask_app.route("/monitor")
def monitor_redirect():
    return redirect(url_for("dashboard"))


@flask_app.route("/analytics")
def analytics_redirect():
    return redirect(url_for("logs"))


@flask_app.route("/alert-triage")
def alert_redirect():
    return redirect(url_for("logs"))


@flask_app.route("/health")
def health():
    return jsonify({"ok": True, "models_ready": shared_models.ready, "time": now_iso()})


def build_readme_school_copy() -> str:
    return "\n".join(
        [
            "# Plithos",
            "",
            "Smart safety monitoring for schools.",
            "",
            "Plithos is an AI-powered monitoring system built for school environments. It watches live camera feeds and helps staff respond faster by detecting fights, fire hazards, and missing safety gear in real time.",
            "",
            "## What It Does",
            "",
            "- Detects fights and aggressive behavior",
            "- Detects fire and smoke hazards",
            "- Shows live camera feeds and alerts in a web dashboard",
            "- Sends alert emails with a CSV report",
            "",
            "## Main Pages",
            "",
            "- Landing page",
            "- Login",
            "- Camera setup",
            "- Dashboard",
            "- All cameras",
            "- Logs",
            "- Settings",
            "",
            "## Technology",
            "",
            "- Python",
            "- Flask",
            "- YOLO",
            "- OpenCV",
            "- AWS",
            "",
            "## Privacy",
            "",
            "Plithos focuses on situations, not identity. It does not use facial recognition.",
            "",
            "## Team",
            "",
            "- Mohammed Fardan â€” Team Lead / Cloud / Frontend",
            "- Yousif Alaali â€” Cloud / Database",
            "- Ali Yasser â€” Software Developer / AI Integration",
            "- Salman Ashoor â€” Software Developer / Hardware / R&D",
            "",
            f"Â© {now_local().year} Plithos. All rights reserved.",
        ]
    )


def init_application() -> None:
    init_db()
    shared_models.ensure_loaded()
    refresh_camera_runtimes()


init_application()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plithos Web Server")
    parser.add_argument("--port", default=5000, type=int)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()
    print(f"\n[Plithos] Web interface -> http://localhost:{args.port}\n")
    flask_app.run(host=args.host, port=args.port, threaded=True, use_reloader=False, debug=False)