#!/usr/bin/env python3
"""
clickhouse_client.py — lightweight HTTP client for inserting data into ClickHouse.

Usage example:
    from clickhouse_client import ClickHouseHTTP

    ch = ClickHouseHTTP(
        host="localhost",
        port=8123,
        user="default",
        password="",
        database="trajectories"
    )

    rows = [
        "video1.mp4,0,0.0,2025-11-08 10:00:00.000,1,car,100,200,50,60,12.34,56.78\n",
        "video1.mp4,1,0.07,2025-11-08 10:00:00.070,1,car,101,201,51,61,12.35,56.79\n",
    ]
    ch.insert_csv_rows(rows)
"""

import os
import time
import logging
from typing import List, Optional

import requests

# ------------------------------------------------------------------------------


class ClickHouseHTTP:
    """Tiny HTTP client for ClickHouse CSV ingestion."""

    def __init__(
        self,
        host: str = "clickhouse_server_auto",
        port: int = 8123,
        user: str = "default",
        password: str = "",
        database: str = "trajectories",
        timeout: int = 5,
        retries: int = 2,
        logger: Optional[logging.Logger] = None,
    ):
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.db = database
        self.timeout = timeout
        self.retries = retries
        self.logger = logger or logging.getLogger("ClickHouseHTTP")

        self._session = requests.Session()
        self._auth = (self.user, self.password) if (self.user or self.password) else None

        # Flag so we only try to create schema once per process
        self._schema_ensured = False

        # Precomputed ingest URL with settings for correct NULL handling
        self._ingest_url = (
            f"http://{self.host}:{self.port}/"
            f"?database={self.db}"
            f"&query=INSERT%20INTO%20{self.db}.raw%20FORMAT%20CSVWithNames"
            "&input_format_csv_empty_as_default=0"
            "&input_format_null_as_default=0"
            "&format_csv_null_representation="  # empty string means no special NULL token
        )


        self.header = ",".join([
            "video", "frame", "secs", "timestamp", "track_id", "class",
            "cam_x", "cam_y", "map_px_x", "map_px_y", "map_m_x", "map_m_y"
        ]) + "\n"

        # Ensure DB + table exist (best-effort; failures are logged)
        try:
            self.ensure_schema()
        except Exception as e:
            self.logger.error(f"Failed to ensure ClickHouse schema: {e}")


    # ------------------------------------------------------------------

    def video_already_ingested(self, video: str) -> bool:
        """
        Return True if any rows already exist in {db}.raw for the given video.
        """
        # Very simple escaping of single quotes
        safe_video = video.replace("'", "\\'")
        sql = f"SELECT count() FROM {self.db}.raw WHERE video = '{safe_video}'"

        try:
            resp = self._post_sql(sql, use_db=False)
            count = int(resp.text.strip())
            return count > 0
        except Exception as e:
            self.logger.error(f"Failed to check existing rows for video={video!r}: {e}")
            # If ClickHouse is down or this fails, better to process than silently skip
            return False

    # ------------------------------------------------------------------

    def _post_sql(self, sql: str, use_db: bool = False, timeout: Optional[int] = None):
        """
        Helper: POST raw SQL to ClickHouse HTTP interface.
        If use_db is True, sends ?database=<self.db> so unqualified table names use that DB.
        """
        url = f"http://{self.host}:{self.port}/"
        if use_db:
            url += f"?database={self.db}"

        r = self._session.post(
            url,
            data=sql.encode("utf-8"),
            auth=self._auth,
            timeout=timeout or max(self.timeout, 10),
        )
        r.raise_for_status()
        return r

    # ------------------------------------------------------------------

    def ensure_schema(self) -> None:
        """
        Create the database and table if they do not exist.

        This is idempotent and safe to call multiple times; it will
        early-return after the first success in this process.
        """
        if self._schema_ensured:
            return

        # 1) Create database (must not rely on ?database before it exists)
        sql_db = f"CREATE DATABASE IF NOT EXISTS {self.db}"
        self.logger.info(f"Ensuring ClickHouse database exists: {self.db}")
        self._post_sql(sql_db, use_db=False)

        # 2) Create table in that database
        # Schema matches the CSV header:
        # video,frame,secs,timestamp,track_id,class,cam_x,cam_y,map_px_x,map_px_y,map_m_x,map_m_y
        sql_table = f"""
        CREATE TABLE IF NOT EXISTS {self.db}.raw
        (
            video      String,
            frame      UInt32,
            secs       Float64,
            timestamp  DateTime64(3),
            track_id   UInt32,
            class      String,
            cam_x      Float64,
            cam_y      Float64,
            map_px_x   Float64,
            map_px_y   Float64,
            map_m_x    Float64,
            map_m_y    Float64
        )
        ENGINE = MergeTree
        ORDER BY (video, track_id, frame)
        """
        self.logger.info(f"Ensuring ClickHouse table exists: {self.db}.raw")
        self._post_sql(sql_table, use_db=False)

        self._schema_ensured = True
        self.logger.info(f"ClickHouse schema ensured for {self.db}.raw")

    # ------------------------------------------------------------------

    def ping(self) -> bool:
        """Returns True if ClickHouse responds to SELECT 1."""
        try:
            r = self._session.get(
                f"http://{self.host}:{self.port}/?query=SELECT%201",
                auth=self._auth,
                timeout=self.timeout,
            )
            return r.ok
        except Exception:
            return False

    # ------------------------------------------------------------------

    def insert_csv_rows(self, rows: List[str]) -> None:
        """
        Insert a list of CSV lines into ClickHouse.
        Each row must already be comma-separated and end with '\n'.
        """
        if not rows:
            return

        # Make sure schema exists before we try to insert
        if not self._schema_ensured:
            try:
                self.ensure_schema()
            except Exception as e:
                self.logger.error(f"ensure_schema() failed before insert: {e}")

        payload = (self.header + "".join(rows)).encode("utf-8")

        # Use a clear "attempt count" so logs look sane.
        total_attempts = self.retries + 1
        for attempt in range(total_attempts):
            try:
                r = self._session.post(
                    self._ingest_url,
                    data=payload,
                    auth=self._auth,
                    timeout=max(self.timeout, 30),
                )
                if r.status_code == 200:
                    if attempt > 0:
                        self.logger.info(
                            f"ClickHouse insert succeeded after retry #{attempt}"
                        )
                    return
                else:
                    raise RuntimeError(
                        f"ClickHouse HTTP {r.status_code}: {r.text[:200]}"
                    )
            except Exception as e:
                self.logger.error(
                    f"ClickHouse insert failed (attempt {attempt + 1}/{total_attempts}): {e}"
                )
                if attempt == self.retries:
                    # re-raise on final failure
                    raise
                time.sleep(1.5 * (attempt + 1))

    # ------------------------------------------------------------------

    def insert_dataframe(self, df, table: str = "trajectories.raw") -> None:
        """
        Optional helper: insert a pandas DataFrame (must have same column order).
        """
        import io

        # Make sure schema exists before DataFrame insert
        if not self._schema_ensured:
            try:
                self.ensure_schema()
            except Exception as e:
                self.logger.error(f"ensure_schema() failed before DF insert: {e}")

        buf = io.StringIO()
        df.to_csv(buf, index=False)
        payload = buf.getvalue().encode("utf-8")

        url = (
            f"http://{self.host}:{self.port}/"
            f"?database={self.db}"
            f"&query=INSERT%20INTO%20{table}%20FORMAT%20CSVWithNames"
        )
        r = self._session.post(
            url,
            data=payload,
            auth=self._auth,
            timeout=max(self.timeout, 30),
        )
        r.raise_for_status()


# ------------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ch = ClickHouseHTTP()
    print("Ping:", ch.ping())
