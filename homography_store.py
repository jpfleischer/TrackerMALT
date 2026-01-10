#!/usr/bin/env python3
"""
homography_store.py — store/load 3x3 homography matrices in ClickHouse.

Lives under roadpairer/, so it's decoupled from braking/.
"""

from __future__ import annotations

import json
from typing import Optional

import numpy as np

from clickhouse_client import ClickHouseHTTP


def ensure_homography_schema(ch: ClickHouseHTTP) -> None:
    """
    Create a small table for homographies if it doesn't exist.

    Columns:
      - intersection_id, approach_id, tag: logical key
      - h_values: Array(Float64), length 9 (row-major 3x3)
      - note: free-form description
      - created_at: provides version history
    """
    db = ch.db
    sql = f"""
    CREATE TABLE IF NOT EXISTS {db}.homography_metadata
    (
        intersection_id String,
        approach_id     String,
        tag             String,
        h_values        Array(Float64),
        note            String,
        created_at      DateTime DEFAULT now()
    )
    ENGINE = MergeTree
    ORDER BY (intersection_id, approach_id, tag, created_at)
    """
    ch._post_sql(sql, use_db=False)


def save_homography_to_clickhouse(
    ch: ClickHouseHTTP,
    intersection_id: str,
    approach_id: str,
    tag: str,
    H: np.ndarray,
    note: str = "",
) -> None:
    """
    Save a 3x3 homography as one row in homography_metadata.

    - If a row already exists for (intersection_id, approach_id, tag),
      it is deleted before inserting the new one (logical overwrite).
    """
    ensure_homography_schema(ch)

    H = np.asarray(H, dtype=float)
    if H.shape != (3, 3):
        raise ValueError(f"Expected 3x3 homography, got {H.shape}")

    flat = H.reshape(-1)
    vals = ",".join(str(float(x)) for x in flat)

    # naive escaping: keep consistent with existing code
    isect_sql = intersection_id.replace("'", "\\'")
    appr_sql = approach_id.replace("'", "\\'")
    tag_sql  = tag.replace("'", "\\'")
    note_sql = note.replace("'", "\\'")

    db = ch.db

    # 1) Delete any existing row for this logical key
    delete_sql = f"""
    ALTER TABLE {db}.homography_metadata
    DELETE WHERE
        intersection_id = '{isect_sql}' AND
        approach_id     = '{appr_sql}' AND
        tag             = '{tag_sql}'
    """
    ch._post_sql(delete_sql, use_db=False)

    # 2) Insert the new homography row
    insert_sql = f"""
    INSERT INTO {db}.homography_metadata
        (intersection_id, approach_id, tag, h_values, note)
    VALUES
        ('{isect_sql}', '{appr_sql}', '{tag_sql}', [{vals}], '{note_sql}')
    """
    ch._post_sql(insert_sql, use_db=False)


def load_latest_homography_from_clickhouse(
    ch: ClickHouseHTTP,
    intersection_id: str,
    approach_id: str,
    tag: str = "cam_to_map",
) -> Optional[np.ndarray]:
    """
    Load the most recent 3x3 H for a given (intersection, approach, tag).

    Returns:
      - 3x3 np.ndarray if found
      - None if not found
    """
    ensure_homography_schema(ch)

    isect_sql = intersection_id.replace("'", "\\'")
    appr_sql = approach_id.replace("'", "\\'")
    tag_sql = tag.replace("'", "\\'")

    db = ch.db
    sql = f"""
    SELECT h_values
    FROM {db}.homography_metadata
    WHERE intersection_id = '{isect_sql}'
      AND approach_id     = '{appr_sql}'
      AND tag             = '{tag_sql}'
    ORDER BY created_at DESC
    LIMIT 1
    FORMAT JSONEachRow
    """
    resp = ch._post_sql(sql, use_db=False)
    text = resp.text.strip()
    if not text:
        return None

    row = json.loads(text.splitlines()[0])
    arr = np.array(row["h_values"], dtype=float)
    if arr.size != 9:
        raise ValueError(f"Expected 9 elements for 3x3 H, got {arr.size}")
    return arr.reshape(3, 3)
