# ai_timestamp_reader.py
import os
import cv2
import base64
import json
import datetime as dt
import logging
from typing import Optional

import requests
from dotenv import load_dotenv

load_dotenv()

UF_BASE_URL = os.getenv("UF_AI_BASE_URL", "example.com")
UF_MODEL    = os.getenv("UF_AI_MODEL", "gemma-3-27b-it")
UF_API_KEY  = os.getenv("UF_AI_API_KEY") 


def _get_logger(logger: Optional[logging.Logger]) -> logging.Logger:
    if logger is not None:
        return logger
    logging.basicConfig(level=logging.INFO)
    return logging.getLogger("ai_timestamp_reader")


def read_first_frame(video_path: str, logger: Optional[logging.Logger] = None):
    log = _get_logger(logger)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        log.error("Could not open video: %s", video_path)
        return None
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        log.error("Could not read first frame from: %s", video_path)
        return None
    return frame


def crop_bottom_left_timestamp(frame, height_frac: float = 0.25, width_frac: float = 0.4):
    h, w = frame.shape[:2]
    y1 = int(h * (1.0 - height_frac))
    y2 = h
    x1 = 0
    x2 = int(w * width_frac)
    return frame[y1:y2, x1:x2]


def call_vision_ocr(img_b64: str, logger: Optional[logging.Logger] = None) -> Optional[str]:
    log = logger or logging.getLogger("ai_timestamp_reader")

    if not UF_API_KEY:
        log.error("UF_AI_API_KEY is not set in environment / .env")
        return None

    headers = {
        "Authorization": f"Bearer {UF_API_KEY}",
        "Content-Type": "application/json",
    }

    prompt = (
        "You are an OCR assistant. The image shows a digital timestamp overlay "
        "from the bottom-left corner of a traffic camera video.\n"
        "Read the timestamp text EXACTLY as it appears on the image and normalize it.\n\n"
        "Return ONLY a JSON object, with no extra text, like:\n"
        '{\"timestamp\": \"MM-DD-YYYY HH:MM:SS\"}\n'
        "Examples:\n"
        '{\"timestamp\": \"02-13-2025 05:50:34\"}\n'
        '{\"timestamp\": \"11-14-2025 18:10:47\"}\n'
        "If you cannot read it, return: {\"timestamp\": null}"
    )


    data = {
        "model": UF_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img_b64}"
                        },
                    },
                ],
            }
        ],
        "max_tokens": 128,
    }

    resp = requests.post(f"{UF_BASE_URL}/chat/completions", headers=headers, json=data, timeout=200)
    try:
        resp.raise_for_status()
    except Exception:
        log.error("Vision API error %s: %s", resp.status_code, resp.text[:500])
        return None

    content = resp.json()["choices"][0]["message"]["content"].strip()
    # Try to parse JSON; if the model added extra text, salvage the JSON part
    try:
        obj = json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}")
        if start == -1 or end == -1:
            log.error("Could not find JSON in model response: %r", content)
            return None
        try:
            obj = json.loads(content[start : end + 1])
        except Exception:
            log.error("Failed to parse JSON from: %r", content)
            return None

    ts = obj.get("timestamp")
    if ts is None:
        return None
    if isinstance(ts, str):
        return ts.strip() or None
    return None


def parse_timestamp(ts_str: str, logger: Optional[logging.Logger] = None) -> Optional[dt.datetime]:
    log = _get_logger(logger)
    if not ts_str:
        return None

    # Try a few common formats; extend as needed
    formats = [
        "%m-%d-%Y %H:%M:%S",  # canonical: 02-13-2025 05:50:34
        # optional fallbacks, in case the model or overlay changes:
        "%Y-%m-%d %H:%M:%S",
        "%m/%d/%Y %H:%M:%S",
        "%m/%d/%y %H:%M:%S",
        "%m-%d-%y %H:%M:%S",
    ]

    for fmt in formats:
        try:
            return dt.datetime.strptime(ts_str, fmt)
        except ValueError:
            continue

    log.warning("Could not parse timestamp %r with known formats", ts_str)
    return None


def infer_video_start_time(video_path: str, logger: Optional[logging.Logger] = None) -> Optional[dt.datetime]:
    """
    Load first frame of video, crop timestamp region, call vision model, parse datetime.
    Returns a datetime or None if anything fails.
    """
    log = _get_logger(logger)
    frame = read_first_frame(video_path, logger=log)
    if frame is None:
        return None

    roi = crop_bottom_left_timestamp(frame)
    ok, buf = cv2.imencode(".jpg", roi)
    if not ok:
        log.error("Failed to encode ROI as JPEG")
        return None

    img_b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
    raw_ts = call_vision_ocr(img_b64, logger=log)
    if raw_ts is None:
        log.warning("Vision model did not return a timestamp")
        return None

    log.info("Raw timestamp text from model: %r", raw_ts)
    dt_val = parse_timestamp(raw_ts, logger=log)
    if dt_val is None:
        log.warning("Failed to parse timestamp %r into datetime", raw_ts)
        return None

    return dt_val
