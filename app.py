#!/usr/bin/env python3
from __future__ import annotations

import os
import time
import logging
from pathlib import Path
from typing import Optional, List
import datetime as dt
from clickhouse_client import ClickHouseHTTP

import cv2
import pika
import numpy as np
from ultralytics import YOLO

from box_smoothing import BoxSmootherEMA


LOG = logging.getLogger("ultra_pipeline")
VIDEO_BASE = Path(os.getenv("VIDEO_BASE_DIR", "/mnt/video_pipeline"))


def env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    return default if v is None or v.strip() == "" else float(v)


def env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    return default if v is None or v.strip() == "" else int(v)


def env_str(name: str, default: str) -> str:
    v = os.getenv(name)
    return default if v is None else v


def parse_classes(s: str) -> Optional[List[int]]:
    s = (s or "").strip()
    if not s:
        return None
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def publish_log(channel: pika.adapters.blocking_connection.BlockingChannel, log_queue: str, msg: str) -> None:
    try:
        channel.basic_publish(exchange="", routing_key=log_queue, body=msg.encode("utf-8"))
    except Exception:
        LOG.exception("Failed to publish log message to queue=%s", log_queue)


def draw_label(
    frame: np.ndarray,
    text: str,
    x: int,
    y: int,
    font_scale: float,
    thickness: int,
) -> None:
    """
    Small readable label with black outline.
    """
    y = max(0, y)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (255, 255, 255), thickness, cv2.LINE_AA)


def process_video_smoothed(
    model: YOLO,
    video_path: Path,
    out_root: Path,
    tracker_yaml: str,
    imgsz: int,
    conf: float,
    iou: float,
    classes: Optional[List[int]],
    device: str,
    smooth_alpha: float,
    smooth_stationary_px: float,
    smooth_max_size_change_stationary: float,
    font_scale: float,
    box_thickness: int,
    draw_conf: bool = True,
    draw_class: bool = False,
    video_start_dt: Optional[dt.datetime] = None,
    on_detections=None,  # callable(frame_idx, fps, ids, clss, confs, xyxy_i, video_start_dt)
) -> Path:

    """
    Runs Ultralytics tracking and writes an annotated MP4 with per-track smoothed boxes.
    Returns path to the output video.
    """
    if not video_path.exists():
        raise FileNotFoundError(str(video_path))

    # Output folder per video
    out_dir = out_root / video_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    out_video = out_dir / f"{video_path.stem}_track_smooth.mp4"

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_video), fourcc, fps, (w, h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open VideoWriter: {out_video}")

    smoother = BoxSmootherEMA(
        alpha=smooth_alpha,
        max_age_frames=int(fps * 10),  # keep state ~10s
        stationary_px=smooth_stationary_px,
        max_size_change_stationary=smooth_max_size_change_stationary,
    )

    # Stream inference frame-by-frame
    results = model.track(
        source=str(video_path),
        tracker=tracker_yaml,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        classes=classes,
        device=device,
        persist=True,
        stream=True,
        verbose=True,
        save=False,      # we write ourselves
    )

    frame_idx = 0
    for r in results:
        frame = r.orig_img  # BGR numpy array
        if frame is None:
            break

        # Tracking outputs
        if r.boxes is not None and r.boxes.id is not None and len(r.boxes) > 0:
            ids = r.boxes.id.cpu().numpy().astype(int)

            xyxy = r.boxes.xyxy.cpu().numpy().astype(np.float32)  # (N,4)
            xyxy_s = smoother.update_many(frame_idx, ids, xyxy)

            confs = r.boxes.conf.cpu().numpy() if r.boxes.conf is not None else None
            clss = r.boxes.cls.cpu().numpy().astype(int) if r.boxes.cls is not None else None

            # Build clipped int boxes once (used for drawing + ClickHouse)
            xyxy_i = xyxy_s.astype(np.int32, copy=True)
            xyxy_i[:, 0] = np.clip(xyxy_i[:, 0], 0, w - 1)
            xyxy_i[:, 2] = np.clip(xyxy_i[:, 2], 0, w - 1)
            xyxy_i[:, 1] = np.clip(xyxy_i[:, 1], 0, h - 1)
            xyxy_i[:, 3] = np.clip(xyxy_i[:, 3], 0, h - 1)

            if on_detections is not None:
                try:
                    on_detections(frame_idx, fps, ids, clss, confs, xyxy_i, video_start_dt)
                except Exception:
                    LOG.exception("on_detections hook failed (continuing)")


            for i, tid in enumerate(ids):
                x1i, y1i, x2i, y2i = map(int, xyxy_i[i])


                # clip to frame bounds
                x1i = max(0, min(x1i, w - 1))
                y1i = max(0, min(y1i, h - 1))
                x2i = max(0, min(x2i, w - 1))
                y2i = max(0, min(y2i, h - 1))

                if x2i <= x1i or y2i <= y1i:
                    continue

                cv2.rectangle(frame, (x1i, y1i), (x2i, y2i), (0, 255, 0), box_thickness)

                parts = [f"id:{tid}"]
                if draw_class and clss is not None:
                    parts.insert(0, f"c{int(clss[i])}")
                if draw_conf and confs is not None:
                    parts.append(f"{float(confs[i]):.2f}")

                label = " ".join(parts)
                draw_label(frame, label, x1i, y1i - 5, font_scale, max(1, box_thickness))

        writer.write(frame)
        frame_idx += 1

    writer.release()
    cap.release()
    return out_video


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)8s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Required env for RabbitMQ
    rbbt_usr = os.getenv("RBBT_USR")
    rbbt_psw = os.getenv("RBBT_PSW")
    rbbt_ip = os.getenv("RBBT_IP")
    qname = os.getenv("qname")  # output queue name (keep your convention)

    if not rbbt_usr or not rbbt_psw or not rbbt_ip or not qname:
        raise SystemExit("Missing env vars: RBBT_USR, RBBT_PSW, RBBT_IP, qname")

    in_queue = env_str("IN_QUEUE", "video_filename2")
    log_queue = env_str("LOG_QUEUE", "logs")

    # Ultralytics params
    model_path = env_str("MODEL", "yolo11s.pt")
    tracker_yaml = env_str("TRACKER", "/usr/src/app/trackers/botsort_traffic.yaml")
    imgsz = env_int("IMGSZ", 960)
    conf = env_float("CONF", 0.35)
    iou = env_float("IOU", 0.50)
    classes = parse_classes(env_str("CLASSES", "2,3,5,7"))
    device = env_str("DEVICE", "0")  # "0" for GPU:0, or "cpu"

    out_root = Path(env_str("OUT_DIR", "/mnt/video_pipeline/tracking_ultra"))

    # Smoothing + drawing knobs
    smooth_alpha = env_float("SMOOTH_ALPHA", 0.20)
    smooth_stationary_px = env_float("SMOOTH_STATIONARY_PX", 2.0)
    smooth_max_size_change_stationary = env_float("SMOOTH_MAX_SIZE_FRAC", 0.02)

    font_scale = env_float("FONT_SCALE", 0.45)          # smaller labels
    box_thickness = env_int("BOX_THICKNESS", 1)

    draw_conf = env_int("DRAW_CONF", 1) == 1
    draw_class = env_int("DRAW_CLASS", 0) == 1

    # --------------------------
    # Optional AI timestamp reader
    # --------------------------
    use_ai_ts = env_int("USE_AI_TS", 0) == 1  # set USE_AI_TS=1 to enable

    # --------------------------
    # ClickHouse setup (uses your existing clickhouse_client.py schema/table)
    # --------------------------
    ch_enabled = env_int("CH_ENABLED", 0) == 1  # set CH_ENABLED=1 to enable
    ch_client: Optional[ClickHouseHTTP] = None
    if ch_enabled:
        CH_HOST = env_str("CH_HOST", "clickhouse_server_auto")
        CH_PORT = env_int("CH_PORT", 8123)
        CH_USER = env_str("CH_USER", "default")
        CH_PASSWORD = env_str("CH_PASSWORD", "")
        CH_DB = env_str("CH_DB", "trajectories")

        ch_client = ClickHouseHTTP(
            host=CH_HOST,
            port=CH_PORT,
            user=CH_USER,
            password=CH_PASSWORD,
            database=CH_DB,
            logger=LOG,
        )

    # If missing file comes in, requeue or drop (default drop)
    missing_requeue = env_int("MISSING_REQUEUE", 0) == 1


    # Init model once
    LOG.info("Loading YOLO model: %s", model_path)
    model = YOLO(model_path)

    # RabbitMQ setup
    creds = pika.PlainCredentials(rbbt_usr, rbbt_psw)
    params = pika.ConnectionParameters(
        host=rbbt_ip,
        port=5672,
        virtual_host="/",
        credentials=creds,
        heartbeat=600,
        blocked_connection_timeout=300,
    )
    connection = pika.BlockingConnection(params)
    channel = connection.channel()

    channel.queue_declare(queue=in_queue, durable=False)
    channel.queue_declare(queue=log_queue, durable=False)
    channel.queue_declare(queue=qname, durable=False)

    channel.basic_qos(prefetch_count=1)

    LOG.info("Consuming from queue=%s; output queue=%s; logs=%s", in_queue, qname, log_queue)

    def callback(ch, method, properties, body: bytes):
        t0 = time.time()
        try:
            filename = body.decode("utf-8").strip()
            LOG.info("RX message bytes=%d body=%r", len(body), body[:200])
            LOG.info("RX filename=%r", filename)

            video_path = Path(filename)

            # If message is a bare filename or relative path, resolve it under /mnt/video_pipeline
            if not video_path.is_absolute():
                video_path = (VIDEO_BASE / video_path).resolve()

            video_key = video_path.name

            # --- ClickHouse skip check (optional) ---
            if ch_client is not None:
                try:
                    if ch_client.video_already_ingested(video_key):
                        msg = f"[ULTRA] Skipping {video_key} — already in ClickHouse"
                        LOG.warning(msg)
                        publish_log(ch, log_queue, msg)
                        ch.basic_ack(delivery_tag=method.delivery_tag)
                        return
                except Exception as e:
                    publish_log(ch, log_queue, f"[ULTRA][WARN] CH skip-check failed for {video_key}: {e}")
                    # fall through and process anyway


            # publish_log(channel, log_queue, f"[ULTRA] Received {filename}")
            LOG.info("Processing video: orig=%r resolved=%s", filename, video_path)


            if not video_path.exists():
                msg = f"[ULTRA][ERROR] Missing file inside container: {filename} (resolved={video_path})"
                LOG.error(msg)
                publish_log(ch, log_queue, msg)

                # IMPORTANT: requeue so it doesn't just disappear
                ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
                return
            
            video_start_dt: Optional[dt.datetime] = None
            if use_ai_ts:
                try:
                    from ai_timestamp_reader import infer_video_start_time
                    video_start_dt = infer_video_start_time(str(video_path), logger=LOG)
                    if video_start_dt is not None:
                        publish_log(ch, log_queue, f"[ULTRA] AI start time {video_key}: {video_start_dt}")
                    else:
                        publish_log(ch, log_queue, f"[ULTRA][WARN] AI start time not found for {video_key}")
                except Exception as e:
                    publish_log(ch, log_queue, f"[ULTRA][WARN] AI timestamp reader failed: {e}")
                    video_start_dt = None


            publish_log(ch, log_queue, f"[ULTRA] Tracking {video_path.name} (alpha={smooth_alpha})")

            ch_rows: List[str] = []

            def on_detections(frame_idx, fps, ids, clss, confs, xyxy_i, vstart):
                if ch_client is None:
                    return
                if ids is None or len(ids) == 0:
                    return

                secs = float(frame_idx) / float(fps or 15.0)

                # Your table requires DateTime64(3) NOT Nullable; so if AI TS missing,
                # we must still provide a valid timestamp. Use UNIX epoch as fallback.
                if vstart is None:
                    ts = dt.datetime(1970, 1, 1) + dt.timedelta(seconds=secs)
                else:
                    ts = vstart + dt.timedelta(seconds=secs)

                ts_str = ts.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

                # Insert rows matching clickhouse_client.py header:
                # video,frame,secs,timestamp,track_id,class,cam_x,cam_y,map_px_x,map_px_y,map_m_x,map_m_y
                # For Ultralytics we don't have map coords yet, so send blanks.
                for i, tid in enumerate(ids):
                    x1, y1, x2, y2 = map(int, xyxy_i[i])
                    cx = 0.5 * (x1 + x2)
                    cy = float(y2)  # bottom-center footpoint (like your FastMOT code)
                    cls_name = str(int(clss[i])) if clss is not None else "unk"

                    ch_rows.append(
                        ",".join([
                            video_key,
                            str(int(frame_idx)),
                            f"{secs:.6f}",
                            ts_str,
                            str(int(tid)),
                            cls_name,
                            f"{cx:.3f}",
                            f"{cy:.3f}",
                            "", "", "", "",   # map_px_x, map_px_y, map_m_x, map_m_y
                        ]) + "\n"
                    )

                # flush periodically to keep memory bounded
                if len(ch_rows) >= 5000:
                    ch_client.insert_csv_rows(ch_rows)
                    ch_rows.clear()


            out_video = process_video_smoothed(
                model=model,
                video_path=video_path,
                out_root=out_root,
                tracker_yaml=tracker_yaml,
                imgsz=imgsz,
                conf=conf,
                iou=iou,
                classes=classes,
                device=device,
                smooth_alpha=smooth_alpha,
                smooth_stationary_px=smooth_stationary_px,
                smooth_max_size_change_stationary=smooth_max_size_change_stationary,
                font_scale=font_scale,
                box_thickness=box_thickness,
                draw_conf=draw_conf,
                draw_class=draw_class,
                video_start_dt=video_start_dt,
                on_detections=on_detections,
            )
            if ch_client is not None and ch_rows:
                try:
                    ch_client.insert_csv_rows(ch_rows)
                    ch_rows.clear()
                except Exception as e:
                    publish_log(ch, log_queue, f"[ULTRA][ERROR] Final ClickHouse flush failed: {e}")


            elapsed = time.time() - t0
            publish_log(ch, log_queue, f"[ULTRA] Done {video_path.name} in {elapsed:.2f}s -> {out_video}")

            # notify downstream via qname: output path
            ch.basic_publish(exchange="", routing_key=qname, body=str(out_video).encode("utf-8"))

            ch.basic_ack(delivery_tag=method.delivery_tag)

        except Exception as e:
            LOG.exception("Processing failed")
            publish_log(ch, log_queue, f"[ULTRA][ERROR] Exception: {e}")
            # Don't poison-loop forever
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)

    channel.basic_consume(queue=in_queue, on_message_callback=callback, auto_ack=False)

    try:
        channel.start_consuming()
    finally:
        try:
            channel.close()
        except Exception:
            pass
        try:
            connection.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
