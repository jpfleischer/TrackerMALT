#!/usr/bin/env python3
from __future__ import annotations

import os
import time
import logging
from pathlib import Path
from typing import Optional, List
import datetime as dt
from clickhouse_client import ClickHouseHTTP
import subprocess

import cv2
import pika
import numpy as np
from ultralytics import YOLO

import rasterio as rio
from affine import Affine


from box_smoothing import BoxSmootherEMA

US_SURVEY_FT_TO_M = 0.30480060960121924


LOG = logging.getLogger("ultra_pipeline")
VIDEO_BASE = Path(os.getenv("VIDEO_BASE_DIR", "/mnt/video_pipeline"))
# === BEV / Homography inputs ===
H_PATH     = os.getenv("H_PATH", "H_cam_to_map.npy")
ORTHO_PATH = os.getenv("ORTHO_PATH", "ortho_zoom.tif")


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

def is_night_window(t: dt.time) -> bool:
    """
    True if time-of-day is in [19:00:01, 23:59:59] OR [00:00:00, 06:59:59].
    """
    return (t >= dt.time(19, 0, 1)) or (t <= dt.time(6, 59, 59))

def ffmpeg_compress(
    in_path: Path,
    out_path: Path,
    crf: int = 28,
    preset: str = "veryfast",
    threads: int = 0,
    use_nvenc: bool = True,
    cq: int = 23,
    nvenc_preset: str = "p4",
) -> None:
    """
    Compress MP4 using either NVENC or CPU.
    If NVENC fails (missing libnvidia-encode.so.1, no GPU passthrough, etc),
    automatically falls back to libx264.
    """
    def run_cmd(cmd: list[str]) -> None:
        subprocess.run(cmd, check=True)

    base = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(in_path)]

    # Try NVENC first if requested
    if use_nvenc:
        cmd_nv = base + [
            "-c:v", "hevc_nvenc",  # or h264_nvenc
            "-preset", nvenc_preset,
            "-rc", "vbr",
            "-cq", str(int(cq)),
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            "-an",
            str(out_path),
        ]
        try:
            run_cmd(cmd_nv)
            return
        except subprocess.CalledProcessError as e:
            LOG.warning("NVENC encode failed; falling back to libx264. err=%s", e)

    # CPU fallback (always works if ffmpeg has libx264)
    cmd_cpu = base + [
        "-c:v", "libx264",
        "-preset", str(preset),
        "-crf", str(int(crf)),
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-an",
        str(out_path),
    ]
    if int(threads) > 0:
        cmd_cpu += ["-threads", str(int(threads))]

    run_cmd(cmd_cpu)


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
    text_thickness: int,   # <-- ADD THIS
    draw_conf: bool = True,
    draw_class: bool = False,
    video_start_dt: Optional[dt.datetime] = None,
    on_detections=None,
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
    out_video_raw = out_dir / f"{video_path.stem}_track_smooth_raw.mp4"
    out_video = out_dir / f"{video_path.stem}_track_smooth.mp4"


    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    cap.release()


    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_video_raw), fourcc, fps, (w, h))
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
                draw_label(frame, label, x1i, y1i - 5, font_scale, text_thickness)


        writer.write(frame)
        frame_idx += 1

    writer.release()
    # compress output video
    use_nvenc = os.getenv("OUT_USE_NVENC", "1") == "1"

    # CPU (libx264) settings
    out_crf = int(os.getenv("OUT_CRF", "28"))            # typical 18–30
    out_preset = os.getenv("OUT_PRESET", "veryfast")     # ultrafast..veryslow
    out_threads = int(os.getenv("OUT_THREADS", "0"))     # 0 = auto

    # GPU (NVENC) settings
    out_cq = int(os.getenv("OUT_CQ", "23"))              # typical 19–28
    out_nvenc_preset = os.getenv("OUT_NVENC_PRESET", "p4")  # p1..p7

    ffmpeg_compress(
        out_video_raw,
        out_video,
        crf=out_crf,
        preset=out_preset,
        threads=out_threads,
        use_nvenc=False,
        cq=out_cq,
        nvenc_preset=out_nvenc_preset,
    )


    # optionally delete the raw
    if os.getenv("KEEP_RAW", "0") != "1":
        try:
            out_video_raw.unlink()
        except Exception:
            pass

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

    font_scale = env_float("FONT_SCALE", 0.45)
    box_thickness = env_int("BOX_THICKNESS", 1)
    text_thickness = env_int("TEXT_THICKNESS", 2)


    draw_conf = env_int("DRAW_CONF", 1) == 1
    draw_class = env_int("DRAW_CLASS", 0) == 1

    # --------------------------
    # BEV / Homography setup (camera px -> ortho px -> meters)
    # --------------------------
    try:
        H = np.load(H_PATH)
        if H.shape != (3, 3):
            raise ValueError(f"H must be 3x3, got {H.shape}")
        LOG.info("Loaded homography: %s", H_PATH)
    except Exception as e:
        LOG.error("Could not load homography %s: %s", H_PATH, e)
        H = None

    ortho = None
    ortho_w = ortho_h = None
    ortho_transform: Optional[Affine] = None

    try:
        ortho = rio.open(ORTHO_PATH)
        ortho_w, ortho_h = ortho.width, ortho.height
        ortho_transform = ortho.transform
        LOG.info("Loaded ortho: %s size=%sx%s CRS=%s", ORTHO_PATH, ortho_w, ortho_h, ortho.crs)
    except Exception as e:
        LOG.error("Could not open ortho %s: %s", ORTHO_PATH, e)
        ortho = None

    def cam_to_map_px(cx: float, cy: float) -> Optional[tuple[float, float]]:
        """(cx,cy) camera px -> (mx,my) ortho px"""
        if H is None:
            return None
        x, y = float(cx), float(cy)
        w = H[2, 0]*x + H[2, 1]*y + H[2, 2]
        if w == 0:
            return None
        mx = (H[0, 0]*x + H[0, 1]*y + H[0, 2]) / w
        my = (H[1, 0]*x + H[1, 1]*y + H[1, 2]) / w
        if not (np.isfinite(mx) and np.isfinite(my)):
            return None
        return float(mx), float(my)


    def map_px_to_meters(mx_px: float, my_px: float) -> Optional[tuple[float, float]]:
        """
        Ortho pixel -> meters.

        ortho_zoom.tif CRS is EPSG:6438 (US survey foot), so affine outputs feet.
        Convert to meters so map_m_x/map_m_y are truly meters.
        """
        if ortho_transform is None:
            return None
        X_ft, Y_ft = ortho_transform * (mx_px, my_px)  # CRS units: US survey feet
        return float(X_ft) * US_SURVEY_FT_TO_M, float(Y_ft) * US_SURVEY_FT_TO_M


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

            # AI timestamp (optional)
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


            # Refuse to process nighttime videos (drop message; do NOT requeue)
            if video_start_dt is not None:
                tod = video_start_dt.time()
                if is_night_window(tod):
                    msg = (
                        f"[ULTRA] Dropping {video_key} (start={video_start_dt}) "
                        "— nighttime window 19:00:01–06:59:59"
                    )
                    LOG.warning(msg)
                    publish_log(ch, log_queue, msg)
                    ch.basic_ack(delivery_tag=method.delivery_tag)  # drop it
                    return
            else:
                # If you want to be strict: drop when timestamp is unknown
                # (optional; remove if you prefer to process unknowns)
                msg = f"[ULTRA] Dropping {video_key} — start time unknown (night filter enabled)"
                LOG.warning(msg)
                publish_log(ch, log_queue, msg)
                ch.basic_ack(delivery_tag=method.delivery_tag)
                return

            publish_log(ch, log_queue, f"[ULTRA] Tracking {video_path.name} (alpha={smooth_alpha})")

            ch_rows: List[str] = []

            def on_detections(frame_idx, fps, ids, clss, confs, xyxy_i, vstart):
                if ch_client is None:
                    return
                if ids is None or len(ids) == 0:
                    return

                secs = float(frame_idx) / float(fps or 15.0)

                # DateTime64(3) fallback (non-null)
                if vstart is None:
                    ts = dt.datetime(1970, 1, 1) + dt.timedelta(seconds=secs)
                else:
                    ts = vstart + dt.timedelta(seconds=secs)

                ts_str = ts.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

                for j, tid in enumerate(ids):
                    x1, y1, x2, y2 = map(int, xyxy_i[j])
                    cx = 0.5 * (x1 + x2)
                    cy = float(y2)  # bottom-center footpoint

                    cls_name = str(int(clss[j])) if clss is not None else "unk"

                    map_px_x = map_px_y = map_m_x = map_m_y = None
                    mp = cam_to_map_px(cx, cy)

                    if mp is not None and ortho_w is not None and ortho_h is not None:
                        mx_px, my_px = mp
                        if 0 <= mx_px < ortho_w and 0 <= my_px < ortho_h:
                            map_px_x, map_px_y = mx_px, my_px
                            mxy = map_px_to_meters(mx_px, my_px)
                            if mxy is not None:
                                map_m_x, map_m_y = mxy

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
                            "" if map_px_x is None else f"{map_px_x:.3f}",
                            "" if map_px_y is None else f"{map_px_y:.3f}",
                            "" if map_m_x is None else f"{map_m_x:.3f}",
                            "" if map_m_y is None else f"{map_m_y:.3f}",
                        ]) + "\n"
                    )

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
                text_thickness=text_thickness,
                draw_conf=draw_conf,
                draw_class=draw_class,
                video_start_dt=video_start_dt,
                on_detections=on_detections,
            )

            # final ClickHouse flush
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

            # ACK message (success)
            ch.basic_ack(delivery_tag=method.delivery_tag)

        except Exception as e:
            LOG.exception("Processing failed")
            publish_log(ch, log_queue, f"[ULTRA][ERROR] Exception: {e}")
            # Don't poison-loop forever
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)

    # IMPORTANT: basic_consume/start_consuming MUST be OUTSIDE callback
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
        try:
            if ortho is not None:
                ortho.close()
        except Exception:
            pass



if __name__ == "__main__":
    main()
