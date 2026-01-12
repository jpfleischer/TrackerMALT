# Dockerfile
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System deps for OpenCV + general
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev \
    ffmpeg \
    libgl1 libglib2.0-0 \
    ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*


# Install mediainfo from MediaArea official repo (Ubuntu focal)
RUN apt-get update && \
    apt-get install -y wget gnupg && \
    wget -qO - https://mediaarea.net/repo/deb/ubuntu/pubkey.gpg | apt-key add - && \
    echo "deb https://mediaarea.net/repo/deb/ubuntu focal main" > /etc/apt/sources.list.d/mediaarea.list && \
    apt-get update && \
    apt-get install -y mediainfo && \
    rm -rf /var/lib/apt/lists/*


WORKDIR /usr/src/app

# Python deps
COPY requirements.txt /usr/src/app/requirements.txt
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install --no-cache-dir -r requirements.txt


# Copy ALL source files needed at runtime
COPY app.py /usr/src/app/app.py
COPY box_smoothing.py /usr/src/app/box_smoothing.py
COPY trackers /usr/src/app/trackers
COPY ai_timestamp_reader.py /usr/src/app/ai_timestamp_reader.py
COPY clickhouse_client.py /usr/src/app/clickhouse_client.py
COPY homography_store.py /usr/src/app/homography_store.py
COPY H_cam_to_map.npy /usr/src/app/H_cam_to_map.npy
COPY 451.npy /usr/src/app/451.npy
COPY poseB.npy /usr/src/app/poseB.npy

COPY ortho_zoom.tif /usr/src/app/ortho_zoom.tif
COPY truman_zoom.tif /usr/src/app/truman_zoom.tif


# Default command
ENTRYPOINT ["python3", "/usr/src/app/app.py"]
