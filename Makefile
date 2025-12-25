# Makefile
IMAGE ?= ultra-video-pipeline
NAME  ?= ultra-video-pipeline-1

# RabbitMQ
qname     ?= garbage
IN_QUEUE  ?= video_filename2
LOG_QUEUE ?= logs

# Paths (host -> container)
HOST_DATA ?= /data
HOST_PIPE ?= /data/video_pipeline


# GPU
GPU ?= all

# --- Hardcoded Ultralytics runtime defaults ---
MODEL   := yolo11s.pt
TRACKER := /usr/src/app/trackers/botsort_traffic.yaml
IMGSZ   := 960
CONF    := 0.35
IOU     := 0.50
CLASSES := 2,3,5,7
OUT_DIR := /mnt/video_pipeline/tracking_ultra
DEVICE  := 0

# --- Hardcoded smoothing + drawing defaults ---
SMOOTH_ALPHA         := 0.15
SMOOTH_STATIONARY_PX := 7
SMOOTH_MAX_SIZE_FRAC := 0.015
FONT_SCALE           := 0.40
TEXT_THICKNESS	     := 2
BOX_THICKNESS        := 3
DRAW_CONF            := 1
DRAW_CLASS           := 0

default: build run

.PHONY: build run stop down logs

build:
	docker build -t $(IMAGE) .

run: network
	-docker rm -f $(NAME) >/dev/null 2>&1 || true
	docker run -d \
		--network clickhouse \
		--name $(NAME) \
		--gpus $(GPU) \
		--env-file .env \
		-v "$(HOST_DATA):/mnt" \
		-v "$(HOST_PIPE):/mnt/video_pipeline" \
		-e USE_AI_TS=1 \
		-e CH_ENABLED=1 \
		-e IN_QUEUE=$(IN_QUEUE) \
		-e LOG_QUEUE=$(LOG_QUEUE) \
		-e MODEL=$(MODEL) \
		-e TRACKER=$(TRACKER) \
		-e IMGSZ=$(IMGSZ) \
		-e CONF=$(CONF) \
		-e IOU=$(IOU) \
		-e CLASSES=$(CLASSES) \
		-e OUT_DIR=$(OUT_DIR) \
		-e DEVICE=$(DEVICE) \
		-e SMOOTH_ALPHA=$(SMOOTH_ALPHA) \
		-e SMOOTH_STATIONARY_PX=$(SMOOTH_STATIONARY_PX) \
		-e SMOOTH_MAX_SIZE_FRAC=$(SMOOTH_MAX_SIZE_FRAC) \
		-e FONT_SCALE=$(FONT_SCALE) \
		-e BOX_THICKNESS=$(BOX_THICKNESS) \
		-e DRAW_CONF=$(DRAW_CONF) \
		-e DRAW_CLASS=$(DRAW_CLASS) \
		$(IMAGE)

## Ensure docker network exists
network:
	@docker network inspect clickhouse >/dev/null 2>&1 || \
		docker network create clickhouse


logs:
	docker logs -f $(NAME)

stop down:
	-docker rm -f $(NAME) >/dev/null 2>&1 || true
