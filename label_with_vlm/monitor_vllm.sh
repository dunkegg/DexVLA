#!/bin/bash

VLLM_SCRIPT="start_vllm.sh"
LOG_FILE="vllm_monitor.log"
PID_FILE="vllm_server.pid"
PROCESS_IMAGE_PID_FILE="process_images.pid"
CHECK_INTERVAL=30
VLLM_API_URL="http://localhost:8000/v1"

log() {
    echo "$(date '+%F %T') : $*" | tee -a "$LOG_FILE"
}

is_vllm_alive() {
    curl -s --max-time 10 "$VLLM_API_URL" > /dev/null
    return $?  # 0 if success, non-zero if failed
}

kill_process_images() {
    if [ -f "$PROCESS_IMAGE_PID_FILE" ]; then
        pid=$(cat "$PROCESS_IMAGE_PID_FILE")
        if ps -p "$pid" > /dev/null 2>&1; then
            log "Killing previous process_images_action.py (PID $pid)"
            kill "$pid"
        fi
        rm -f "$PROCESS_IMAGE_PID_FILE"
    fi
}

start_process_images() {
    log "Starting process_images_action.py..."
    python3 process_images_action.py > process_images.out 2>&1 &
    echo $! > "$PROCESS_IMAGE_PID_FILE"
}

start_vllm_server() {
    log "Starting vLLM..."
    bash "$VLLM_SCRIPT" > vllm_stdout.log 2>&1 &
    echo $! > "$PID_FILE"
    log "Started vLLM with PID $(cat $PID_FILE)"
}

while true; do
    # Step 1: check vLLM status
    if ! is_vllm_alive; then
        log "vLLM not responsive. Restarting..."
        kill_process_images
        start_vllm_server
        sleep 150  # wait longer for vllm to fully start
        continue
    fi
    log "vllm OK"
    # Step 2: ensure process_images_action.py is running
    if [ ! -f "$PROCESS_IMAGE_PID_FILE" ] || ! ps -p $(cat "$PROCESS_IMAGE_PID_FILE") > /dev/null 2>&1; then
        log "vLLM is alive, but process_images_action.py not running. Starting..."如果这个成立
        start_process_images
    fi
    sleep "$CHECK_INTERVAL"
    log "OK"
done
