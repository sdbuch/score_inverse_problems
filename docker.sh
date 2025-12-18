#!/usr/bin/env bash

set -e

IMAGE_NAME="score-inverse-problems"
CONTAINER_NAME="score-inverse-problems-dev"

function show_help() {
    cat << EOF
Docker management script for score-inverse-problems

Usage: ./docker.sh [command]

Commands:
    build       Build the Docker image
    start       Start the container in detached mode
    stop        Stop the running container
    restart     Restart the container
    enter       Enter the running container with bash
    run         Run a command in the container (e.g., ./docker.sh run python main.py)
    logs        Show container logs
    clean       Remove container and image
    status      Show container status

Examples:
    ./docker.sh build
    ./docker.sh start
    ./docker.sh enter
    ./docker.sh run uv sync
    ./docker.sh run uv run python main.py
EOF
}

function build() {
    echo "Building Docker image for AMD64 platform..."
    docker build --platform linux/amd64 -t "$IMAGE_NAME" .
    echo "✓ Build complete"
}

function start() {
    if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        echo "Container already exists. Starting..."
        docker start "$CONTAINER_NAME"
    else
        echo "Creating and starting container..."
        docker run -d \
            --platform linux/amd64 \
            --name "$CONTAINER_NAME" \
            -v "$(pwd):/workspace" \
            -w /workspace \
            --init \
            "$IMAGE_NAME" \
            tail -f /dev/null
    fi
    echo "✓ Container started"
}

function stop() {
    echo "Stopping container..."
    docker stop "$CONTAINER_NAME" 2>/dev/null || echo "Container not running"
    echo "✓ Container stopped"
}

function restart() {
    stop
    start
}

function enter() {
    if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        echo "Container is not running. Starting it first..."
        start
    fi
    echo "Entering container..."
    docker exec -it "$CONTAINER_NAME" /bin/bash
}

function run_command() {
    if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        echo "Container is not running. Starting it first..."
        start
    fi
    shift # Remove 'run' from arguments
    docker exec -it "$CONTAINER_NAME" "$@"
}

function logs() {
    docker logs -f "$CONTAINER_NAME"
}

function clean() {
    echo "Cleaning up..."
    docker rm -f "$CONTAINER_NAME" 2>/dev/null || true
    docker rmi "$IMAGE_NAME" 2>/dev/null || true
    echo "✓ Cleanup complete"
}

function status() {
    echo "Container status:"
    docker ps -a --filter "name=^${CONTAINER_NAME}$" --format "table {{.Names}}\t{{.Status}}\t{{.Image}}"
}

case "${1:-}" in
    build)
        build
        ;;
    start)
        start
        ;;
    stop)
        stop
        ;;
    restart)
        restart
        ;;
    enter)
        enter
        ;;
    run)
        run_command "$@"
        ;;
    logs)
        logs
        ;;
    clean)
        clean
        ;;
    status)
        status
        ;;
    help|--help|-h|"")
        show_help
        ;;
    *)
        echo "Unknown command: $1"
        echo ""
        show_help
        exit 1
        ;;
esac
