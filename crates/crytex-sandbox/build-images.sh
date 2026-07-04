#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
TAG="${1:-latest}"

for img in rust node python; do
    echo "Building crytex/sandbox-${img}:${TAG} ..."
    docker build \
        -t "crytex/sandbox-${img}:${TAG}" \
        -f "${SCRIPT_DIR}/images/${img}/Dockerfile" \
        "${SCRIPT_DIR}/images/${img}/"
done

echo "Done. Available images:"
docker images 'crytex/sandbox-*'
