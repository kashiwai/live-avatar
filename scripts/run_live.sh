#!/usr/bin/env bash
set -euo pipefail

IMAGE=${1:-assets/avatars/example.jpg}

python -m src.live_avatar.main --mode live --image "$IMAGE"

