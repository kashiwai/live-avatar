#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
EXT_DIR="$ROOT_DIR/external"
REPO_DIR="$EXT_DIR/MuseTalk"

mkdir -p "$EXT_DIR"

if [ -d "$REPO_DIR" ] && [ ! -d "$REPO_DIR/.git" ]; then
  echo "Found existing non-git folder at $REPO_DIR. Removing it to re-clone."
  rm -rf "$REPO_DIR"
fi

if [ ! -d "$REPO_DIR/.git" ]; then
  echo "Cloning MuseTalk into $REPO_DIR ..."
  git clone https://github.com/TMElyralab/MuseTalk "$REPO_DIR" || {
    echo "Clone failed. If the directory exists, removing and retrying..."; \
    rm -rf "$REPO_DIR"; \
    git clone https://github.com/TMElyralab/MuseTalk "$REPO_DIR"; \
  }
else
  echo "MuseTalk repo already present. Fetching updates..."
  (cd "$REPO_DIR" && git fetch --all)
fi

cd "$REPO_DIR"

# Try to checkout latest 1.5 tag/branch if present
if git rev-parse --verify origin/v1.5 >/dev/null 2>&1; then
  git checkout v1.5
elif git tag -l | grep -E "(^|-)1\.5(\.|$)" >/dev/null 2>&1; then
  LATEST=$(git tag -l | grep -E "(^|-)1\.5(\.|$)" | tail -n1)
  echo "Checking out tag $LATEST"
  git checkout "$LATEST"
else
  echo "Could not find a v1.5 branch or tag. Staying on default branch."
fi

echo "MuseTalk setup complete. Review its README for installing dependencies."
