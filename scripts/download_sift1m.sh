#!/usr/bin/env bash
# Downloads SIFT1M (1M x 128 float vectors, 10k queries, exact ground truth).
set -euo pipefail
DEST="${1:-data}"
mkdir -p "$DEST"
cd "$DEST"
if [ ! -f sift/sift_base.fvecs ]; then
  curl -L -o sift.tar.gz ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz
  tar xzf sift.tar.gz
fi
ls -lh sift/
