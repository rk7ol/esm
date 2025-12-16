#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

S3_BASE="s3://plm-ml/code/esm-3"
S3_ESM_DIR="${S3_BASE}/esm"
S3_INFER_PY="${S3_BASE}/infer.py"
S3_REQUIREMENT_TXT="${S3_BASE}/requirement.txt"

SRC_ESM_DIR="${SCRIPT_DIR}/esm"
SRC_INFER_PY="${SCRIPT_DIR}/infer.py"
SRC_REQUIREMENTS_TXT="${SCRIPT_DIR}/requirements.txt"

if ! command -v aws >/dev/null 2>&1; then
  echo "ERROR: aws CLI not found. Install and configure AWS CLI before running this script." >&2
  exit 1
fi

if [[ ! -d "${SRC_ESM_DIR}" ]]; then
  echo "ERROR: Source directory not found: ${SRC_ESM_DIR}" >&2
  exit 1
fi
if [[ ! -f "${SRC_INFER_PY}" ]]; then
  echo "ERROR: Source file not found: ${SRC_INFER_PY}" >&2
  exit 1
fi
if [[ ! -f "${SRC_REQUIREMENTS_TXT}" ]]; then
  echo "ERROR: Source file not found: ${SRC_REQUIREMENTS_TXT}" >&2
  exit 1
fi

echo "Uploading directory: ${SRC_ESM_DIR} -> ${S3_ESM_DIR}"
aws s3 sync "${SRC_ESM_DIR}" "${S3_ESM_DIR}" \
  --exclude "*__pycache__*" \
  --exclude "*.pyc"

echo "Uploading file: ${SRC_INFER_PY} -> ${S3_INFER_PY}"
aws s3 cp "${SRC_INFER_PY}" "${S3_INFER_PY}"

echo "Uploading file: ${SRC_REQUIREMENTS_TXT} -> ${S3_REQUIREMENT_TXT}"
aws s3 cp "${SRC_REQUIREMENTS_TXT}" "${S3_REQUIREMENT_TXT}"

echo "Done."
