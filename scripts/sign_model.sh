#!/usr/bin/env bash
# Sign a model artefact using cosign (keyless / OIDC-based signing)
# Usage: ./scripts/sign_model.sh models/bleaching_risk/model.joblib
#
# Requires: cosign installed (https://docs.sigstore.dev/cosign/installation)
# Produces: <model_path>.sig (detached signature)
#
# Verify: cosign verify-blob --signature <model_path>.sig <model_path>

set -euo pipefail

MODEL_PATH="${1:?Usage: $0 <model_path>}"

if ! command -v cosign &> /dev/null; then
    echo "ERROR: cosign not found. Install: https://docs.sigstore.dev/cosign/installation"
    exit 1
fi

echo "Computing SHA256 digest..."
DIGEST=$(sha256sum "$MODEL_PATH" | cut -d' ' -f1)
echo "Digest: $DIGEST"

echo "Signing with cosign (keyless / OIDC)..."
cosign sign-blob "$MODEL_PATH" --output-signature "${MODEL_PATH}.sig" --yes

echo "Signature written: ${MODEL_PATH}.sig"
echo ""
echo "To verify:"
echo "  cosign verify-blob --signature ${MODEL_PATH}.sig ${MODEL_PATH}"
