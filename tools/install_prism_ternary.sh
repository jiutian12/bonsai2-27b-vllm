#!/usr/bin/env bash
# Install the prism_ternary vLLM plugin into a venv WITHOUT network/build backend:
# put the plugin's src/ on sys.path via a .pth file and create a dist-info that
# carries the `vllm.general_plugins` entry point. Useful when the venv has no
# build backend (no hatchling/setuptools) or you cannot reach PyPI.
#
# usage: VENV=/path/to/venv PLUGIN_SRC=/path/to/bonsai-vllm/src ./install_prism_ternary.sh
set -e

VENV="${VENV:-$HOME/venv}"
PLUGIN_SRC="${PLUGIN_SRC:-$HOME/bonsai-vllm/src}"
SP="$VENV/lib/python$( "$VENV/bin/python" -c 'import sys;print(".".join(map(str,sys.version_info[:2])))' )/site-packages"
DI="$SP/prism_ternary-0.1.0.dist-info"

[ -d "$SP" ]        || { echo "!! site-packages not found: $SP"; exit 1; }
[ -d "$PLUGIN_SRC" ] || { echo "!! plugin src not found: $PLUGIN_SRC"; exit 1; }

echo "$PLUGIN_SRC" > "$SP/prism_ternary_local.pth"
mkdir -p "$DI"
cat > "$DI/METADATA" <<'EOF'
Metadata-Version: 2.1
Name: prism-ternary
Version: 0.1.0
Summary: vLLM plugin for Prism ML ternary (Bonsai) checkpoints
Requires-Python: >=3.12
EOF
cat > "$DI/entry_points.txt" <<'EOF'
[vllm.general_plugins]
prism_ternary = prism_ternary.quant:register
EOF
printf 'prism-ternary\n' > "$DI/INSTALLER"
: > "$DI/RECORD"
echo "installed plugin shim:"
ls -la "$SP/prism_ternary_local.pth" "$DI"

echo "=== verify import + entry point ==="
"$VENV/bin/python" - <<'PY'
import importlib.metadata as md
print("entry points:", [e for e in md.entry_points(group="vllm.general_plugins")])
import prism_ternary.quant as q
print("plugin module:", q.__file__)
print("QUANT_METHOD:", q.QUANT_METHOD)
from vllm.model_executor.layers.quantization import get_quantization_config
print("registered:", get_quantization_config("prism_ternary"))
PY
