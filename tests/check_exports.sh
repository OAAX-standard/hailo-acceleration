#!/bin/bash
# Hardware-free guard: libRuntimeLibrary.so must export only the OAAX v2 API and contain no
# GNU_UNIQUE symbols. ORT's DataTypeRegistry singleton is one of those; with it, a runtime
# reselect (new copy of the library loaded next to the pinned old one) fails every model load
# with "MLDataType for: tensor(float) is not currently registered".
# Usage: tests/check_exports.sh path/to/libRuntimeLibrary.so
set -euo pipefail

lib="${1:?usage: $0 libRuntimeLibrary.so}"
expected="runtime_cleanup runtime_enqueue_input runtime_get_error runtime_get_info runtime_get_name runtime_get_version runtime_init runtime_load_models runtime_retrieve_output"

unique=$(readelf -s --wide "$lib" | awk '$5=="UNIQUE"' | wc -l)
exported=$(readelf --dyn-syms --wide "$lib" | awk '$7!="UND" && $8 ~ /^runtime_/ {sub(/@.*/, "", $8); print $8}' | sort | tr '\n' ' ' | sed 's/ $//')
total=$(readelf --dyn-syms --wide "$lib" | awk '$7!="UND" && $8!="" && $8!="Name"' | wc -l)

status=0
[ "$unique" -eq 0 ] || { echo "FAIL: $unique GNU_UNIQUE symbols in $lib"; status=1; }
[ "$exported" = "$expected" ] || { echo "FAIL: exported OAAX symbols differ: $exported"; status=1; }
# 9 functions + the version node name
[ "$total" -le 10 ] || { echo "FAIL: $total dynamic symbols exported, expected only the OAAX API"; status=1; }
[ "$status" -eq 0 ] && echo "OK: $lib exports only the OAAX API, no GNU_UNIQUE symbols"
exit "$status"
