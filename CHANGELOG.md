# Unreleased
- Fixed: queue overflow dropped the oldest item without freeing its payload (leak); the runtime now frees dropped inputs/outputs via `Queue.drop_item`.
- Fixed: `runtime_retrieve_output()` before `runtime_load_models()` dereferenced a NULL queue; it now returns `RUNTIME_STATUS_MODEL_NOT_LOADED`.
- Fixed: a failed worker-thread creation left already-started threads unjoined during cleanup.
- Fixed: logger leaked when `runtime_init` failed to get the ORT API.
- Docs: `n_threads`, `n_replicas`, `perf_mode` are `runtime_init` keys (`ModelConfig.config` is unused); corrected S3 paths, supported HailoRT versions (4.17–4.20) and the AGPL note for Ultralytics models.
- CI: runtime packages built per HailoRT version; generated protobuf files excluded from clang-format lint; added hardware-free `queue_test`.
- Tests: removed hard-coded local paths (env-var overrides); untracked `__pycache__`.
- `ThirdPartyNotices.txt`: HailoRT is MIT (firmware EULA noted); added protobuf-c and COCO images.

# Version 2.0.0 (2026-06-04)
- Introduced OAAX v2 API (`oaax_runtime.h`): replaces the v1 API with a richer interface supporting multi-model loading, per-model config, structured `RuntimeStatus` codes, async enqueue/retrieve with timeouts, and a `runtime_get_info()` diagnostic call.
- Removed old single-model v1 functions (`runtime_initialization`, `runtime_model_loading`, `send_input`, `receive_output`, `runtime_destruction`).
- Queue made generic (`void*` payload) to support both input `Tensors*` and output `OutputItem*`.
- Added `VERSION` and `CHANGELOG.md`.

# Version 1.0.0 (2024-01-01)
- Initial release of the Hailo-8 OAAX runtime and conversion toolchain.
