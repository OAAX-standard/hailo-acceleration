# Version 2.0.0 (2026-06-04)
- Introduced OAAX v2 API (`oaax_runtime.h`): replaces the v1 API with a richer interface supporting multi-model loading, per-model config, structured `RuntimeStatus` codes, async enqueue/retrieve with timeouts, and a `runtime_get_info()` diagnostic call.
- Removed old single-model v1 functions (`runtime_initialization`, `runtime_model_loading`, `send_input`, `receive_output`, `runtime_destruction`).
- Queue made generic (`void*` payload) to support both input `Tensors*` and output `OutputItem*`.
- Added `VERSION` and `CHANGELOG.md`.

# Version 1.0.0 (2024-01-01)
- Initial release of the Hailo-8 OAAX runtime and conversion toolchain.
