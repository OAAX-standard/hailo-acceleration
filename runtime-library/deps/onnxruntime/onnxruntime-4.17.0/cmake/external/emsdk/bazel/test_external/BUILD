load("@emsdk//emscripten_toolchain:wasm_rules.bzl", "wasm_cc_binary")

cc_binary(
    name = "hello-world",
    srcs = ["hello-world.cc"],
)

wasm_cc_binary(
    name = "hello-world-wasm",
    cc_target = ":hello-world",
)

BASE_LINKOPTS = [
    "--bind",  # Enable embind
    "-sMODULARIZE",
]

RELEASE_OPTS = [
    "--closure=1",  # Run the closure compiler
]

DEBUG_OPTS = [
    "--closure=0",  # Do not use closure
]

config_setting(
    name = "release_opts",
    values = {"compilation_mode": "opt"},
)

config_setting(
    name = "debug_opts",
    values = {"compilation_mode": "dbg"},
)

cc_binary(
    name = "hello-embind",
    srcs = ["hello-embind.cc"],
    linkopts = select({
        ":debug_opts": BASE_LINKOPTS + DEBUG_OPTS,
        ":release_opts": BASE_LINKOPTS + RELEASE_OPTS,
        "//conditions:default": BASE_LINKOPTS + RELEASE_OPTS,
    }),
    # This target won't build successfully on its own because of missing emscripten
    # headers etc. Therefore, we hide it from wildcards.
    tags = ["manual"],
)

wasm_cc_binary(
    name = "hello-embind-wasm",
    cc_target = ":hello-embind",
)
