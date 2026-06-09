#include "oaax_runtime.h"
#include "runtime_utils.h"
#include "logger.h"
#include "queue.h"
#include <errno.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#ifndef ONNXRUNTIME_API_VERSION
#define ONNXRUNTIME_API_VERSION 15
#endif

#define QUEUE_CAPACITY  100
#define WORKER_POLL_MS  5
#define MAX_MODELS      8
#define RUNTIME_VERSION "2.0.0"
#define RUNTIME_NAME    "OAAX Hailo Runtime"

/* ── Shared globals (also extern'd by runtime_utils.c and queue.c) ─────── */

const OrtApi *api = NULL;
Logger *logger = NULL;

/* ── Internal types ─────────────────────────────────────────────────────── */

typedef struct {
    int model_id;
    Tensors *tensors;
} OutputItem;

typedef struct ModelState ModelState;

typedef struct {
    ModelState *model;
    OrtSession *session;
    int replica_id;
} WorkerArg;

struct ModelState {
    int model_id;
    int active;
    int n_replicas;
    OrtSession **sessions;       /* array of n_replicas */
    WorkerArg *worker_args;      /* array of n_replicas */
    pthread_t *threads;          /* array of n_replicas */
    OrtRunOptions *run_options;
    OrtAllocator *allocator;
    OrtMemoryInfo *memory_info;
    OrtEnv *env;
    OrtSessionOptions *session_options;
    Queue *input_queue;
    atomic_int stop;
    char **input_names;
    int num_inputs;
    char **output_names;
    int num_outputs;
};

/* ── Module state ───────────────────────────────────────────────────────── */

static int g_initialized = 0;
static int g_models_loaded = 0;
static char g_last_error[1024] = {0};
static char g_info_json[512] = {0};
static LogLevel g_log_level = LOG_INFO;
static char g_log_file[256] = "runtime.log";
static int g_n_threads = 4;
static int g_n_replicas = 1;

static ModelState g_models[MAX_MODELS];
static int g_num_models = 0;
static Queue *g_output_queue = NULL;

/* ── Helpers ────────────────────────────────────────────────────────────── */

static void set_error(const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    vsnprintf(g_last_error, sizeof(g_last_error), fmt, args);
    va_end(args);
    if (logger) log_error(logger, "%s", g_last_error);
}

static int config_get_int(const Config *cfg, const char *key, int fallback) {
    for (int i = 0; i < cfg->length; i++) {
        if (cfg->keys[i] && strcmp(cfg->keys[i], key) == 0 && cfg->values[i])
            return atoi(cfg->values[i]);
    }
    return fallback;
}

static const char *config_get_str(const Config *cfg, const char *key, const char *fallback) {
    for (int i = 0; i < cfg->length; i++) {
        if (cfg->keys[i] && strcmp(cfg->keys[i], key) == 0 && cfg->values[i])
            return cfg->values[i];
    }
    return fallback;
}

/* ── Output builder ─────────────────────────────────────────────────────── */

static Tensors *build_output(ModelState *m, OrtValue **output_values, int request_id) {
    Tensors *out = (Tensors *)malloc(sizeof(Tensors));
    if (!out) return NULL;

    out->id = request_id;
    out->num_tensors = m->num_outputs;
    out->tensors = (TensorDescriptor *)calloc(m->num_outputs, sizeof(TensorDescriptor));
    if (!out->tensors) {
        free(out);
        return NULL;
    }

    for (int i = 0; i < m->num_outputs; i++) {
        OrtTensorTypeAndShapeInfo *shape_info = NULL;
        if (process_ort_status(api->GetTensorTypeAndShape(output_values[i], &shape_info)) != 0) {
            free_tensors(out);
            return NULL;
        }

        ONNXTensorElementDataType ort_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
        api->GetTensorElementType(shape_info, &ort_type);

        size_t rank = 0;
        api->GetDimensionsCount(shape_info, &rank);

        int64_t *dims = (int64_t *)malloc(rank * sizeof(int64_t));
        if (!dims) {
            api->ReleaseTensorTypeAndShapeInfo(shape_info);
            free_tensors(out);
            return NULL;
        }
        api->GetDimensions(shape_info, dims, rank);

        size_t elem_count = 0;
        api->GetTensorShapeElementCount(shape_info, &elem_count);
        api->ReleaseTensorTypeAndShapeInfo(shape_info);

        TensorElementType elem_type = ort_type_to_tensor_element_type(ort_type);
        size_t elem_size = get_element_byte_size(elem_type);
        size_t total_bytes = elem_count * elem_size;

        out->tensors[i].name = strdup(m->output_names[i]);
        out->tensors[i].data_type = elem_type;
        out->tensors[i].rank = (int)rank;
        out->tensors[i].shape = (int *)malloc(rank * sizeof(int));
        out->tensors[i].data_size = total_bytes;
        out->tensors[i].data = malloc(total_bytes);

        if (!out->tensors[i].name || !out->tensors[i].shape || (total_bytes > 0 && !out->tensors[i].data)) {
            free(dims);
            free_tensors(out);
            return NULL;
        }

        for (size_t j = 0; j < rank; j++)
            out->tensors[i].shape[j] = (int)dims[j];
        free(dims);

        if (total_bytes > 0) {
            void *raw = NULL;
            api->GetTensorMutableData(output_values[i], &raw);
            memcpy(out->tensors[i].data, raw, total_bytes);
        }
    }

    return out;
}

/* ── Inference execution ────────────────────────────────────────────────── */

static Tensors *run_inference(ModelState *m, OrtSession *session, const Tensors *input) {
    int n_in = input->num_tensors;
    OrtValue **input_values = (OrtValue **)calloc(n_in, sizeof(OrtValue *));
    OrtValue **output_values = (OrtValue **)calloc(m->num_outputs, sizeof(OrtValue *));
    Tensors *result = NULL;
    int64_t *shape_buf = NULL;

    if (!input_values || !output_values) {
        set_error("[model %d] OOM in run_inference", m->model_id);
        goto cleanup;
    }

    for (int i = 0; i < n_in; i++) {
        TensorDescriptor *td = &input->tensors[i];
        ONNXTensorElementDataType ort_type = tensor_element_type_to_ort_type(td->data_type);

        shape_buf = (int64_t *)malloc(td->rank * sizeof(int64_t));
        if (!shape_buf) {
            set_error("[model %d] OOM allocating shape buffer", m->model_id);
            goto cleanup;
        }
        for (int j = 0; j < td->rank; j++)
            shape_buf[j] = (int64_t)td->shape[j];

        if (process_ort_status(api->CreateTensorWithDataAsOrtValue(
                m->memory_info, td->data, td->data_size,
                shape_buf, (size_t)td->rank, ort_type,
                &input_values[i])) != 0) {
            free(shape_buf);
            shape_buf = NULL;
            goto cleanup;
        }
        free(shape_buf);
        shape_buf = NULL;
    }

    if (process_ort_status(api->Run(
            session, m->run_options,
            (const char *const *)m->input_names,
            (const OrtValue *const *)input_values, n_in,
            (const char *const *)m->output_names, m->num_outputs,
            output_values)) != 0) {
        set_error("[model %d] Inference run failed", m->model_id);
        goto cleanup;
    }

    result = build_output(m, output_values, input->id);
    if (!result)
        set_error("[model %d] build_output failed (OOM)", m->model_id);

cleanup:
    if (input_values) {
        for (int i = 0; i < n_in; i++)
            if (input_values[i]) api->ReleaseValue(input_values[i]);
        free(input_values);
    }
    if (output_values) {
        for (int i = 0; i < m->num_outputs; i++)
            if (output_values[i]) api->ReleaseValue(output_values[i]);
        free(output_values);
    }
    free(shape_buf);
    return result;
}

/* ── Worker thread ──────────────────────────────────────────────────────── */

static void *worker_loop(void *arg) {
    WorkerArg *wa = (WorkerArg *)arg;
    ModelState *m = wa->model;
    OrtSession *session = wa->session;

    log_info(logger, "[model %d] Worker thread %d started", m->model_id, wa->replica_id);

    while (1) {
        Tensors *input = (Tensors *)dequeue(m->input_queue, WORKER_POLL_MS);

        if (atomic_load(&m->stop)) {
            if (input) free_tensors(input);
            break;
        }

        if (input == NULL) continue;

        log_debug(logger, "[model %d] Replica %d running inference (request id=%d)",
                  m->model_id, wa->replica_id, input->id);
        Tensors *output = run_inference(m, session, input);
        free_tensors(input);

        if (!output) {
            log_warning(logger, "[model %d] Replica %d inference failed, dropping result",
                        m->model_id, wa->replica_id);
            continue;
        }

        OutputItem *item = (OutputItem *)malloc(sizeof(OutputItem));
        if (!item) {
            free_tensors(output);
            log_warning(logger, "[model %d] OOM for OutputItem, dropping result", m->model_id);
            continue;
        }
        item->model_id = m->model_id;
        item->tensors = output;

        if (enqueue(g_output_queue, item) != 0) {
            log_warning(logger, "[model %d] Output queue full, dropping result", m->model_id);
            free_tensors(output);
            free(item);
        }
    }

    log_info(logger, "[model %d] Worker thread %d stopped", m->model_id, wa->replica_id);
    return NULL;
}

/* ── Per-model load/unload ──────────────────────────────────────────────── */

static int load_one_model(int idx, const ModelConfig *mc, int threads_per_replica, int n_replicas) {
    ModelState *m = &g_models[idx];
    memset(m, 0, sizeof(ModelState));
    m->model_id = idx;
    m->n_replicas = n_replicas;
    atomic_store(&m->stop, 0);

    m->sessions    = (OrtSession **)calloc(n_replicas, sizeof(OrtSession *));
    m->worker_args = (WorkerArg *)calloc(n_replicas, sizeof(WorkerArg));
    m->threads     = (pthread_t *)calloc(n_replicas, sizeof(pthread_t));
    if (!m->sessions || !m->worker_args || !m->threads) {
        set_error("[model %d] OOM allocating replica arrays", idx);
        return 1;
    }

    if (process_ort_status(api->CreateEnv(ORT_LOGGING_LEVEL_FATAL, RUNTIME_NAME, &m->env)) != 0) {
        set_error("[model %d] Failed to create ORT environment", idx);
        return 1;
    }

    if (process_ort_status(api->CreateSessionOptions(&m->session_options)) != 0) {
        set_error("[model %d] Failed to create session options", idx);
        return 1;
    }

    api->SetSessionGraphOptimizationLevel(m->session_options, ORT_ENABLE_ALL);
    api->SetIntraOpNumThreads(m->session_options, threads_per_replica);
    api->SetInterOpNumThreads(m->session_options, 1);
    api->SetSessionExecutionMode(m->session_options, ORT_SEQUENTIAL);
    if (process_ort_status(api->SessionOptionsAppendExecutionProvider_Hailo(m->session_options, true)) != 0) {
        set_error("[model %d] Failed to append Hailo execution provider — check that libonnxruntime_providers_hailo.so is alongside libRuntimeLibrary.so", idx);
        return 1;
    }

    /* Log available providers (once, for the first model) */
    if (idx == 0) {
        char **providers = NULL;
        int n_providers = 0;
        api->GetAvailableProviders(&providers, &n_providers);
        for (int j = 0; j < n_providers; j++)
            log_info(logger, "Available provider: %s", providers[j]);
        api->ReleaseAvailableProviders(providers, n_providers);
    }

    /* Create one session per replica */
    for (int r = 0; r < n_replicas; r++) {
        int session_err = 0;
        if (mc->file_path) {
            if (r == 0)
                log_info(logger, "[model %d] Loading from: %s (%d replica(s), %d thread(s) each)",
                         idx, mc->file_path, n_replicas, threads_per_replica);
            session_err = process_ort_status(
                api->CreateSession(m->env, mc->file_path, m->session_options, &m->sessions[r]));
        } else if (mc->model_data && mc->model_size > 0) {
            if (r == 0)
                log_info(logger, "[model %d] Loading from memory (%zu bytes, %d replica(s), %d thread(s) each)",
                         idx, mc->model_size, n_replicas, threads_per_replica);
            session_err = process_ort_status(
                api->CreateSessionFromArray(m->env, mc->model_data, mc->model_size,
                                            m->session_options, &m->sessions[r]));
        } else {
            set_error("[model %d] file_path and model_data are both null", idx);
            return 1;
        }
        if (session_err != 0) {
            set_error("[model %d] Failed to create ORT session for replica %d", idx, r);
            return 1;
        }
    }

    if (process_ort_status(api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &m->memory_info)) != 0) {
        set_error("[model %d] Failed to create memory info", idx);
        return 1;
    }

    /* Allocator and I/O names are derived from replica 0 — all replicas share the same graph */
    if (process_ort_status(api->CreateAllocator(m->sessions[0], m->memory_info, &m->allocator)) != 0) {
        set_error("[model %d] Failed to create allocator", idx);
        return 1;
    }

    if (process_ort_status(api->CreateRunOptions(&m->run_options)) != 0) {
        set_error("[model %d] Failed to create run options", idx);
        return 1;
    }

    m->input_names  = get_input_names(m->sessions[0], m->allocator, &m->num_inputs);
    m->output_names = get_output_names(m->sessions[0], m->allocator, &m->num_outputs);
    if (!m->input_names || !m->output_names) {
        set_error("[model %d] Failed to get I/O names", idx);
        return 1;
    }

    m->input_queue = new_queue(QUEUE_CAPACITY, true);
    if (!m->input_queue) {
        set_error("[model %d] Failed to create input queue", idx);
        return 1;
    }

    /* Spawn one worker thread per replica, all draining the shared input queue */
    for (int r = 0; r < n_replicas; r++) {
        m->worker_args[r].model      = m;
        m->worker_args[r].session    = m->sessions[r];
        m->worker_args[r].replica_id = r;
        if (pthread_create(&m->threads[r], NULL, worker_loop, &m->worker_args[r]) != 0) {
            set_error("[model %d] Failed to create worker thread for replica %d", idx, r);
            return 1;
        }
    }

    m->active = 1;
    log_info(logger, "[model %d] Ready (%d inputs, %d outputs, %d replica(s), %d thread(s) each)",
             idx, m->num_inputs, m->num_outputs, n_replicas, threads_per_replica);
    return 0;
}

static void unload_model(ModelState *m) {
    if (m->active) {
        atomic_store(&m->stop, 1);
        if (m->input_queue) shutdown_queue(m->input_queue);
        for (int r = 0; r < m->n_replicas; r++)
            pthread_join(m->threads[r], NULL);
    }

    if (m->input_queue) {
        Tensors *t;
        while ((t = (Tensors *)dequeue(m->input_queue, 0)) != NULL)
            free_tensors(t);
        free_queue(m->input_queue);
        m->input_queue = NULL;
    }

    free_string_array(m->input_names, m->num_inputs);
    free_string_array(m->output_names, m->num_outputs);

    if (m->run_options)     api->ReleaseRunOptions(m->run_options);
    if (m->allocator)       api->ReleaseAllocator(m->allocator);
    if (m->memory_info)     api->ReleaseMemoryInfo(m->memory_info);
    if (m->sessions) {
        for (int r = 0; r < m->n_replicas; r++)
            if (m->sessions[r]) api->ReleaseSession(m->sessions[r]);
        free(m->sessions);
    }
    if (m->session_options) api->ReleaseSessionOptions(m->session_options);
    if (m->env)             api->ReleaseEnv(m->env);
    free(m->worker_args);
    free(m->threads);

    memset(m, 0, sizeof(ModelState));
}

/* ── Public API ─────────────────────────────────────────────────────────── */

RuntimeStatus runtime_init(Config config) {
    if (g_initialized) {
        snprintf(g_last_error, sizeof(g_last_error), "Already initialized — call runtime_cleanup() first");
        return RUNTIME_STATUS_ALREADY_INITIALIZED;
    }

    int log_level = config_get_int(&config, "log_level", (int)LOG_INFO);
    if (log_level < 0 || log_level > 3) log_level = (int)LOG_INFO;
    g_log_level = (LogLevel)log_level;

    const char *log_file = config_get_str(&config, "log_file", "runtime.log");
    strncpy(g_log_file, log_file, sizeof(g_log_file) - 1);

    /* n_threads takes precedence over perf_mode */
    int n_threads = config_get_int(&config, "n_threads", -1);
    if (n_threads > 0) {
        if (n_threads > 16) n_threads = 16;
        g_n_threads = n_threads;
    } else {
        const char *perf_mode = config_get_str(&config, "perf_mode", NULL);
        if (perf_mode) {
            int ncores = (int)sysconf(_SC_NPROCESSORS_ONLN);
            if (strcmp(perf_mode, "eco") == 0)
                g_n_threads = (int)(ncores * 0.4);
            else if (strcmp(perf_mode, "power") == 0)
                g_n_threads = (int)(ncores * 0.9);
            if (g_n_threads < 1) g_n_threads = 1;
        }
        /* else: keep default of 4 */
    }

    int n_replicas = config_get_int(&config, "n_replicas", 1);
    if (n_replicas < 1) n_replicas = 1;
    g_n_replicas = n_replicas;

    logger = create_logger(RUNTIME_NAME, g_log_file, g_log_level, LOG_INFO);
    if (!logger) {
        snprintf(g_last_error, sizeof(g_last_error), "Failed to create logger");
        return RUNTIME_STATUS_ERROR;
    }

    log_info(logger, "Initializing %s v%s", RUNTIME_NAME, RUNTIME_VERSION);

    api = OrtGetApiBase()->GetApi(ONNXRUNTIME_API_VERSION);
    if (!api) {
        set_error("Failed to get ORT API (version %d)", ONNXRUNTIME_API_VERSION);
        return RUNTIME_STATUS_ERROR;
    }

    g_initialized = 1;
    log_info(logger, "Initialization complete");
    return RUNTIME_STATUS_SUCCESS;
}

RuntimeStatus runtime_load_models(int num_models, const ModelConfig *model_configs) {
    if (!g_initialized) return RUNTIME_STATUS_NOT_INITIALIZED;
    if (g_models_loaded) {
        set_error("Models already loaded — call runtime_cleanup() first");
        return RUNTIME_STATUS_ALREADY_INITIALIZED;
    }
    if (num_models <= 0 || num_models > MAX_MODELS || !model_configs) {
        set_error("Invalid arguments: num_models=%d (max %d)", num_models, MAX_MODELS);
        return RUNTIME_STATUS_INVALID_ARGUMENT;
    }

    /* Split thread budget evenly across replicas (each model gets the same allocation) */
    int threads_per_replica = g_n_threads / g_n_replicas;
    if (threads_per_replica < 1) threads_per_replica = 1;

    log_info(logger, "Loading %d model(s): %d replica(s) each, %d thread(s)/replica",
             num_models, g_n_replicas, threads_per_replica);

    g_output_queue = new_queue(QUEUE_CAPACITY, true);
    if (!g_output_queue) {
        set_error("Failed to create output queue");
        return RUNTIME_STATUS_OUT_OF_MEMORY;
    }

    for (int i = 0; i < num_models; i++) {
        if (load_one_model(i, &model_configs[i], threads_per_replica, g_n_replicas) != 0) {
            for (int j = 0; j <= i; j++) unload_model(&g_models[j]);
            free_queue(g_output_queue);
            g_output_queue = NULL;
            return RUNTIME_STATUS_ERROR;
        }
    }

    g_num_models = num_models;
    g_models_loaded = 1;
    log_info(logger, "%d model(s) loaded", g_num_models);
    return RUNTIME_STATUS_SUCCESS;
}

RuntimeStatus runtime_enqueue_input(int model_id, Tensors *input_tensors) {
    if (!g_initialized) return RUNTIME_STATUS_NOT_INITIALIZED;
    if (!g_models_loaded) return RUNTIME_STATUS_MODEL_NOT_LOADED;
    if (model_id < 0 || model_id >= g_num_models) {
        set_error("Invalid model_id: %d (loaded: %d)", model_id, g_num_models);
        return RUNTIME_STATUS_INVALID_MODEL_ID;
    }
    if (!input_tensors) {
        set_error("input_tensors is NULL");
        return RUNTIME_STATUS_INVALID_ARGUMENT;
    }

    if (enqueue(g_models[model_id].input_queue, input_tensors) != 0) {
        set_error("[model %d] Failed to enqueue input", model_id);
        return RUNTIME_STATUS_ERROR;
    }

    log_debug(logger, "[model %d] Input queued (id=%d)", model_id, input_tensors->id);
    return RUNTIME_STATUS_SUCCESS;
}

RuntimeStatus runtime_retrieve_output(int *model_id, Tensors **output_tensors, int timeout_ms) {
    if (!g_initialized) return RUNTIME_STATUS_NOT_INITIALIZED;
    if (!model_id || !output_tensors) {
        set_error("Null output parameter");
        return RUNTIME_STATUS_INVALID_ARGUMENT;
    }

    OutputItem *item = (OutputItem *)dequeue(g_output_queue, (long)timeout_ms);
    if (!item) return RUNTIME_STATUS_NO_OUTPUT_AVAILABLE;

    *model_id = item->model_id;
    *output_tensors = item->tensors;
    log_debug(logger, "[model %d] Output retrieved (id=%d)", item->model_id, item->tensors->id);
    free(item);
    return RUNTIME_STATUS_SUCCESS;
}

RuntimeStatus runtime_cleanup(void) {
    if (!g_initialized) return RUNTIME_STATUS_SUCCESS;  /* idempotent */

    log_info(logger, "Cleaning up runtime");

    for (int i = 0; i < g_num_models; i++)
        unload_model(&g_models[i]);
    g_num_models = 0;

    if (g_output_queue) {
        OutputItem *item;
        while ((item = (OutputItem *)dequeue(g_output_queue, 0)) != NULL) {
            free_tensors(item->tensors);
            free(item);
        }
        free_queue(g_output_queue);
        g_output_queue = NULL;
    }

    g_initialized = 0;
    g_models_loaded = 0;
    g_n_threads = 4;
    g_n_replicas = 1;
    memset(g_last_error, 0, sizeof(g_last_error));

    log_info(logger, "Cleanup complete");
    close_logger(logger);
    logger = NULL;
    api = NULL;

    return RUNTIME_STATUS_SUCCESS;
}

const char *runtime_get_error(void) {
    return g_last_error[0] ? g_last_error : NULL;
}

const char *runtime_get_version(void) {
    return RUNTIME_VERSION;
}

const char *runtime_get_name(void) {
    return RUNTIME_NAME;
}

const char *runtime_get_info(void) {
    if (!g_initialized) return NULL;

    int in_flight = 0;
    for (int i = 0; i < g_num_models; i++) {
        if (g_models[i].active && g_models[i].input_queue)
            in_flight += g_models[i].input_queue->size;
    }

    const char *ort_ver = api ? OrtGetApiBase()->GetVersionString() : "unknown";
    snprintf(g_info_json, sizeof(g_info_json),
             "{\"loaded_models\":%d,\"requests_in_flight\":%d,\"backend_version\":\"%s\"}",
             g_num_models, in_flight, ort_ver ? ort_ver : "unknown");

    return g_info_json;
}
