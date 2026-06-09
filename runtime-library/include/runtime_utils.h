#ifndef RUNTIME_UTILS_H
#define RUNTIME_UTILS_H

#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <onnxruntime/core/session/onnxruntime_c_api.h>

#include "oaax_runtime.h"

extern const OrtApi* api;

/* Log and release an OrtStatus. Returns 0 on success, 1 on error. */
int process_ort_status(OrtStatus* status);

/* Get input/output names from session. Caller frees each string and the array. */
char** get_input_names(OrtSession* session, OrtAllocator* allocator, int* count);
char** get_output_names(OrtSession* session, OrtAllocator* allocator, int* count);

/* Free an array of strings returned by get_input/output_names. */
void free_string_array(char** arr, int count);

/* Map between OAAX and ORT tensor element types. */
ONNXTensorElementDataType tensor_element_type_to_ort_type(TensorElementType type);
TensorElementType ort_type_to_tensor_element_type(ONNXTensorElementDataType ort_type);

/* Return byte size per element for the given type. */
size_t get_element_byte_size(TensorElementType type);

/* Free a Tensors struct and all its contents. Safe to call with NULL. */
void free_tensors(Tensors* tensors);

#endif /* RUNTIME_UTILS_H */
