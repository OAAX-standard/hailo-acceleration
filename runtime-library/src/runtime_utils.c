#include "runtime_utils.h"

#include <stdio.h>

#include "logger.h"

extern Logger *logger;

int process_ort_status(OrtStatus *status) {
  if (status == NULL)
    return 0;
  log_error(logger, "ORT error: %s", api->GetErrorMessage(status));
  api->ReleaseStatus(status);
  return 1;
}

char **get_input_names(OrtSession *session, OrtAllocator *allocator,
                       int *count) {
  size_t n = 0;
  process_ort_status(api->SessionGetInputCount(session, &n));
  *count = (int)n;

  char **names = (char **)malloc(n * sizeof(char *));
  char **tmp = (char **)malloc(n * sizeof(char *));
  if (!names || !tmp) {
    free(names);
    free(tmp);
    return NULL;
  }

  for (int i = 0; i < (int)n; i++) {
    process_ort_status(
        api->SessionGetInputName(session, i, allocator, &tmp[i]));
    names[i] = (char *)malloc(strlen(tmp[i]) + 1);
    strcpy(names[i], tmp[i]);
    process_ort_status(api->AllocatorFree(allocator, tmp[i]));
  }
  free(tmp);
  return names;
}

char **get_output_names(OrtSession *session, OrtAllocator *allocator,
                        int *count) {
  size_t n = 0;
  process_ort_status(api->SessionGetOutputCount(session, &n));
  *count = (int)n;

  char **names = (char **)malloc(n * sizeof(char *));
  char **tmp = (char **)malloc(n * sizeof(char *));
  if (!names || !tmp) {
    free(names);
    free(tmp);
    return NULL;
  }

  for (int i = 0; i < (int)n; i++) {
    process_ort_status(
        api->SessionGetOutputName(session, i, allocator, &tmp[i]));
    names[i] = (char *)malloc(strlen(tmp[i]) + 1);
    strcpy(names[i], tmp[i]);
    process_ort_status(api->AllocatorFree(allocator, tmp[i]));
  }
  free(tmp);
  return names;
}

void free_string_array(char **arr, int count) {
  if (!arr)
    return;
  for (int i = 0; i < count; i++)
    free(arr[i]);
  free(arr);
}

ONNXTensorElementDataType
tensor_element_type_to_ort_type(TensorElementType type) {
  switch (type) {
  case DATA_TYPE_FLOAT:
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  case DATA_TYPE_UINT8:
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
  case DATA_TYPE_INT8:
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8;
  case DATA_TYPE_UINT16:
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16;
  case DATA_TYPE_INT16:
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16;
  case DATA_TYPE_INT32:
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
  case DATA_TYPE_INT64:
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
  case DATA_TYPE_BOOL:
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL;
  case DATA_TYPE_FLOAT16:
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
  case DATA_TYPE_DOUBLE:
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE;
  case DATA_TYPE_UINT32:
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32;
  case DATA_TYPE_UINT64:
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64;
  case DATA_TYPE_BFLOAT16:
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16;
  default:
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  }
}

TensorElementType
ort_type_to_tensor_element_type(ONNXTensorElementDataType ort_type) {
  switch (ort_type) {
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
    return DATA_TYPE_FLOAT;
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
    return DATA_TYPE_UINT8;
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
    return DATA_TYPE_INT8;
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
    return DATA_TYPE_UINT16;
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
    return DATA_TYPE_INT16;
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
    return DATA_TYPE_INT32;
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
    return DATA_TYPE_INT64;
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
    return DATA_TYPE_BOOL;
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
    return DATA_TYPE_FLOAT16;
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
    return DATA_TYPE_DOUBLE;
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
    return DATA_TYPE_UINT32;
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
    return DATA_TYPE_UINT64;
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
    return DATA_TYPE_BFLOAT16;
  default:
    return DATA_TYPE_UNDEFINED;
  }
}

size_t get_element_byte_size(TensorElementType type) {
  switch (type) {
  case DATA_TYPE_FLOAT:
    return 4;
  case DATA_TYPE_FLOAT16:
    return 2;
  case DATA_TYPE_BFLOAT16:
    return 2;
  case DATA_TYPE_DOUBLE:
    return 8;
  case DATA_TYPE_INT8:
    return 1;
  case DATA_TYPE_INT16:
    return 2;
  case DATA_TYPE_INT32:
    return 4;
  case DATA_TYPE_INT64:
    return 8;
  case DATA_TYPE_UINT8:
    return 1;
  case DATA_TYPE_UINT16:
    return 2;
  case DATA_TYPE_UINT32:
    return 4;
  case DATA_TYPE_UINT64:
    return 8;
  case DATA_TYPE_BOOL:
    return 1;
  default:
    return 0;
  }
}

void free_tensors(Tensors *tensors) {
  if (!tensors)
    return;
  if (tensors->tensors) {
    for (int i = 0; i < tensors->num_tensors; i++) {
      free(tensors->tensors[i].name);
      free(tensors->tensors[i].shape);
      free(tensors->tensors[i].data);
    }
    free(tensors->tensors);
  }
  free(tensors);
}
