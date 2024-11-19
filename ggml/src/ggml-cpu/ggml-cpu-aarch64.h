#pragma once

#include "ggml.h"
#include "ggml-cpu-traits.h"

// GGML internal header

#ifdef __cplusplus
extern "C" {
#endif

// TODO: move ggml_backend_cpu_aarch64_buffer* here!
const struct ggml_cpu_tensor_traits* ggml_aarch64_get_optimal_repack_type(const struct ggml_tensor * cur);

#ifdef __cplusplus
}
#endif

