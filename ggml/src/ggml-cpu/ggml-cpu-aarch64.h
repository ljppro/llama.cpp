#pragma once

#include "ggml.h"
#include "ggml-cpu-traits.h"

// GGML internal header

#ifdef __cplusplus
extern "C" {
#endif

#ifdef GGML_USE_CPU_AARCH64
    const struct ggml_cpu_tensor_traits* ggml_aarch64_get_optimal_repack_type(const struct ggml_tensor * cur);
    GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_cpu_aarch64_buffer_type(void);
    GGML_BACKEND_API bool ggml_backend_cpu_buft_is_aarch64(ggml_backend_buffer_type_t buft);
#endif

#ifdef __cplusplus
}
#endif

