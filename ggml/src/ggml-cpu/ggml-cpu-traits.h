#pragma once
#include "ggml.h"

#ifdef  __cplusplus
extern "C" {
#endif

    typedef int (*ggml_repack_t) (struct ggml_tensor *t, int interleave_block, const void * GGML_RESTRICT data,
                                    size_t data_size);
    typedef void (*ggml_from_float_to_mat_t)
                                     (const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t nr, int64_t k, int64_t bs);
    typedef void (*ggml_gemv_t)   (int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT x,
                                     const void * GGML_RESTRICT y, int nr, int nc);
    typedef void (*ggml_gemm_t)   (int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT x,
                                     const void * GGML_RESTRICT y, int nr, int nc);

    struct ggml_cpu_tensor_traits {
        ggml_repack_t            repack;
        int64_t                  blck_size_interleave; // + interleave elements in blocks
        ggml_from_float_to_mat_t from_float_to_mat;    // + mis sur le vec_dot_type ... quantize_mat_q8_0
        enum ggml_type           vec_dot_type;         // +
        int64_t                  nrows;                // ? number of rows to process simultaneously
        int64_t                  ncols;                // ? number of columns to process simultaneously
        ggml_gemv_t              gemv;                 // +
        ggml_gemm_t              gemm;                 // +
    };

#ifdef  __cplusplus
}
#endif
