#include <sycl/sycl.hpp>
#include "outprod.hpp"


void ggml_sycl_op_out_prod(ggml_backend_sycl_context& ctx, const ggml_tensor* src0,
    const ggml_tensor* src1, ggml_tensor* dst) {
    
    
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(dst));

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne0 = dst->ne[0];
    const int64_t ne1 = dst->ne[1];
    const int64_t ne10 = src1->ne[0];
    const int64_t ne11 = src1->ne[1];

    // Get strides
    const int64_t nb10 = src1->nb[0];
    const int64_t nb11 = src1->nb[1];

    // Get SYCL queue
    dpct::queue_ptr stream = ctx.stream();

    // Dimension checks
    GGML_ASSERT(ne01 == ne11);  // Inner dimensions must match
    GGML_ASSERT(ne0 == ne00);   // Output rows match src0 rows
    GGML_ASSERT(ne1 == ne10);   // Output cols match src1 cols

    // Get data pointers
    const float* src0_d = (const float*)src0->data;
    const float* src1_d = (const float*)src1->data;
    float* dst_d = (float*)dst->data;

    // GEMM parameters
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Handle transposition of src1
    const bool src1_T = ggml_is_transposed(src1);
    const oneapi::mkl::transpose src1_op = 
        src1_T ? oneapi::mkl::transpose::nontrans : oneapi::mkl::transpose::trans;
    const int64_t ldb = (src1_T ? nb10 : nb11) / sizeof(float);

    try {
        // Perform matrix multiplication using oneMKL GEMM
        oneapi::mkl::blas::gemm(*stream,
            oneapi::mkl::transpose::nontrans, src1_op,
            ne0, ne1, ne01,
            alpha,
            src0_d, ne00,
            src1_d, ldb,
            beta,
            dst_d, ne0);
    }
    catch (sycl::exception const& exc) {
        std::cerr << exc.what() << std::endl;
        GGML_ASSERT(false);
    }
}