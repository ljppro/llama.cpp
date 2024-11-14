#include <cassert>
#include <algorithm>

#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"
#include "ggml.h"

#include "ggml-fp8.h"

template<int N>
constexpr float exp_p2() {
    return exp_p2<N-1>()*2;
}
template<int N>
constexpr float exp_m2() {
    return exp_m2<N+1>()/2;
}
template<int N>
constexpr int exp_i2() {
    return 1 << N;
}
template<> constexpr float exp_p2<0>() { return 1;}
template<> constexpr float exp_m2<0>() { return 1;}

template<int E> //, int M=7-E>  1.7 bits!
struct FP8 {
    uint8_t bits;
    using type = FP8<E>;
    // static constexpr int E=_E;
    static constexpr int M()      { return 7-E; }
    static constexpr int E_BIAS() { return exp_i2<E-1>()-1; }
    static constexpr float MAX()  { return (2-exp_m2<-M()+1>())*exp_p2<exp_i2<E-1>()>(); }
    static constexpr float MIN()  { return exp_m2<-M()>()*exp_m2<2-exp_i2<E-1>()>(); }
    //=============================================

#ifdef GGML_USE_OPENMP_SIMD
    #pragma omp declare simd
#endif
    FP8& operator=(float value) {
        union {
            float f;
            uint32_t bits;
        } in = {value};
        // the signe:
        bits = (in.bits >> 24) & 0x80;
        // value without signe!
        in.bits &= 0x7fffffff;
        //GGML_ASSERT(in.bits < 0x7f800000); // +/- infini ou NAN
        if (in.f >= MAX()) {
            bits |= 0x7E;
        } else if (in.f<MIN()) { // => 0.
            // OK: S.0000000
        } else {
            in.f *= exp_m2<E_BIAS()-127>();
            in.bits += 1<<(22-M()); // for rounding
            bits |= (in.bits >> (23-M())) & 0x7F;
        }
        return *this;
    }

#ifdef GGML_USE_OPENMP_SIMD
    #pragma omp declare simd
#endif
    operator float () const {
        union {
            float f;
            uint32_t bits;
        } out = {0};
        out.bits = bits & 0x80;
        out.bits <<= 24;
        uint32_t _bits = bits & 0x7F;
        _bits <<= (23-M());
        out.bits |= _bits;
        out.f *= exp_p2<127-E_BIAS()>();
        return out.f;
    }
};

template<int E>
static inline void conv(const FP8<E>* x, float* y, int64_t size) {
#ifdef GGML_USE_OPENMP_SIMD
    #pragma omp simd
#endif
    for (int64_t i=0; i<size; i++) {
        y[i] = (float) x[i];
    }
}

template<int E>
static inline void conv(const float* x, FP8<E>* y, int64_t size) {
#ifdef GGML_USE_OPENMP_SIMD
    #pragma omp simd
#endif
    for (int64_t i=0; i<size; i++) {
        y[i] = x[i];
    }
}

template<int E>
static inline float dot(const FP8<E>* x, const float* y, int64_t size) {
    float z = 0;
#ifdef GGML_USE_OPENMP_SIMD
    #pragma omp simd reduction(+:z)
#endif
    for (int64_t i=0; i<size; i++) {
        z += ((float)x[i])*y[i];
    }
    return z;
}

template <int E, int QK>
struct bloc_fp8 {
    float d;
    FP8<E> qs[QK];
};

template <int E, int QK>
static inline void conv(const bloc_fp8<E, QK>* x, float* y, int64_t size) {
    const auto qk_size = size / QK;
    for (int64_t q=0; q<qk_size; ++q) {
#ifdef GGML_USE_OPENMP_SIMD
        #pragma omp simd
#endif
        for (int64_t i=0; i<QK; i++) {
            y[q*QK+i] = ((float) x[q].qs[i])*(x[q]).d;
        }
    }
}

template <int E, int QK>
static inline void conv(const float* x, bloc_fp8<E, QK>* y, int64_t size) {
    const auto qk_size = size / QK;
    for (int64_t q=0; q<qk_size; ++q) {
        float m = 0;
#ifdef GGML_USE_OPENMP_SIMD
        // did not work on macOS and warn.
        // #pragma omp simd reduction(max:m)
#endif
        for (int64_t i=0; i<QK; i++) {
            m = std::max(std::abs(x[q*QK+i]),m);
        }
        const float D = FP8<E>::MAX()/m;
        y[q].d = m/FP8<E>::MAX();
#ifdef GGML_USE_OPENMP_SIMD
        #pragma omp simd
#endif
        for (int64_t i=0; i<QK; i++) {
            y[q].qs[i] = x[q*QK+i]*D;
        }
    }
}

template <int E, int QK>
static inline float dot(const bloc_fp8<E, QK>* x, const float* y, int64_t size) {
    float z = 0;
    const auto qk_size = size / QK;
    for (int64_t q=0; q<qk_size; ++q) {
        float z0 = 0;
#ifdef GGML_USE_OPENMP_SIMD
        #pragma omp simd reduction(+:z0)
#endif
        for (int64_t i=0; i<QK; i++) {
            z0 += ((float)x[q].qs[i])*y[q*QK+i];
        }
        z += (x[q]).d * z0;
    }
    return z;
}

// the C API.
void ggml_e5m2_to_fp32_row(const ggml_e5m2_t * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    conv(reinterpret_cast<const FP8<5>*>(x), y, k);
}
void ggml_fp32_to_e5m2_row(const float * GGML_RESTRICT x, ggml_e5m2_t * GGML_RESTRICT y, int64_t k) {
    conv(x, reinterpret_cast<FP8<5>*>(y), k);
}
void ggml_fp32_to_e5m2_row_ref(const float * GGML_RESTRICT x, ggml_e5m2_t * GGML_RESTRICT y, int64_t k) {
    for (int64_t i =0; i<k; ++i) {
        reinterpret_cast<FP8<5>*>(y)[i] = x[i];
    }
}

void ggml_e4m3_to_fp32_row(const ggml_e4m3_t * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    conv(reinterpret_cast<const FP8<4>*>(x), y, k);
}
void ggml_fp32_to_e4m3_row(const float * GGML_RESTRICT x, ggml_e4m3_t * GGML_RESTRICT y, int64_t k) {
    conv(x, reinterpret_cast<FP8<4>*>(y), k);
}
void ggml_fp32_to_e4m3_row_ref(const float * GGML_RESTRICT x, ggml_e4m3_t * GGML_RESTRICT y, int64_t k) {
    for (int64_t i =0; i<k; ++i) {
        reinterpret_cast<FP8<4>*>(y)[i] = x[i];
    }
}

void dequantize_row_e4m3_q(const block_e4m3_q * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    conv(reinterpret_cast<const bloc_fp8<4, QK_K>*>(x), y, k);
}
void quantize_row_e4m3_q(const float * GGML_RESTRICT x, block_e4m3_q * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    conv(x, reinterpret_cast<bloc_fp8<4, QK_K>*>(y), k);
}
void quantize_row_e4m3_q_ref(const float * GGML_RESTRICT x, block_e4m3_q * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    conv(x, reinterpret_cast<bloc_fp8<4, QK_K>*>(y), k);
}

void dequantize_row_e3m4_q(const block_e3m4_q * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    conv(reinterpret_cast<const bloc_fp8<3, QK_K>*>(x), y, k);
}
void quantize_row_e3m4_q(const float * GGML_RESTRICT x, block_e3m4_q * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    conv(x, reinterpret_cast<bloc_fp8<3, QK_K>*>(y), k);
}
void quantize_row_e3m4_q_ref(const float * GGML_RESTRICT x, block_e3m4_q * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    conv(x, reinterpret_cast<bloc_fp8<3, QK_K>*>(y), k);
}

// the dot product for FP8 weight
void ggml_vec_dot_e5m2(int n, float * GGML_RESTRICT s, size_t bs, const ggml_e5m2_t * GGML_RESTRICT vx, size_t bx, const float * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(nrc == 1);
    GGML_UNUSED(nrc);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);
    GGML_UNUSED(bs);
    *s = dot(reinterpret_cast<const FP8<5>*>(vx), vy, n);
}

void ggml_vec_dot_e4m3(int n, float * GGML_RESTRICT s, size_t bs, const ggml_e4m3_t * GGML_RESTRICT vx, size_t bx, const float * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(nrc == 1);
    GGML_UNUSED(nrc);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);
    GGML_UNUSED(bs);
    *s = dot(reinterpret_cast<const FP8<4>*>(vx), vy, n);
}

void ggml_vec_dot_e4m3_q(int n, float * GGML_RESTRICT s, size_t bs, const block_e4m3_q * GGML_RESTRICT vx, size_t bx, const float * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(nrc == 1);
    GGML_UNUSED(nrc);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);
    GGML_UNUSED(bs);
    *s = dot(reinterpret_cast<const bloc_fp8<4, QK_K>*>(vx), vy, n);
}

void ggml_vec_dot_e3m4_q(int n, float * GGML_RESTRICT s, size_t bs, const block_e3m4_q * GGML_RESTRICT vx, size_t bx, const float * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(nrc == 1);
    GGML_UNUSED(nrc);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);
    GGML_UNUSED(bs);
    *s = dot(reinterpret_cast<const bloc_fp8<3, QK_K>*>(vx), vy, n);
}
