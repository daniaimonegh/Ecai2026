#include "sparse_gemm.h"
#include <arm_neon.h>

#define TM 4 /* tiling size*/

/* Used for the left over rows which isnt usefull for our test cases (this is just the code for sparse_neon) */
static void sparse_row_neon(const float *a, const sparse_weights *w,
                            float *c, size_t i)
{
    const float *arow = a + i * w->k;
    for (size_t g = 0; g < w->n / 4; ++g) {
        float32x4_t acc = vdupq_n_f32(0.0f);
        for (size_t p = w->block_col_ptr[g]; p < w->block_col_ptr[g + 1]; ++p) {
            const float *ak = arow + 4 * (size_t)w->block_row_idx[p];
            const float *blk = w->block_values + 16 * p;
            for (size_t r = 0; r < 4; ++r)
                acc = vfmaq_f32(acc, vdupq_n_f32(ak[r]), vld1q_f32(blk + 4 * r));
        }
        vst1q_f32(c + i * w->n + 4 * g, acc);
    }
}


int gemm_sparse_openmp_tiled_neon(const float *a, const sparse_weights *w,
                                  float *c, size_t m)
{
    if (!a || !w || !c)
        return SPARSE_ERROR;
    const size_t k = w->k, n = w->n, groups = w->n / 4;
    const size_t tiles = m / TM;
    const uint32_t *col_ptr = w->block_col_ptr;
    const uint32_t *row_idx = w->block_row_idx;
    const float *vals = w->block_values;

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (size_t tile = 0; tile < tiles; ++tile) {
        const size_t i0 = tile * TM;
        const float *a0 = a + (i0 + 0) * k;
        const float *a1 = a + (i0 + 1) * k;
        const float *a2 = a + (i0 + 2) * k;
        const float *a3 = a + (i0 + 3) * k;
        for (size_t g = 0; g < groups; ++g) {
            float32x4_t acc0 = vdupq_n_f32(0.0f), acc1 = vdupq_n_f32(0.0f);
            float32x4_t acc2 = vdupq_n_f32(0.0f), acc3 = vdupq_n_f32(0.0f);
            for (size_t p = col_ptr[g]; p < col_ptr[g + 1]; ++p) {
                const size_t k0 = 4 * (size_t)row_idx[p];
                const float *blk = vals + 16 * p;
                for (size_t r = 0; r < 4; ++r) {
                    const float32x4_t wv = vld1q_f32(blk + 4 * r); /* loaded once */
                    acc0 = vfmaq_f32(acc0, vdupq_n_f32(a0[k0 + r]), wv);
                    acc1 = vfmaq_f32(acc1, vdupq_n_f32(a1[k0 + r]), wv);
                    acc2 = vfmaq_f32(acc2, vdupq_n_f32(a2[k0 + r]), wv);
                    acc3 = vfmaq_f32(acc3, vdupq_n_f32(a3[k0 + r]), wv);
                }
            }
            vst1q_f32(c + (i0 + 0) * n + 4 * g, acc0);
            vst1q_f32(c + (i0 + 1) * n + 4 * g, acc1);
            vst1q_f32(c + (i0 + 2) * n + 4 * g, acc2);
            vst1q_f32(c + (i0 + 3) * n + 4 * g, acc3);
        }
    }
    for (size_t i = tiles * TM; i < m; ++i)
        sparse_row_neon(a, w, c, i);
    return SPARSE_OK;
}
