#include "sparse_gemm.h"

int gemm_sparse_openmp(const float *a, const sparse_weights *w, float *c, size_t m)
{
    if (!a || !w || !c)
        return SPARSE_ERROR;
    const size_t k = w->k, n = w->n, groups = w->n / 4;
    const uint32_t *col_ptr = w->block_col_ptr;
    const uint32_t *row_idx = w->block_row_idx;
    const float *vals = w->block_values;

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (size_t i = 0; i < m; ++i) {
        for (size_t g = 0; g < groups; ++g) {
            float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            for (size_t p = col_ptr[g]; p < col_ptr[g + 1]; ++p) {
                size_t k0 = 4 * (size_t)row_idx[p];
                for (size_t r = 0; r < 4; ++r) {
                    float av = a[i * k + k0 + r];
                    for (size_t col = 0; col < 4; ++col)
                        acc[col] += av * vals[16 * p + 4 * r + col];
                }
            }
            for (size_t col = 0; col < 4; ++col)
                c[i * n + 4 * g + col] = acc[col];
        }
    }
    return SPARSE_OK;
}
