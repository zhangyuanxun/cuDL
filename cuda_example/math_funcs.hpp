#ifndef MATH_FUNCS_H_
#define MATH_FUNCS_H_

// GPU version math funcs
void inner_product_gpu(const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB,
    			       const int M, const int N, const int K, const float alpha,
    			       const float *A, const float *B,
    			       const float beta, float *C);

#endif