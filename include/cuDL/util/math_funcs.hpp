#ifndef UTIL_MATH_FUNCS_H_
#define UTIL_MATH_FUNCS_H_

#include <Accelerate/Accelerate.h>

// CPU version math funcs
void inner_product_cpu(const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB,
    			       const int M, const int N, const int K, const float alpha,
    			       const float *A, const float *B,
    			       const float beta, float *C);


// GPU version math funcs
void inner_product_gpu(const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB,
    			       const int M, const int N, const int K, const float alpha,
    			       const float *A, const float *B,
    			       const float beta, float *C);

#endif