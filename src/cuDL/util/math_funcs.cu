#include <math_functions.h>
#include "cuDL/util/math_funcs.hpp"


void inner_product_gpu(const CBLAS_TRANSPOSE TransA,
                       const CBLAS_TRANSPOSE TransB,
                       const int M, const int N,
                       const int K, const float alpha,
                       const float *A, const float *B,
                       const float beta, float *C) {
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;

  cublasOperation_t cuTransA = (TransA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB = (TransB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasHandle_t handle;
  cublasSgemm(handle, cuTransB, cuTransA, N, M, K, &alpha, B, ldb, A, lda, &beta, C, N);
}

