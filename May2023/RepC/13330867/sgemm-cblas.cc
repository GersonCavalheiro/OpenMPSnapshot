#include "prk_util.h"
#if defined(MKL)
#include <mkl.h>
#ifdef MKL_ILP64
#error Use the MKL library for 32-bit integers!
#endif
#elif defined(ACCELERATE)
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif
#ifdef _OPENMP
#include <omp.h>
#endif
#ifdef PRK_DEBUG
#include <random>
void prk_sgemm_loops(const int order,
const std::vector<float> & A,
const std::vector<float> & B,
std::vector<float> & C)
{
for (int i=0; i<order; ++i) {
for (int j=0; j<order; ++j) {
for (int k=0; k<order; ++k) {
C[i*order+j] += A[i*order+k] * B[k*order+j];
}
}
}
}
#endif
void prk_sgemm(const int order,
const std::vector<float> & A,
const std::vector<float> & B,
std::vector<float> & C)
{
const int n = order;
const float alpha = 1.0;
const float beta  = 1.0;
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
n, n, n, alpha, A.data(), n, B.data(), n, beta, C.data(), n);
}
void prk_sgemm(const int order, const int batches,
const std::vector<std::vector<float>> & A,
const std::vector<std::vector<float>> & B,
std::vector<std::vector<float>> & C)
{
const int n = order;
const float alpha = 1.0;
const float beta  = 1.0;
for (int b=0; b<batches; ++b) {
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
n, n, n, alpha, &(A[b][0]), n, &(B[b][0]), n, beta, &(C[b][0]), n);
}
}
void prk_sgemm(const int order, const int batches, const int nt,
const std::vector<std::vector<float>> & A,
const std::vector<std::vector<float>> & B,
std::vector<std::vector<float>> & C)
{
const int n = order;
const float alpha = 1.0;
const float beta  = 1.0;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic) num_threads(nt)
#endif
for (int b=0; b<batches; ++b) {
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
n, n, n, alpha, A[b].data(), n, B[b].data(), n, beta, C[b].data(), n);
}
}
void prk_sgemm(const int order, const int batches,
float** & A,
float** & B,
float** & C)
{
const int n = order;
const float alpha = 1.0;
const float beta  = 1.0;
const int group_count = 1;
PRK_UNUSED const int group_size[group_count] = { batches };
const CBLAS_TRANSPOSE transa_array[group_count] = { CblasNoTrans };
const CBLAS_TRANSPOSE transb_array[group_count] = { CblasNoTrans };
const int n_array[group_count] = { n };
const float alpha_array[group_count] = { alpha };
const float beta_array[group_count]  = { beta };
#ifdef MKL
cblas_sgemm_batch(CblasRowMajor, transa_array, transb_array,
n_array, n_array, n_array,
alpha_array,
(const float**) A, n_array,
(const float**) B, n_array,
beta_array,
C, n_array,
group_count, group_size);
#else 
for (int b=0; b<batches; ++b) {
cblas_sgemm(CblasRowMajor,
transa_array[0], transb_array[0],
n_array[0], n_array[0], n_array[0],
alpha_array[0],
A[b], n_array[0],
B[b], n_array[0],
beta_array[0],
C[b], n_array[0]);
}
#endif
}
int main(int argc, char * argv[])
{
std::cout << "Parallel Research Kernels version " << PRKVERSION << std::endl;
std::cout << "C++11/CBLAS Dense matrix-matrix multiplication: C += A x B" << std::endl;
int iterations;
int order;
int batches = 0;
int batch_threads = 1;
try {
if (argc < 3) {
throw "Usage: <# iterations> <matrix order> [<batches> <batch threads>]";
}
iterations  = std::atoi(argv[1]);
if (iterations < 1) {
throw "ERROR: iterations must be >= 1";
}
order = std::atoi(argv[2]);
if (order <= 0) {
throw "ERROR: Matrix Order must be greater than 0";
} else if (order > prk::get_max_matrix_size()) {
throw "ERROR: matrix dimension too large - overflow risk";
}
if (argc > 3) {
batches = std::atoi(argv[3]);
}
if (argc>4) {
batch_threads = std::atoi(argv[4]);
} else {
#ifdef _OPENMP
batch_threads = omp_get_max_threads();
#endif
}
}
catch (const char * e) {
std::cout << e << std::endl;
return 1;
}
std::cout << "Number of iterations = " << iterations << std::endl;
std::cout << "Matrix order         = " << order << std::endl;
if (batches == 0) {
std::cout << "No batching" << std::endl;
} else if (batches > 0) {
#ifdef MKL
std::cout << "Batch size           = " <<  batches << " (batched BLAS)" << std::endl;
#else
std::cout << "Batch size           = " << std::abs(batches) << " (loop over legacy BLAS sequentially)" << std::endl;
#endif
} else if (batches < 0) {
if (batch_threads > 1) {
std::cout << "Batch size           = " << std::abs(batches) << " (loop over legacy BLAS with " << batch_threads << " threads)" << std::endl;
} else {
std::cout << "Batch size           = " << std::abs(batches) << " (loop over legacy BLAS sequentially)" << std::endl;
}
}
double gemm_time(0);
const int matrices = (batches==0 ? 1 : abs(batches));
std::vector<float> const M(order*order,0);
std::vector<std::vector<float>> A(matrices,M);
std::vector<std::vector<float>> B(matrices,M);
std::vector<std::vector<float>> C(matrices,M);
for (int b=0; b<matrices; ++b) {
for (int i=0; i<order; ++i) {
for (int j=0; j<order; ++j) {
A[b][i*order+j] = i;
B[b][i*order+j] = i;
C[b][i*order+j] = 0;
}
}
}
float ** pA = new float*[matrices];
float ** pB = new float*[matrices];
float ** pC = new float*[matrices];
for (int b=0; b<matrices; ++b) {
pA[b] = A[b].data();
pB[b] = B[b].data();
pC[b] = C[b].data();
}
{
for (int iter = 0; iter<=iterations; iter++) {
if (iter==1) gemm_time = prk::wtime();
if (batches == 0) {
prk_sgemm(order, A[0], B[0], C[0]);
} else if (batches < 0) {
prk_sgemm(order, matrices, batch_threads, A, B, C);
} else if (batches > 0) {
prk_sgemm(order, matrices, pA, pB, pC);
}
}
gemm_time = prk::wtime() - gemm_time;
}
const double epsilon = 1.0e-8;
const double forder = static_cast<float>(order);
const double reference = 0.25 * prk::pow(forder,3) * prk::pow(forder-1.0,2) * (iterations+1);
double residuum(0);
for (int b=0; b<matrices; ++b) {
const auto checksum = prk::reduce(C[b].begin(), C[b].end(), 0.0);
residuum += std::abs(checksum - reference) / reference;
}
residuum /= matrices;
if (residuum < epsilon) {
#if VERBOSE
std::cout << "Reference checksum = " << reference << "\n"
<< "Actual checksum = " << checksum << std::endl;
#endif
std::cout << "Solution validates" << std::endl;
auto avgtime = gemm_time/iterations/matrices;
auto nflops = 2.0 * prk::pow(forder,3);
std::cout << "Rate (MF/s): " << 1.0e-6 * nflops/avgtime
<< " Avg time (s): " << avgtime << std::endl;
} else {
std::cout << "Reference checksum = " << reference << "\n"
<< "Residuum           = " << residuum << std::endl;
#if VERBOSE
std::cout << "i, j, A, B, C, D" << std::endl;
for (int i=0; i<order; ++i)
for (int j=0; j<order; ++j)
std::cout << i << "," << j << " = " << A[i*order+j] << ", " << B[i*order+j] << ", " << C[i*order+j] << ", " << D[i*order+j] << "\n";
std::cout << std::endl;
#endif
return 1;
}
return 0;
}
