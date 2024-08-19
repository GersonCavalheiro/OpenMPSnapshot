#include <malloc.h>
#include <omp.h>
#include <assert.h>
#include <string.h>
#include <stdbool.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#define TOLERANCE 1e-3
float matrix_similarity(float *M_1, int m, int n, float *M_2)
{
float l2_diff = 0.0;
for (int i = 0; i < m; i++)
{
for (int j = 0; j < n; j++)
{
l2_diff += (M_1[i * n + j] - M_2[i * n + j]) * (M_1[i * n + j] - M_2[i * n + j]);
}
}
l2_diff = sqrtf(l2_diff);
return l2_diff;
}
void transpose(float *M, int m, int n, float *M_T)
{
int i, j;
for (i = 0; i < m; i++)
{
for (j = 0; j < n; j++)
{
M_T[j * m + i] = M[i * n + j];
}
}
}
void multiply(float *M_1, int m1, int n1, float *M_2, int m2, int n2, float *result)
{
assert(n1 == m2);
float sum = 0.0;
float *M_2_T = (float *)malloc(sizeof(float) * n2 * m2);
transpose(M_2, m2, n2, M_2_T);
int i, j, k;
for (i = 0; i < m1; i++)
{
for (j = 0; j < n2; j++)
{
for (k = 0; k < n1; k++)
{
sum += M_1[i * n1 + k] * M_2_T[j * m2 + k];
}
result[i * n2 + j] = sum;
sum = 0.0;
}
}
free(M_2_T);
}
float *initialize_identity(int size)
{
float *I = (float *)calloc(size * size, sizeof(float));
for (int i = 0; i < size; i++)
{
I[i * size + i] = 1.0;
}
return I;
}
float l2_norm(float *v_col, int length)
{
float norm, sum_sq = 0.0;
for (int i = 0; i < length; i++)
{
sum_sq += v_col[i] * v_col[i];
}
return norm = sqrtf(sum_sq);
}
float l2_norm_diagonal_diff(float *A_next, float *A_current, int P)
{
float norm, sum_sq = 0.0;
for (int i = 0; i < P; i++)
{
sum_sq += (A_next[i * P + i] - A_current[i * P + i]) * (A_next[i * P + i] - A_current[i * P + i]);
}
return norm = sqrtf(sum_sq);
}
void print_matrix(float *A, int M, int N, bool console)
{
for (int i = 0; i < M; i++)
{
for (int j = 0; j < N; j++)
{
if (!console)
fprintf(stderr, "%f ", A[i * N + j]);
else
printf("%f ", A[i * N + j]);
}
if (!console)
fprintf(stderr, "\n");
else
printf("\n");
}
}
void classicalGS(float *A_current, float *A_T, int P, float *Q_current, float *R_current)
{
float *v_col = (float *)malloc(sizeof(float) * P);
int col, row, row_;
float result;
for (col = 0; col < P; col++)
{
memcpy(v_col, A_T + col * P, sizeof(float) * P);
#pragma omp for 
{
for (row = 0; row < col; row++)
{
result = 0.0;
for (row_ = 0; row_ < P; row_++)
{
result += (Q_current[row_ * P + row] * (A_T[col * P + row_]));
}
R_current[row * P + col] = result;
for (row_ = 0; row_ < P; row_++)
{
v_col[row_] -= R_current[row * P + col] * Q_current[row_ * P + row];
}
}
}
R_current[col * P + col] = l2_norm(v_col, P);
for (row = 0; row < P; row++)
{
Q_current[row * P + col] = v_col[row] / R_current[col * P + col];
}
}
free(v_col);
}
void compute_V(float **SIGMA, float *D_T, float **U, float **V_T, int N, int P)
{
float *INV_SIGMA = (float *)calloc(N * P, sizeof(float)); 
for (int i = 0; i < P; i++)
{
INV_SIGMA[i * P + i] = 1.0 / ((*SIGMA)[i]);
}
printf("\n inv-sigma:\n");
print_matrix(INV_SIGMA, N, P, 0);
float *U_T = (float *)malloc(sizeof(float) * P * P);
transpose(*U, P, P, U_T);
float *product = (float *)malloc(sizeof(float) * N * P);
multiply(INV_SIGMA, N, P, U_T, P, P, product);
multiply(product, N, P, D_T, P, N, *V_T);
printf("\n compute_V:\n");
print_matrix(*V_T, N, N, 0);
free(INV_SIGMA);
free(U_T);
free(product);
}
inline bool convergence(float diff_norm)
{
if (diff_norm < TOLERANCE)
return true;
return false;
}
void SVD(int N, int P, float *D, float **U, float **SIGMA, float **V_T)
{
omp_set_nested(1); 
int nprocs = omp_get_num_procs();
int nthreads_1 = nprocs / 2;
int nthreads_2 = nprocs - nthreads_1;
float *D_T = (float *)malloc(sizeof(float) * P * N);
transpose(D, N, P, D_T);
float *A = (float *)calloc(P * P, sizeof(float));   
float *A_T = (float *)calloc(P * P, sizeof(float)); 
multiply(D_T, P, N, D, N, P, A);
float *A_current = (float *)malloc(sizeof(float) * P * P);
memcpy(A_current, A, sizeof(float) * P * P); 
float *E_current = initialize_identity(P);   
float *Q_ = (float *)malloc(sizeof(float) * P * P);
float *R_ = (float *)malloc(sizeof(float) * P * P);
float diff_norm;
printf("\n");
int iter = 1;
float *v_col = (float *)malloc(sizeof(float) * P);      
float sum_sq = 0.0;                                     
float *A_next = (float *)malloc(sizeof(float) * P * P); 
float *E_next = (float *)malloc(sizeof(float) * P * P); 
#pragma omp parallel default(none) shared(nthreads_1, nthreads_2, iter, A_current, A_T, P, v_col, Q_, R_, A_next, E_next)
{
int col, row, row_; 
float result;       
do                  
{
printf("iter:%d\n", iter);
int i, j;
#pragma omp for collapse(2) private(i, j)
for (i = 0; i < P; i++)
{
for (j = 0; j < P; j++)
{
A_T[j * P + i] = A[i * P + j];
}
}
float result;
for (col = 0; col < P; col++)
{
#pragma omp single
memcpy(v_col, A_T + col * P, sizeof(float) * P); 
#pragma omp for
for (row = 0; row < col; row++)
{
result = 0.0;
for (row_ = 0; row_ < P; row_++)
{
result += (Q_[row_ * P + row] * (A_T[col * P + row_]));
}
R_[row * P + col] = result;
for (row_ = 0; row_ < P; row_++)
{
v_col[row_] -= R_[row * P + col] * Q_[row_ * P + row];
}
}
#pragma omp for reduction(+: sum_sq)
for (i = 0; i < P; i++)
{
sum_sq += v_col[i] * v_col[i];
}
#pragma omp single
R_[col * P + col] = sqrtf(sum_sq);
#pragma omp for
for (row = 0; row < P; row++)
{
Q_[row * P + col] = v_col[row] / R_[col * P + col];
}
}
#pragma omp single nowait
{
A_next = (float *)malloc(sizeof(float) * P * P); 
E_next = (float *)malloc(sizeof(float) * P * P); 
#pragma omp task
{
float sum = 0.0; 
int k; 
float *Q_T = (float *)malloc(sizeof(float) * P * P); 
#pragma omp parallel num_threads(nthreads1) default(none) shared(Q_T, Q_,P, A_next) private(i, j, k, sum)
{
#pragma omp for collapse(2)
for (i = 0; i < P; i++)
{
for (j = 0; j < P; j++)
{
Q_T[j * P + i] = Q_[i * P + j];
}
}
#pragma omp for  
for (i = 0; i < P; i++)
{
for (j = 0; j < P; j++)
{
for (k = 0; k < P; k++)
{
sum += R_[i * P + k] * Q_T[j * P + k];
}
A_next[i * P + j] = sum;
sum = 0.0;
}
}
}
free(Q_T);
}
#pragma omp task
{            
float sum = 0.0; 
int k; 
float *Q_T = (float *)malloc(sizeof(float) * P * P); 
#pragma omp parallel num_threads(nthreads2) default(none) shared(Q_T, Q_,P, E_next) private(i, j, k, sum)
{
#pragma omp for collapse(2)
for (i = 0; i < P; i++)
{
for (j = 0; j < P; j++)
{
Q_T[j * P + i] = Q_[i * P + j];
}
}
#pragma omp for  
for (i = 0; i < P; i++)
{
for (j = 0; j < P; j++)
{
for (k = 0; k < P; k++)
{
sum += E_current[i * P + k] * Q_T[j * P + k];
}
E_next[i * P + j] = sum;
sum = 0.0;
}
}
}
free(Q_T);                   
}
}
#pragma omp taskwait
diff_norm = l2_norm_diagonal_diff(A_next, A_current, P);
free(A_current);
free(E_current);
A_current = A_next;
E_current = E_next;
printf("diff_norm: %f, tol:%f\n", diff_norm, TOLERANCE);
}
while (diff_norm > TOLERANCE);
}
free(v_col);
float temp = FLT_MAX;
printf("\nPrinting singular-values: ");
for (int i = 0; i < P; i++)
{
(*SIGMA)[i] = sqrtf(A_current[i * P + i]);
if ((*SIGMA)[i] > temp)
{
printf("EXCEPTION!\n");
exit(0);
}
temp = (*SIGMA)[i];
printf("%f ", (*SIGMA)[i]);
}
printf("\n");
printf("\nE: ");
print_matrix(E_current, P, P, 0);
for (int i = 0; i < P; i++)
{
for (int j = 0; j < P; j++)
{
(*U)[i * P + j] = E_current[i * P + j];
}
}
printf("\n U:\n");
print_matrix(*U, P, P, 0);
float *temp_sigma = (float *)calloc(P * N, sizeof(float));
for (int i = 0; i < P; i++)
{
temp_sigma[i * N + i] = (*SIGMA)[i];
}
printf("\n SIGMA:\n");
print_matrix(temp_sigma, P, N, 0);
compute_V(SIGMA, D_T, U, V_T, N, P);
printf("\n V_T:\n");
print_matrix(*V_T, N, N, 0);
float *product_one = (float *)malloc(sizeof(float) * P * N);
multiply(*U, P, P, temp_sigma, P, N, product_one); 
float *product_two = (float *)malloc(sizeof(float) * P * N);
multiply(product_one, P, N, *V_T, N, N, product_two); 
free(product_one);
printf("\nORIGINAL D_T:\n");
print_matrix(D_T, P, N, 0);
printf("\nORIGINAL D:\n");
print_matrix(D, N, P, 0);
printf("\nVERIFIED D_T:\n");
print_matrix(product_two, P, N, 0);
printf("\n A0 = D_TXD: \n");
print_matrix(A, P, P, 0);
matrix_similarity(D_T, P, N, product_two);
free(temp_sigma);
free(product_two);
}
void PCA(int retention, int N, int P, float *D, float *U, float *SIGMA, float **D_HAT, int *K)
{
float sum_eigenvalues = 0.0;
int i;
for (i = 0; i < P; i++)
{
sum_eigenvalues += SIGMA[i];
}
*K = 0;
float retention_ = 0.0;
i = 0;
while ((retention_ < retention) && (i < P))
{
printf("adding to retention: %f\n", SIGMA[i] / sum_eigenvalues);
retention_ += SIGMA[i] / sum_eigenvalues;
(*K)++;
i++;
}
fprintf(stderr, "K: %d, retention_: %f\n", *K, retention_);
*D_HAT = (float *)malloc(sizeof(float) * N * (*K));
multiply(D, N, P, U, P, *K, *D_HAT);
printf("PRINTING D_HAT:\n");
print_matrix(*D_HAT, N, *K, 0);
}
