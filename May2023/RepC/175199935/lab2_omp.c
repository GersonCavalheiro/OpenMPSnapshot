#include <malloc.h>
#include <omp.h>
#include <assert.h>
#include <string.h>
#include <stdbool.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#define TOLERANCE 1e-3
#pragma GCC optimize("Ofast")
#pragma GCC target("sse,sse2,sse3,ssse3,sse4,popcnt,abm,mmx,avx,tune=native")
float matrix_similarity(float *M_1, int m, int n, float *M_2)
{
float l2_diff=0.0;
for (int i=0; i<m; i++)
{
for (int j=0; j<n; j++)
{
l2_diff+=(M_1[i*n+j]-M_2[i*n+j])*(M_1[i*n+j]-M_2[i*n+j]);
}
}
l2_diff = sqrtf(l2_diff);
printf("L2-diff b/w D_T's: %f\n", l2_diff);
return l2_diff;
}
void transpose(float *M, int m, int n, float *M_T)
{
int i, j;
#pragma omp parallel for num_threads(1) private(i, j) collapse(2) schedule(static) 
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
float *M_2_T = (float *) malloc(sizeof(float)*n2*m2);
transpose(M_2, m2, n2, M_2_T);
int i, j, k, temp1, temp2;
#pragma omp parallel for private(i, j, k, sum, temp1, temp2) schedule(static)
for (i = 0; i < m1; i++)
{
temp1 = i*n1; 
for (j = 0; j < n2; j++)
{
sum = 0.0;
temp2 = j*m2;
for (k = 0; k < n1; k++)
{
sum += M_1[temp1 + k] * M_2_T[temp2 + k];
}
result[i * n2 + j] = sum;
}
}
free(M_2_T);
}
float* initialize_identity(int size)
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
float l2_norm_diagonal_diff(float *A_next, float *A_current, int P, float *E_next, float *E_current)
{
int i,j;
float norm, sum_sq = 0.0;
for (i = 0; i < P; i++)
{
sum_sq += (A_next[i * P + i] - A_current[i * P + i]) * (A_next[i * P + i] - A_current[i * P + i]);
}
if (sum_sq>TOLERANCE)
return norm = sqrtf(sum_sq);
#pragma omp parallel for private(i, j) reduction(+: sum_sq)
for (i=0; i<P; i++)
{
for (j=0; j<P; j++)
{
sum_sq+=(E_next[i*P+j]-E_current[i*P+j])*(E_next[i*P+j]-E_current[i*P+j]);
}
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
R_current[col * P + col] = l2_norm(v_col, P);
for (row = 0; row < P; row++)
{
Q_current[row * P + col] = v_col[row] / R_current[col * P + col];
}
}
free(v_col);
}
void modifiedGS(float *A_current, int P, float *Q_current, float *R_current)
{
float *V = (float *)malloc(sizeof(float) * P*P);
memcpy(V, A_current, sizeof(float)*P*P);
int i, j, k;
float l2_norm=0.0, inner_product=0.0;
for (i=0; i<P; i++)
{
l2_norm=0.0;
for (j=0; j<P; j++)
{
l2_norm+=V[j*P+i]*V[j*P+i];
}
l2_norm = sqrtf(l2_norm);
R_current[i*P+i] = l2_norm;
for (j=0; j<P; j++)
{
Q_current[j*P+i] = V[j*P+i]/l2_norm;
}
for (j=i+1; j<P; j++)
{
inner_product=0.0;
for (k=0; k<P; k++)
{
inner_product+=Q_current[k*P+i]*V[k*P+j]; 
}
R_current[i*P+j] = inner_product;
for (k=0; k<P; k++)
{
V[k*P+j]-=R_current[i*P+j]*Q_current[k*P+i];
}
}
}
free(V);
}
void compute_V(float **SIGMA, float *D_T, float **U, float **V_T, int N, int P)
{
float *INV_SIGMA = (float *)calloc(N * P, sizeof(float)); 
for (int i = 0; i < P; i++)
{
INV_SIGMA[i * P + i] = 1.0 / ((*SIGMA)[i]);
}
float *U_T = (float *)malloc(sizeof(float) * P * P);
transpose(*U, P, P, U_T);
float *product = (float *)malloc(sizeof(float) * N * P);
multiply(INV_SIGMA, N, P, U_T, P, P, product);
multiply(product, N, P, D_T, P, N, *V_T);
free(INV_SIGMA);
free(U_T);
free(product);
}
void SVD(int N, int P, float *D, float **U, float **SIGMA, float **V_T)
{
float *D_T = (float *)malloc(sizeof(float) * P * N);
transpose(D, N, P, D_T);
int i, j;
#pragma omp parallel for private(i, j) collapse(2) schedule(static) 
for (i = 0; i < N; i++)
{
for (j = 0; j < P; j++)
{
D_T[j * N + i] = D[i * P + j];
}
}
float *A = (float *)calloc(P * P, sizeof(float));   
float *A_T = (float *)calloc(P * P, sizeof(float)); 
multiply(D_T, P, N, D, N, P, A);
float *A_current = (float *)malloc(sizeof(float) * P * P);
memcpy(A_current, A, sizeof(float) * P * P); 
float *E_current = initialize_identity(P);   
float *Q_ = (float *)calloc(P * P, sizeof(float));
float *R_ = (float *)calloc(P * P, sizeof(float));
float diff_norm;
int iter = 0;
do 
{
modifiedGS(A_current, P, Q_, R_);
float *A_next = (float *)calloc(P * P, sizeof(float));
multiply(R_, P, P, Q_, P, P, A_next);
float *E_next = (float *)calloc(P * P, sizeof(float));
multiply(E_current, P, P, Q_, P, P, E_next);
diff_norm = l2_norm_diagonal_diff(A_next, A_current, P, E_next, E_current);
free(A_current);
free(E_current);
A_current = A_next;
E_current = E_next;
iter++;
} 
while(diff_norm > TOLERANCE && iter<6000);
float temp = FLT_MAX;
for (int i = 0; i < P; i++)
{
(*SIGMA)[i] = sqrtf(A_current[i * P + i]);
if ((*SIGMA)[i] > temp)
{
printf("EXCEPTION!\n");
exit(0);
}
temp = (*SIGMA)[i];
}
for (int i = 0; i < P; i++)
{
for (int j = 0; j < P; j++)
{
(*U)[i * P + j] = E_current[i * P + j];
}
}
compute_V(SIGMA, D_T, U, V_T, N, P);
}
void PCA(int retention, int N, int P, float *D, float *U, float *SIGMA, float **D_HAT, int *K)
{
float sum_eigenvalues = 0.0;
int i,j,k, temp1, temp2;
float sum;
for (i = 0; i < P; i++)
{
sum_eigenvalues += SIGMA[i]*SIGMA[i];
}
*K = 0;
float retention_ = 0.0;
i = 0;
while ((retention_ < retention) && (i < P))
{
retention_ += (SIGMA[i]*SIGMA[i] / sum_eigenvalues) * 100;
(*K)++;
i++;
}
*D_HAT = (float *)malloc(sizeof(float) * N * (*K));
#pragma omp parallel for private(i, j, k, sum, temp1, temp2) schedule(static)
for (i=0; i<N; i++)
{
temp1 = i*P;
for (j=0; j<(*K); j++)
{
sum = 0.0;
for (k=0; k<P; k++)
{
sum += D[temp1+k]*U[k*P+j];
}
(*D_HAT)[i*(*K)+j] = sum;
}
}
}
