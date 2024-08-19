void matmul(double  *A, double *B, double *C, unsigned long NB)
{
int i, j, k, I;
double tmp;
for (i = 0; i < NB; i++)
{
I=i*NB;
for (j = 0; j < NB; j++)
{
tmp=C[I+j];
for (k = 0; k < NB; k++)
{
tmp+=A[I+k]*B[k*NB+j];
}
C[I+j]=tmp;
}
}
}
#if 1
void compute_1(unsigned long NB, unsigned long DIM,
double *A[DIM][DIM], double *B[DIM][DIM], double *C[DIM][DIM])
{
unsigned i, j, k;
for (i = 0; i < DIM; i++)
for (j = 0; j < DIM; j++)
for (k = 0; k < DIM; k++)
#pragma analysis_check assert correctness_incoherent_in_pointed(A, B) correctness_incoherent_out_pointed(C) correctness_auto_storage(A, B, C)
#pragma omp task in(A, B) out(C)
matmul ((double *)A[i][k], (double *)B[k][j], (double *)C[i][j], NB);
}
#endif
#if 1
void compute_2(unsigned long NB, unsigned long DIM,
double *A[DIM][DIM], double *B[DIM][DIM], double *C[DIM][DIM])
{
unsigned i, j, k;
for (i = 0; i < DIM; i++)
for (j = 0; j < DIM; j++)
for (k = 0; k < DIM; k++)
#pragma analysis_check assert correctness_incoherent_out([NB*NB](A[i][k]), [NB*NB](B[k][j]))
#pragma omp task out([NB*NB](A[i][k]), [NB*NB](B[k][j])) inout([NB*NB](C[i][j]))
matmul ((double *)A[i][k], (double *)B[k][j], (double *)C[i][j], NB);
#pragma omp taskwait
}
#endif
#if 1
void compute_3(unsigned long NB, unsigned long DIM,
double *A[DIM][DIM], double *B[DIM][DIM], double *C[DIM][DIM])
{
unsigned i, j, k;
for (i = 0; i < DIM; i++)
for (j = 0; j < DIM; j++)
for (k = 0; k < DIM; k++)
#pragma analysis_check assert correctness_incoherent_in_pointed(A[i][k], B[k][j]) correctness_incoherent_out_pointed(C[i][j])
#pragma omp task in(A[i][k], B[k][j]) out(C[i][j])
matmul ((double *)A[i][k], (double *)B[k][j], (double *)C[i][j], NB);
#pragma omp taskwait
}
#endif
#if 1
void compute_4(unsigned long NB, unsigned long DIM,
double *A[DIM][DIM], double *B[DIM][DIM], double *C[DIM][DIM])
{
unsigned i, j, k;
for (i = 0; i < DIM; i++)
for (j = 0; j < DIM; j++)
for (k = 0; k < DIM; k++)
#pragma analysis_check assert correctness_incoherent_out_pointed(C[i][j]) correctness_incoherent_in_pointed(C[i][j]) correctness_incoherent_out(A[i][k], B[k][j])
#pragma omp task out(A[i][k], B[k][j]) inout(C[i][j])
matmul ((double *)A[i][k], (double *)B[k][j], (double *)C[i][j], NB);
#pragma omp taskwait
}
#endif
#if 1
void compute_5(unsigned long NB, unsigned long DIM,
double *A[DIM][DIM], double *B[DIM][DIM], double *C[DIM][DIM])
{
unsigned i, j, k;
for (i = 0; i < DIM; i++)
for (j = 0; j < DIM; j++)
for (k = 0; k < DIM; k++)
#pragma analysis_check assert correctness_incoherent_in_pointed(A[i], B[k], C[i]) correctness_incoherent_out_pointed(C[i])
#pragma omp task in(A[i], B[k]) inout(C[i])
matmul ((double *)A[i][k], (double *)B[k][j], (double *)C[i][j], NB);
#pragma omp taskwait
}
#endif