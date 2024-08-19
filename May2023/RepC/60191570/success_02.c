#include <stdlib.h>
const unsigned int N = 1024;
const unsigned int CHUNK_SIZE = 128;
const unsigned int N_CHUNKS = 8;
#if 1
double dot_product_1(double A[N], double B[N])
{
long actual_size;
int j;
double acc;
double *C = malloc (N_CHUNKS*sizeof(double));   
acc=0.0;
j=0;
long i;
for (i=0; i<N; i+=CHUNK_SIZE) {
actual_size = (N-CHUNK_SIZE>=CHUNK_SIZE) ? CHUNK_SIZE : (N-CHUNK_SIZE);
#pragma analysis_check assert correctness_auto_storage(A, B, C) correctness_incoherent_in_pointed(A, B) correctness_incoherent_out(A, B) correctness_incoherent_in(C) correctness_race(C[j])
#pragma omp task label( dot_prod ) firstprivate(j, i, actual_size) in(C) inout(A, B)
{
C[j]=0;
long ii;
for (ii=0; ii<actual_size; ii++)
C[j]+= A[i+ii] * B[i+ii];
}
#pragma analysis_check assert correctness_race(acc, C[j]) correctness_auto_storage(acc)
#pragma omp task label(increment) firstprivate(j) shared(acc)
acc += C[j];
j++;
}
return(acc);
}
#endif
#if 1
double dot_product_2(double A[N], double B[N])
{
long actual_size;
int j;
double acc;
double *C = malloc (N_CHUNKS*sizeof(double));   
acc=0.0;
j=0;
long i;
for (i=0; i<N; i+=CHUNK_SIZE) {
actual_size = (N-CHUNK_SIZE>=CHUNK_SIZE) ? CHUNK_SIZE : (N-CHUNK_SIZE);
#pragma analysis_check assert correctness_incoherent_out_pointed(C) correctness_incoherent_out([N]A, [N]B) correctness_race(C[j])
#pragma omp task label(dot_prod) firstprivate(j, i, actual_size, N) out(C) inout([N]A, [N]B)
{
C[j]=0;
long ii;
for (ii=0; ii<actual_size; ii++)
C[j]+= A[i+ii] * B[i+ii];
}
#pragma analysis_check assert correctness_race(acc, C[j])
#pragma omp task label(increment) firstprivate(j) shared(acc)
acc += C[j];
j++;
}
#pragma omp taskwait
return(acc);
}
#endif
#if 1
double dot_product_3(double A[N], double B[N])
{
long actual_size;
int j;
double acc;
double *C = malloc (N_CHUNKS*sizeof(double));
acc=0.0;
j=0;
long i;
for (i=0; i<N; i+=CHUNK_SIZE) {
actual_size = (N-CHUNK_SIZE>=CHUNK_SIZE) ? CHUNK_SIZE : (N-CHUNK_SIZE);
#pragma analysis_check assert correctness_race(C[j]) correctness_incoherent_out_pointed(C)
#pragma omp task label(dot_prod) firstprivate(j, i, actual_size, N) out(C) in([N]A, [N]B)
{
C[j]=0;
long ii;
for (ii=0; ii<actual_size; ii++)
C[j]+= A[i+ii] * B[i+ii];
}
#pragma analysis_check assert correctness_dead(acc) correctness_race(C[j])
#pragma omp task label(increment) firstprivate(j) in(C[j])
acc += C[j];
j++;
}
#pragma omp taskwait
return(acc);
}
#endif
#if 1
double dot_product_4(double A[N], double B[N])
{
long actual_size;
int j;
double acc;
double *C = malloc (N_CHUNKS*sizeof(double));
acc=0.0;
j=0;
long i;
for (i=0; i<N; i+=CHUNK_SIZE) {
actual_size = (N-CHUNK_SIZE>=CHUNK_SIZE) ? CHUNK_SIZE : (N-CHUNK_SIZE);
#pragma analysis_check assert correctness_race(C[j]) correctness_incoherent_p(j)
#pragma omp task label(dot_prod) private(j) firstprivate(i, actual_size, N) out(C[j]) in([N]A, [N]B)
{
C[j]=0;
long ii;
for (ii=0; ii<actual_size; ii++)
C[j]+= A[i+ii] * B[i+ii];
}
#pragma analysis_check assert correctness_race(C[j]) correctness_incoherent_in(C[j+1]) correctness_dead(acc)
#pragma omp task label(increment) firstprivate(j) commutative(acc) in(C[j+1])
acc += C[j];
j++;
}
#pragma omp taskwait
return(acc);
}
#endif
