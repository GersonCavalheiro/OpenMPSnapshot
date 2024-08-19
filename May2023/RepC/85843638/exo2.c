#include <stdio.h>
#include <omp.h>
#include <x86intrin.h>
#define NBEXPERIMENTS    22
static long long unsigned int experiments [NBEXPERIMENTS] ;
#define N              512
#define TILE           16
typedef double vector [N] ;
typedef double matrix [N][N] ;
static vector a, b, c ;
static matrix M1, M2 ;
long long unsigned int average (long long unsigned int *exps)
{
unsigned int i ;
long long unsigned int s = 0 ;
for (i = 2; i < (NBEXPERIMENTS-2); i++)
{
s = s + exps [i] ;
}
return s / (NBEXPERIMENTS-2) ;
}
void init_vector (vector X, const double val)
{
register unsigned int i ;
for (i = 0 ; i < N; i++)
X [i] = val ;
return ;
}
void init_matrix (matrix X, const double val)
{
register unsigned int i, j;
for (i = 0; i < N; i++)
{
for (j = 0 ;j < N; j++)
{
X [i][j] = val ;
}
}
}
void print_vectors (vector X, vector Y)
{
register unsigned int i ;
for (i = 0 ; i < N; i++)
printf (" X [%d] = %le Y [%d] = %le\n", i, X[i], i,Y [i]) ;
return ;
}
void add_vectors1 (vector X, vector Y, vector Z)
{
register unsigned int i ;
#pragma omp parallel for schedule(static)  
for (i=0; i < N; i++)
X[i] = Y[i] + Z[i];
return ;
}
void add_vectors2 (vector X, vector Y, vector Z)
{
register unsigned int i ;
omp_sched_t s ;
int chunk_size;
s = omp_sched_dynamic ;
omp_get_schedule(&s, &chunk_size); 
printf("Chunk size is : %d\n", chunk_size);
#pragma omp parallel for schedule(dynamic)  
for (i=0; i < N; i++)
{
X[i] = Y[i] + Z[i];
} 
return ;
}
double dot1 (vector X, vector Y)
{
register unsigned int i ;
register double dot ;
dot = 0.0 ;
#pragma omp parallel for schedule(static) reduction (+:dot)
for (i=0; i < N; i++)
dot += X [i] * Y [i];
return dot ;
}
double dot2 (vector X, vector Y)
{
register unsigned int i ;
register double dot ;
dot = 0.0 ;
#pragma omp parallel for schedule(dynamic) reduction (+:dot)
for (i=0; i < N; i++)
dot += X [i] * Y [i];
return dot ;
}
double dot3 (vector X, vector Y)
{
register unsigned int i ;
register double dot ;
dot = 0.0 ;
#pragma omp parallel for schedule(static) reduction (+:dot)
for (i = 0; i < N; i = i + 8)
{
dot += X [i] * Y [i];
dot += X [i + 1] * Y [i + 1];
dot += X [i + 2] * Y [i + 2];
dot += X [i + 3] * Y [i + 3];
dot += X [i + 4] * Y [i + 4];
dot += X [i + 5] * Y [i + 5];
dot += X [i + 6] * Y [i + 6];
dot += X [i + 7] * Y [i + 7];
}
return dot ;
}
void mult_mat_vect0 (matrix M, vector b, double *c)
{
return ;
}
void mult_mat_vect1 (matrix M, vector b, vector c)
{
return ;
}
void mult_mat_vect2 (matrix M, vector b, vector c)
{
return ;
}
void mult_mat_vect3 (matrix M, vector b, vector c)
{
return ;
}
void mult_mat_mat0 (matrix A, matrix B, matrix C)
{
return ;
}
void mult_mat_mat1 (matrix A, matrix B, matrix C)
{
return ;
}
void mult_mat_mat2 (matrix A, matrix B, matrix C)
{
return ;
}
void mult_mat_mat3 (matrix A, matrix B, matrix C)
{
return ;
}
int main ()  
{
int nthreads, maxnthreads ;
int tid;
unsigned long long int start, end ;
unsigned long long int residu ;
unsigned long long int av ;
double r ;
int exp ;
start = _rdtsc () ;
end = _rdtsc () ;
residu = end - start ;
nthreads = omp_get_num_threads();
maxnthreads = omp_get_max_threads () ;
printf("Sequential execution: \n# threads %d\nmax threads %d \n", nthreads, maxnthreads);
init_vector (a, 1.0) ;
init_vector (b, 2.0) ;
printf ("=============== ADD ==========================================\n") ;
init_vector (a, 1.0) ;
init_vector (b, 2.0) ;
for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
{
start = _rdtsc () ;
add_vectors1 (c, a, b) ;
end = _rdtsc () ;
experiments [exp] = end - start ;
}
av = average (experiments) ;
printf ("OpenMP static loop %Ld cycles\n", av-residu) ;
init_vector (a, 1.0) ;
init_vector (b, 2.0) ;
for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
{
start = _rdtsc () ;
add_vectors2 (c, a, b) ;
end = _rdtsc () ;
experiments [exp] = end - start ;
}
av = average (experiments) ;
printf ("OpenMP dynamic loop %Ld cycles\n", av-residu) ;
printf ("==============================================================\n") ;
printf ("====================DOT =====================================\n") ;
init_vector (a, 1.0) ;
init_vector (b, 2.0) ;
for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
{
start = _rdtsc () ;
r = dot1 (a, b) ;
end = _rdtsc () ;
experiments [exp] = end - start ;
}
av = average (experiments) ;
printf ("OpenMP static loop dot %e: %Ld cycles\n", r, av-residu) ;
init_vector (a, 1.0) ;
init_vector (b, 2.0) ;
for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
{
start = _rdtsc () ;
r = dot2 (a, b) ;
end = _rdtsc () ;
experiments [exp] = end - start ;
}
av = average (experiments) ;
printf ("OpenMP dynamic loop dot %e: %Ld cycles\n", r, av-residu) ;
init_vector (a, 1.0) ;
init_vector (b, 2.0) ;
for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
{
start = _rdtsc () ;
r = dot3 (a, b) ;
end = _rdtsc () ;
experiments [exp] = end - start ;
}
av = average (experiments) ;
printf ("OpenMP static unrolled loop dot %e: %Ld cycles\n", r, av-residu) ;
printf ("=============================================================\n") ;
printf ("======================== Mult Mat Vector =====================================\n") ;
init_matrix (M1, 1.0) ;
init_vector (b, 2.0) ;
for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
{
start = _rdtsc () ;
mult_mat_vect0 (M1, b, a) ; 
end = _rdtsc () ;
experiments [exp] = end - start ;
}
av = average (experiments) ;
printf ("OpenMP static loop MultMatVect0: %Ld cycles\n", av-residu) ;
init_matrix (M1, 1.0) ;
init_vector (b, 2.0) ;
for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
{
start = _rdtsc () ;
mult_mat_vect1 (M1, b, a) ; 
end = _rdtsc () ;
experiments [exp] = end - start ;
}
av = average (experiments) ;
printf ("OpenMP static loop MultMatVect1: %Ld cycles\n", av-residu) ;
init_matrix (M1, 1.0) ;
init_vector (b, 2.0) ;
for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
{
start = _rdtsc () ;
mult_mat_vect2 (M1, b, a) ; 
end = _rdtsc () ;
experiments [exp] = end - start ;
}
av = average (experiments) ;
printf ("OpenMP static loop MultMatVect2: %Ld cycles\n", av-residu) ;
init_matrix (M1, 1.0) ;
init_vector (b, 2.0) ;
for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
{
start = _rdtsc () ;
mult_mat_vect3 (M1, b, a) ; 
end = _rdtsc () ;
experiments [exp] = end - start ;
}
av = average (experiments) ;
printf ("OpenMP static loop MultMatVect3: %Ld cycles\n", av-residu) ;
printf ("==============================================================\n") ;
printf ("======================== Mult Mat Mat =====================================\n") ;
init_matrix (M1, 1.0) ;
init_matrix (M2, 2.0) ;
for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
{
start = _rdtsc () ;
mult_mat_mat0 (M1, M2, M2) ; 
end = _rdtsc () ;
experiments [exp] = end - start ;
}
av = average (experiments) ;
printf ("Sequential matrice vector multiplication:\t %Ld cycles\n", av-residu) ;
init_matrix (M1, 1.0) ;
init_matrix (M2, 2.0) ;
for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
{
start = _rdtsc () ;
mult_mat_mat1 (M1, M2, M2) ; 
end = _rdtsc () ;
experiments [exp] = end - start ;
}
av = average (experiments) ;
printf ("OpenMP static loop MultMatVect1:\t\t %Ld cycles\n", av-residu) ;
init_matrix (M1, 1.0) ;
init_matrix (M2, 2.0) ;
for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
{
start = _rdtsc () ;
mult_mat_mat2 (M1, M2, M2) ; 
end = _rdtsc () ;
experiments [exp] = end - start ;
}
av = average (experiments) ;
printf ("OpenMP unrolled loop MultMatMat2: %Ld cycles\n", av-residu) ;
init_matrix (M1, 1.0) ;
init_matrix (M2, 2.0) ;
for (exp = 0 ; exp < NBEXPERIMENTS; exp++)
{
start = _rdtsc () ;
mult_mat_mat3 (M1, M2, M2) ; 
end = _rdtsc () ;
experiments [exp] = end - start ;
}
av = average (experiments) ;
printf ("OpenMP Tiled loop MultMatMat3: %Ld cycles\n", av-residu) ;
return 0;
}
