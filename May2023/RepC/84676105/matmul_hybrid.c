#include <omp.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#define N		1024 
#define OMP_NTHR	4    
#define ALLOC_ERR(arr) \
{perror("malloc ["arr"]"); \
MPI_Abort(MPI_COMM_WORLD, errno);}
#define ERROR(msg) \
{fprintf(stderr, msg); \
MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);}
enum stat_e {CALC, COMM, MIN_CALC, MIN_COMM, MAX_CALC, MAX_COMM};
int   readmat(char *fname, int *mat, int n),
writemat(char *fname, int *mat, int n);
int main(int argc, char **argv)
{
int     myrank, nnodes, WORK, i, j, k, sum,
*a_strip, *c_strip, *A, *C, B[N][N];
double  t0, t1, t2, t3, time[2], stats[6];
MPI_Init(&argc, &argv);
MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
MPI_Comm_size(MPI_COMM_WORLD, &nnodes);
WORK = N / nnodes;
a_strip = (int *) malloc(WORK * N * sizeof(int));
if (!a_strip)
ALLOC_ERR("a_strip");
c_strip = (int *) malloc(WORK * N * sizeof(int));
if (!c_strip)
ALLOC_ERR("c_strip");
if (myrank == 0)
{
A = (int *) malloc(N * N * sizeof(int));
if (!A)
ALLOC_ERR("A");
C = (int *) malloc(N * N * sizeof(int));
if (!C)
ALLOC_ERR("C");
if (readmat("Amat1024", A, N) < 0)
ERROR("Amat1024: File error\n");
if (readmat("Bmat1024", (int *) B, N) < 0)
ERROR("Bmat1024: File error\n");
}
t0 = MPI_Wtime();
MPI_Bcast(B, N*N, MPI_INT, 0, MPI_COMM_WORLD);
MPI_Scatter(A, WORK*N, MPI_INT, a_strip, WORK*N, MPI_INT, 0, MPI_COMM_WORLD);
t1 = MPI_Wtime();
#pragma omp parallel for private(j,k,sum) schedule(guided) num_threads(OMP_NTHR)
for (i = 0; i < WORK; i++)
for (j = 0; j < N; j++)
{
for (k = sum = 0; k < N; k++)
sum += a_strip[i*N + k]*B[k][j];
c_strip[i*N + j] = sum;
};
t2 = MPI_Wtime();
MPI_Gather(c_strip, WORK*N, MPI_INT, C, WORK*N, MPI_INT, 0, MPI_COMM_WORLD);
t3 = MPI_Wtime();
time[CALC] = t2 - t1;
time[COMM] = t3 - t0 - time[CALC];
MPI_Reduce(time, stats,            2, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
MPI_Reduce(time, stats + MIN_CALC, 2, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
MPI_Reduce(time, stats + MAX_CALC, 2, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
if (myrank == 0)
{
stats[CALC] /= nnodes;
stats[COMM] /= nnodes;
printf("Time:\n");
printf("   Total   : %2lf\n", stats[CALC] + stats[COMM]);
printf("   Avg Calc: %2lf\n", stats[CALC]);
printf("   Max Calc: %2lf\n", stats[MAX_CALC]);
printf("   Min Calc: %2lf\n", stats[MIN_CALC]);
printf("   Avg Comm: %2lf\n", stats[COMM]);
printf("   Max Comm: %2lf\n", stats[MAX_COMM]);
printf("   Min Comm: %2lf\n", stats[MIN_COMM]);
writemat("Cmat1024", C, N);
}
MPI_Finalize();
return (EXIT_SUCCESS);
}
#define _mat(i,j) (mat[(i)*n + (j)])
int readmat(char *fname, int *mat, int n)
{
FILE *fp;
int  i, j;
if ((fp = fopen(fname, "r")) == NULL)
return (-1);
for (i = 0; i < n; i++)
for (j = 0; j < n; j++)
if (fscanf(fp, "%d", &_mat(i,j)) == EOF)
{
fclose(fp);
return (-1);
};
fclose(fp);
return (EXIT_SUCCESS);
}
int writemat(char *fname, int *mat, int n)
{
FILE *fp;
int  i, j;
if ((fp = fopen(fname, "w")) == NULL)
return (-1);
for (i = 0; i < n; i++, fprintf(fp, "\n"))
for (j = 0; j < n; j++)
fprintf(fp, " %d", _mat(i, j));
fclose(fp);
return (EXIT_SUCCESS);
}
