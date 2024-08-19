#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#define N       1024 
#define NTHR    4    
int A[N][N], B[N][N], C[N][N];
int readmat(char *fname, int *mat, int n),
writemat(char *fname, int *mat, int n);
int main(int argc, char **argv)
{
int        i, j, k, sum;
double     start_time, elapsed_time;
char      *sched_policy;
if (argc < 2)
{
fprintf(stderr, "Invalid number of command-line arguments!\n");
exit(EXIT_FAILURE);
}
sched_policy = strdup(argv[1]);
if (!sched_policy)
{
perror("strdup");
exit(errno);
}
if (strncmp(sched_policy, "dynamic", 7) == 0)
omp_set_schedule(omp_sched_dynamic, -1);
else if (strncmp(sched_policy, "static", 6) == 0)
omp_set_schedule(omp_sched_static, -1);
else {
fprintf(stderr, "Invalid sched policy: %s\n", sched_policy);
exit(EXIT_FAILURE);
}
if (readmat("Amat1024", (int *) A, N) < 0)
exit( 1 + printf("file problem\n") );
if (readmat("Bmat1024", (int *) B, N) < 0)
exit( 1 + printf("file problem\n") );
start_time = omp_get_wtime();
#pragma omp parallel for private(j,k,sum) schedule(runtime) num_threads(NTHR)
for (i = 0; i < N; i++)
for (j = 0; j < N; j++)
{
for (k = sum = 0; k < N; k++)
sum += A[i][k]*B[k][j];
C[i][j] = sum;
};
elapsed_time = omp_get_wtime() - start_time;
printf("Loop #0:\tschedule: %s\ttime: %lf sec.\n", sched_policy,
elapsed_time);
writemat("Cmat1024", (int *) C, N);
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
