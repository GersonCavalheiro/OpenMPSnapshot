#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define N       1024   
#define NTHR    4      
int A[N][N], B[N][N], C[N][N];
int  readmat(char *fname, int *mat, int n),
writemat(char *fname, int *mat, int n);
void checkerboard(int x, int y, int s);
int main(int argc, char **argv)
{
int      m,        
ntasks,   
task_id,  
s,        
x, y;     
double   start_time, elapsed_time;
if (argc < 2)
{
fprintf(stderr, "Invalid number of command-line arguments!\n");
exit(EXIT_FAILURE);
}
ntasks = atoi(argv[1]);
if (ntasks < 1)
{
fprintf(stderr, "Invalid number of OpenMP tasks!\n");
exit(EXIT_FAILURE);
}
m = (int) sqrt((double)ntasks);
if (m*m != ntasks)
{
fprintf(stderr, "Invalid number of OpenMP tasks!\n");
fprintf(stderr, "Could not calculate exact square root!\n");
exit(EXIT_FAILURE);
}
s = N/m;
if (s*m != N)
{
fprintf(stderr, "Invalid number of OpenMP tasks!\n");
fprintf(stderr, "The square root of the number of tasks " \
"should evenly divide %d!\n", N);
exit(EXIT_FAILURE);
}
if (readmat("Amat1024", (int *) A, N) < 0)
exit( 1 + printf("file problem\n") );
if (readmat("Bmat1024", (int *) B, N) < 0)
exit( 1 + printf("file problem\n") );
start_time = omp_get_wtime();
#pragma omp parallel num_threads(NTHR)
{
#pragma omp single nowait
{
for (task_id = 0; task_id < ntasks; task_id++)
{
x = task_id / m;
y = task_id % m;
#pragma omp task firstprivate(x,y,s)
{
checkerboard(x,y,s);
}
}
}
}
elapsed_time = omp_get_wtime() - start_time;
printf("time: %lf sec.\n", elapsed_time);
writemat("Cmat1024", (int *) C, N);
return (EXIT_SUCCESS);
}
void checkerboard(int x, int y, int s)
{
int i, j, k, sum;
for (i = x*s; i < (x+1)*s; i++)
for (j = y*s; j < (y+1)*s; j++)
{
for (k = sum = 0; k < N; k++)
sum += A[i][k]*B[k][j];
C[i][j] = sum;
};
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
