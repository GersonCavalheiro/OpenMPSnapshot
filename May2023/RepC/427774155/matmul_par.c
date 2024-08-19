

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>
#include <omp.h>

#define N 2048
#define Afile "Amat2048"
#define Bfile "Bmat2048"
#define Cfile "CmatPar2048"
#define _NUM_THREADS 4

int A[N][N], B[N][N], C[N][N];
int readmat(char *fname, int *mat, int n), 
writemat(char *fname, int *mat, int n);

int main(int argc, char **argv)
{
int i, j, k, sum;
int WORK, nproc, myid, (*Aa)[N],(*Bb)[N],(*Cc)[N];
double start, end, t1, t2, total, t3, t4, overheads, comp_time;

start = MPI_Wtime();
MPI_Init(&argc, &argv);
MPI_Comm_rank(MPI_COMM_WORLD, &myid);
MPI_Comm_size(MPI_COMM_WORLD, &nproc);

WORK = N/nproc; 

if(myid == 0){

printf("Parallel matrix multiplication...\n");

t1 = MPI_Wtime();
if (readmat(Afile, (int *) A, N) < 0){
exit( 1 + printf("file problem\n") );
}
if (readmat(Bfile, (int *) B, N) < 0){
exit( 1 + printf("file problem\n") );
}
t2 = MPI_Wtime();
}

Aa = (int(*)[N]) malloc(WORK*N*sizeof(int)); 
Cc = (int(*)[N]) malloc(WORK*N*sizeof(int)); 

MPI_Scatter(A, WORK*N, MPI_INT, Aa, WORK*N, MPI_INT, 0, MPI_COMM_WORLD); 
MPI_Bcast(B, N*N, MPI_INT, 0, MPI_COMM_WORLD); 

t3 = MPI_Wtime();
omp_set_num_threads(_NUM_THREADS);
#pragma omp parallel for private(j, k, sum) shared(Aa, B, Cc) 
for (i = 0; i < WORK; i++)
for (j = 0; j < N; j++)
{
for (k = sum = 0; k < N; k++){
sum += Aa[i][k]*B[k][j];
}
Cc[i][j] = sum; 
}
t4 = MPI_Wtime();

MPI_Gather(Cc, WORK*N, MPI_INT, C, N*WORK, MPI_INT, 0, MPI_COMM_WORLD); 

free(Aa);
free(Cc);

if(myid == 0){
end = MPI_Wtime();
total = (end - start) - (t2 - t1);
overheads = total - (t4 - t3);
comp_time = t4 - t3;


writemat(Cfile, (int *) C, N);
printf("Total time: %lf \nOverheads time: %lf \nComputation time: %lf \n", total, overheads, comp_time);
}


MPI_Finalize();
return (0);
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
return (0);
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
return (0);
}
