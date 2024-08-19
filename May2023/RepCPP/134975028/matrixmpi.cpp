#include <stdio.h>
#include <omp.h>
#include <mpi.h>
#include <string>
#define N 512

using namespace std;

int main(int argc, char **argv)
{
MPI_Init(NULL, NULL);
int mA[N][N];
int mB[N][N];
int mC[N][N];
int mCtemp[N][N];
int threads;	
int i,h;
int world_size;
MPI_Comm_size(MPI_COMM_WORLD, &world_size);
int world_rank;
MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
for(h=0;h<1;h++)
{
if(world_rank == 0)
{
for(i = 0; i < N; i++)
{
int j;
for(j = 0; j < N; j++)
{
mA[i][j] = (i+1)*(j+1);
mB[i][j] = mA[i][j];
}
}	
}
MPI_Barrier(MPI_COMM_WORLD);

MPI_Bcast(mA, N*N, MPI_INT, 0, MPI_COMM_WORLD);
MPI_Bcast(mB, N*N, MPI_INT, 0, MPI_COMM_WORLD);
#pragma omp parallel for shared(mA,mB,mC)
for(i = world_rank * N/ world_size; i < (world_rank + 1) * N/ world_size; i++)
{
int j;
for(j = 0; j < N; j++)
{
int k;
int sum = 0;
for(k = 0; k < N; k++)
sum += mA[i][k]*mB[k][j];
mCtemp[i][j] = sum;
}
}
MPI_Gather(&mCtemp[world_rank * N / world_size][0], N * N/ world_size, MPI_INT, mC, N * N/ world_size, MPI_INT, 0, MPI_COMM_WORLD);
MPI_Finalize();
}
return mC[2][4];
}