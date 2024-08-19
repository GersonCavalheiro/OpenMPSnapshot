#include <iostream>
#include <windows.h>
#include <pthread.h>
#include <windows.h>
#include <xmmintrin.h> 
#include <emmintrin.h> 
#include <pmmintrin.h> 
#include <tmmintrin.h> 
#include <smmintrin.h> 
#include <nmmintrin.h> 
#include <immintrin.h> 
#include <semaphore.h>
#include <mpi.h>
#include <stdint.h>

#if def_OPENMP
#include<omp.h>
#endif

using namespace std;

#define n 10
#define thread_count 1

float A[n][n];
int id[thread_count];
long long head, tail , freq;
sem_t sem_parent;
pthread_barrier_t childbarrier_row;
pthread_barrier_t childbarrier_col;

struct data
{
int id;
int begin;
int end;
int myid;
}datagroups[thread_count];

void init()
{
for(int i=0;i<n;i++)
for(int j=i;j<n;j++)
A[i][j]=i+j+2;

for(int i=1;i<n;i++)
for(int j=0;j<n;j++)
A[i][j]=A[i][j]+A[0][j];
for(int i=0;i<n;i++)
for(int j=0;j<n;j++)
A[j][i]=A[j][i]+A[j][0];
}
void printA()
{
for(int i=0;i<n;i++)
{
for(int j=0;j<n;j++)
cout<<A[i][j]<<" ";
cout<<endl;
}
}
void normal_gausseliminate()
{
for(int k=0;k<n;k++)
{

for(int j=k+1;j<n;j++)
{
A[k][j]=A[k][j]/A[k][k];
}
A[k][k]=1;
for(int i=k+1;i<n;i++)
{
for(int j=k+1;j<n;j++)
{
A[i][j]=A[i][j]-A[i][k]*A[k][j];
}
A[i][k]=0;
}
}
}


int main(int argc,char* argv[]){
QueryPerformanceFrequency((LARGE_INTEGER *)&freq);
pthread_barrier_init(&childbarrier_row, NULL,thread_count+1);
pthread_barrier_init(&childbarrier_col,NULL, thread_count+1);
sem_init(&sem_parent, 0, 0);
pthread_t threadID[thread_count];

int myid, numprocs;
MPI_Init(&argc,&argv);
MPI_Comm_rank(MPI_COMM_WORLD,&myid);
MPI_Comm_size(MPI_COMM_WORLD,&numprocs);
int distributecol=n/(numprocs);
if(myid==0)
{
init();
for(int i=1;i< numprocs;i++)
{
int begin=i*distributecol;
int end=begin+distributecol;
if(i==numprocs-1)
end=n;
MPI_Datatype block;
MPI_Type_vector(n,(end-begin),n,MPI_FLOAT,&block);
MPI_Type_commit(&block);
MPI_Send((void *)(A[0]+begin),1,block,i,0,MPI_COMM_WORLD);
}
}
else
{
int begin=myid*distributecol;
int end=begin+distributecol;
if(myid==numprocs-1)
end=n;
MPI_Datatype block;
MPI_Type_vector(n,(end-begin),n,MPI_FLOAT,&block);
MPI_Type_commit(&block);
MPI_Recv((void *)(A[0]+begin),1,block,0,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
}

int begin=(myid)*distributecol;
int end=begin+distributecol;
if(myid==numprocs-1)
end=n;
# pragma omp parallel for num_threads(thread_count)\
shared(A)
for(int k=0;k<n;k++)
{
int source=k/distributecol;
if(source>=numprocs)
source=numprocs-1;
MPI_Datatype temcol;
MPI_Type_vector(n-k,1,n,MPI_FLOAT,&temcol);
MPI_Type_commit(&temcol);
MPI_Bcast((void *)(A[k]+k),1,temcol,source,MPI_COMM_WORLD);

for(int j=(begin>=(k+1)?begin:(k+1));j<end;j++)
A[k][j]=A[k][j]/A[k][k];
A[k][k]=1;

for(int j=k+1;j<n;j++)
{
for(int i=(begin>=(k+1)?begin:(k+1));i<end;i++)
{
A[j][i]=A[j][i]-A[j][k]*A[k][i];
}
A[j][k]=0;
}

}


if(myid!=0)
{
MPI_Datatype block;
MPI_Type_vector(n,(end-begin),n,MPI_FLOAT,&block);
MPI_Type_commit(&block);
MPI_Send((void *)(A[0]+begin),1,block,0,1,MPI_COMM_WORLD);
}   
else
{
for(int i=1;i<numprocs;i++)
{
int begin=i*distributecol;
int end=begin+distributecol;
if(i==numprocs-1)
end=n;
MPI_Datatype block;
MPI_Type_vector(n,(end-begin),n,MPI_FLOAT,&block);
MPI_Type_commit(&block);
MPI_Recv((void *)(A[0]+begin),1,block,i,1,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
} 
}
if(myid==0)
printA();

MPI_Finalize();

}
