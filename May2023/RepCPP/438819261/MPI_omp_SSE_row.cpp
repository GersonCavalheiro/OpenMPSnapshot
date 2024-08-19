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

#define n 20
#define thread_count 4

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
int distributerow=n/(numprocs-1);
if(myid==0)
{
init();
for(int i=1;i<numprocs;i++)
{
int begin=(i-1)*distributerow;
int end=begin+distributerow;
if(i==numprocs-1)
end=n;
int count=(end-begin)*n;
MPI_Send((void *)A[begin],count,MPI_FLOAT,i,0,MPI_COMM_WORLD);
}
printA();
}
else
{
int begin=(myid-1)*distributerow;
int end=begin+distributerow;
if(myid==numprocs-1)
end=n;
int count=(end-begin)*n;
MPI_Recv((void *)A[begin],count,MPI_FLOAT,0,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
}

int begin=(myid-1)*distributerow;
int end=begin+distributerow;
if(myid==numprocs-1)
end=n;
int count=(end-begin)*n;


for(int k=0;k<n;k++)
{
if(myid==0)
{
if(k!=0)
{
int source=(k/distributerow+1)<(numprocs-1)?(k/distributerow+1):(numprocs-1);
MPI_Recv((void *)(A[k]+k), n-k, MPI_FLOAT, source, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}
__m128 t1,t2,t3;
int preprocessnumber=(n-k-1)%4;
int begincol=k+1+preprocessnumber;
float head[4]={A[k][k],A[k][k],A[k][k],A[k][k]};
t2=_mm_loadu_ps(head);
for(int j=k+1;j<k+1+preprocessnumber;j++)
{
A[k][j]=A[k][j]/A[k][k];
}
# pragma omp parallel for num_threads(thread_count)\
shared(A)
for(int j=begincol;j<n;j=j+4)
{
t1=_mm_loadu_ps(A[k]+j);
t1=_mm_div_ps(t1,t2);
_mm_storeu_ps(A[k]+j,t1);
}
A[k][k]=1;          
}
MPI_Bcast((void *)(A[k]+k),n-k,MPI_FLOAT,0,MPI_COMM_WORLD);

if(myid!=0)
{


__m128 t1,t2,t3;
int preprocessnumber=(n-k-1)%4;
int begincol=k+1+preprocessnumber;
for(int i=k+1;i<n;i++)
{
for(int j=k+1;j<k+1+preprocessnumber;j++)
{
A[i][j]=A[i][j]-A[i][k]*A[k][j];
}
A[i][k]=0;
}
# pragma omp parallel for num_threads(thread_count)\
shared(A)
for(int i=k+1;i<n;i++)
{
float head1[4]={A[i][k],A[i][k],A[i][k],A[i][k]};
t3=_mm_loadu_ps(head1);
for(int j=begincol;j<n;j+=4)
{
t1=_mm_loadu_ps(A[k]+j);
t2=_mm_loadu_ps(A[i]+j);
t1=_mm_mul_ps(t1,t3);
t2=_mm_sub_ps(t2,t1);
_mm_storeu_ps(A[i]+j,t2);
}
A[i][k]=0;
}
if((k+1<n)&&(k+1)>=begin&&(k+1)<end)
{
MPI_Send((void *)(A[k+1]+k+1), n-(k+1), MPI_FLOAT, 0, 2, MPI_COMM_WORLD);
}
}
}
if(myid!=0)
{
MPI_Send((void *)A[begin],count,MPI_FLOAT,0,1,MPI_COMM_WORLD);
}   
else
{
for(int i=1;i<numprocs;i++)
{
int begin=(i-1)*distributerow;
int end=begin+distributerow;
if(i==numprocs-1)
end=n;
int count=(end-begin)*n;
MPI_Recv((void *)A[begin],count,MPI_FLOAT,i,1,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
} 
}
if(myid==0)
printA();
MPI_Finalize();
}
