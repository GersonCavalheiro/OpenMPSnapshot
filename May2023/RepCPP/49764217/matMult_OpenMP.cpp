

#include <iostream>
#include <cstdlib>
#include <omp.h>
#include <mpi.h>
#include <time.h>

#define N 500               

using namespace std;

int main(int argc, char* argv[])
{
float *A, *B, *C;         
int max_threads;		  
double start, end;		  
size_t matrix_size = N;   

A = new float[N*N];
B = new float[N*N];
C = new float[N*N];

for(int i=0; i<N; i++)  
{
for(int j=0; j<N; j++)
{                         
A[i] = i+j;
B[i] = i*j;
}
}	

start=MPI_Wtime();		   			            
max_threads = omp_get_max_threads();   			

cout<<"Max threads available :"<<max_threads<<endl;

#pragma omp parallel for shared (A, B, C, matrix_size) 

for(int i = 0; i < N; i++)
{
for(int j = 0; j < N; j++)
{
C[j + i*N] = 0;
for(int k = 0; k < N; k++)
{
C[j + i*N] += A[i*N + k]*B[j + N*k];
}    
} 
}

end=MPI_Wtime();				 			    

cout<<"Time Taken : "<< (end-start) <<" seconds"<<endl;
}
