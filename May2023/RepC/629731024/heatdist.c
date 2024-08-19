

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h> 
#include <string.h>
#include <omp.h>


#define index(i, j, N)  ((i)*(N)) + (j)


int numthreads = 0;



void  seq_heat_dist(float *, unsigned int, unsigned int);
void  parallel_heat_dist(float *, unsigned int, unsigned int);
void  check_result(int, unsigned int, float * ); 




int main(int argc, char * argv[])
{
unsigned int N; 
int which_code = 0; 
int iterations = 0;
int i,j;


float * playground; 

double time_taken;
clock_t start, end;

if(argc != 5)
{
fprintf(stderr, "usage: heatdist num  iterations  who\n");
fprintf(stderr, "num = dimension of the square matrix \n");
fprintf(stderr, "iterations = number of iterations till stopping (1 and up)\n");
fprintf(stderr, "who = 0: sequential code on CPU, 1: OpenMP version\n");
fprintf(stderr, "threads = number of threads for the  OpenMP version\n");
exit(1);
}

which_code = atoi(argv[3]);
N = (unsigned int) atoi(argv[1]);
iterations = (unsigned int) atoi(argv[2]);
numthreads = (unsigned int) atoi(argv[4]);



playground = (float *)calloc(N*N, sizeof(float));
if( !playground )
{
fprintf(stderr, " Cannot allocate the %u x %u array\n", N, N);
exit(1);
}


for(i = 0; i< N; i++) playground[index(i,0,N)] = 100;
for(i = 0; i< N; i++) playground[index(i,N-1,N)] = 100;
for(j = 0; j< N; j++) playground[index(0,j,N)] = 100;
for(j = 0; j< N; j++) playground[index(N-1,j,N)] = 100;

switch(which_code)
{
case 0: printf("CPU sequential version:\n");
start = clock();
seq_heat_dist(playground, N, iterations);
end = clock();
break;

case 1: printf("OpenMP version:\n");
start = clock();
parallel_heat_dist(playground, N, iterations); 
end = clock();  
check_result(iterations, N, playground); 
break;


default: printf("Invalid device type\n");
exit(1);
}

time_taken = ((double)(end - start))/ CLOCKS_PER_SEC;

printf("Time taken = %lf\n", time_taken);

free(playground);

return 0;

}



void  seq_heat_dist(float * playground, unsigned int N, unsigned int iterations)
{
int i, j, k;
int upper = N-1; 

unsigned int num_bytes = 0;

float * temp; 


temp = (float *)calloc(N*N, sizeof(float));
if( !temp )
{
fprintf(stderr, " Cannot allocate temp %u x %u array\n", N, N);
exit(1);
}


num_bytes = N*N*sizeof(float);


memcpy((void *)temp, (void *) playground, num_bytes);

for( k = 0; k < iterations; k++)
{

for(i = 1; i < upper; i++)
for(j = 1; j < upper; j++)
temp[index(i,j,N)] = (playground[index(i-1,j,N)] + 
playground[index(i+1,j,N)] + 
playground[index(i,j-1,N)] + 
playground[index(i,j+1,N)])/4.0;




memcpy((void *)playground, (void *) temp, num_bytes);
}

free(temp);
}


void check_result(int iterations, unsigned int N, float * playground){

float * temp;
int i, j;

temp = (float *)calloc(N*N, sizeof(float));
if( !temp )
{
fprintf(stderr, " Cannot allocate temp %u x %u array in check_result\n", N, N);
exit(1);
}


for(i = 0; i< N; i++) temp[index(i,0,N)] = 100;
for(i = 0; i< N; i++) temp[index(i,N-1,N)] = 100;
for(j = 0; j< N; j++) temp[index(0,j,N)] = 100;
for(j = 0; j< N; j++) temp[index(N-1,j,N)] = 100;

seq_heat_dist(temp, N, iterations);

for(i = 0; i < N; i++)
for (j = 0; j < N; j++)
if(fabsf(playground[index(i, j, N)] - temp[index(i, j, N)]) > 0.01)
{
printf("play[%d %d] = %f   temp[%d %d] = %f  index = %d\n", i, j, playground[index(i, j, N)], i, j, temp[index(i, j, N)], index(i, j, N));
printf("There is a mismatch in some elements between the sequential and parallel version\n");
free(temp);
return;
}

printf("Result is correct!\n");
free(temp);



}






void  parallel_heat_dist(float * playground, unsigned int N, unsigned int iterations)
{
int i, j, k;
int upper = N-1; 

unsigned int num_bytes = 0;

float * temp; 


temp = (float *)calloc(N*N, sizeof(float));
if( !temp )
{
fprintf(stderr, " Cannot allocate temp %u x %u array\n", N, N);
exit(1);
}


num_bytes = N*N*sizeof(float);


memcpy((void *)temp, (void *) playground, num_bytes);
#pragma omp parallel num_threads(numthreads) private(k)
for (k = 0; k < iterations; k++)
{

#pragma omp for schedule(static) collapse(2)
for (i = 1; i < upper; i++)
{
for (j = 1; j < upper; j++){
temp[index(i, j, N)] = (playground[index(i - 1, j, N)] +
playground[index(i + 1, j, N)] +
playground[index(i, j - 1, N)] +
playground[index(i, j + 1, N)]) / 4.0;
}
}


#pragma omp for schedule(static) collapse(2)
for (i = 1; i < upper; i++)
{
for (j = 1; j < upper; j++)
playground[index(i, j, N)] = temp[index(i, j, N)];
}
}
free(temp); 
}



