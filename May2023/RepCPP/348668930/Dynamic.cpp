#include <stdio.h>
#include <math.h>
#include <sys/time.h> 
#include <sys/resource.h>  
#include <omp.h>

double timeGetTime() 
{     
struct timeval time;     
struct timezone zone;     
gettimeofday(&time, &zone);     
return time.tv_sec + time.tv_usec*1e-6; 
}  


const long int VERYBIG = 100000;

int main( void )
{
int i;
long int j, k, sum;
double sumx, sumy, total, z;
double starttime, elapsedtime;
int iterationNumber = 6;
double times[iterationNumber];
double allExperimentTime = 0;
double averageTime;
double threadTimes[4];
printf( "OpenMP Parallel Timings for %ld iterations \n\n", VERYBIG );

for( i=0; i<iterationNumber; i++ )
{
printf("Iteration %d:\n", i+1);
starttime = timeGetTime();
sum = 0;
total = 0.0;

double wtime;

#pragma omp parallel num_threads(4) private( sumx, sumy, k, wtime ) shared(threadTimes)
{
wtime = omp_get_wtime();
#pragma omp for reduction( +: sum, total ) schedule( dynamic, 2000 ) nowait
for( int j=0; j<VERYBIG; j++ )
{
sum += 1;

sumx = 0.0;
for( k=0; k<j; k++ )
sumx = sumx + (double)k;

sumy = 0.0;
for( k=j; k>0; k-- )
sumy = sumy + (double)k;

if( sumx > 0.0 )total = total + 1.0 / sqrt( sumx );
if( sumy > 0.0 )total = total + 1.0 / sqrt( sumy );
}
wtime = omp_get_wtime() - wtime;
int threadNumber = omp_get_thread_num();
threadTimes[threadNumber] = wtime;
}

elapsedtime = timeGetTime() - starttime;
times[i] = elapsedtime;
allExperimentTime += elapsedtime;
for (int i = 0; i < 4; i++)
printf("\tThread %d Calculation Time: %.3f Seconds\n", i, threadTimes[i]); 

printf("\tTime Elapsed %10d mSecs Total=%lf Check Sum = %ld\n",
(int)(elapsedtime * 1000), total, sum );
}
averageTime = allExperimentTime / iterationNumber;
printf("Average Execution Time: %.3f Seconds\n", averageTime);

return 0;
}
