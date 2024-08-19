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
double threadStarttime[4], threadElapsedtime[4];
printf( "OpenMP Parallel Timings for %ld iterations, (dynamic, 2000) \n\n", VERYBIG );

for( i=0; i<6; i++ )
{
starttime = timeGetTime();
sum = 0;
total = 0.0;

#pragma omp parallel     \
num_threads (4) \
private( sumx, sumy, k ) shared(threadStarttime, threadElapsedtime) 
{
threadStarttime[omp_get_thread_num()] = timeGetTime();

#pragma omp for reduction( +: sum, total ) schedule(dynamic, 2000) nowait
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

threadElapsedtime[omp_get_thread_num()] = timeGetTime();
}

elapsedtime = timeGetTime() - starttime;

printf("Thread report time: ");
for( int z=0; z<4; z++)
printf("Thread %d had time %10d, ", z, (int) ((threadElapsedtime[z]-threadStarttime[z])*1000));
printf("\n");

printf("Time Elapsed %10d mSecs Total=%lf Check Sum = %ld\n",
(int)(elapsedtime * 1000), total, sum );
}

return 0;
}
