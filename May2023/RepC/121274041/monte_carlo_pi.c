#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <sys/time.h>
#include <time.h>
void Usage(char* prog_name);
static long MULTIPLIER = 1366;
static long ADDEND = 150889;
static long PMOD = 714025;
long random_last = 0;
#pragma omp threadprivate(random_last)
double randomU()
{
long random_next;
random_next = (MULTIPLIER*random_last + ADDEND)%PMOD;
random_last = random_next;
return ((double)random_next/(double)PMOD);
}
int main(int argc, char* argv[]) {
long long n, i;
double x=0.0,y=0.0;                     
int thread_count;
long long count=0;                     
double z=0.0;                          
double pi=0.0;                        
if (argc != 3) Usage(argv[0]);
thread_count = strtol(argv[1], NULL, 10);
n = strtoll(argv[2], NULL, 10);
if (thread_count < 1 || n < 1) Usage(argv[0]);
struct timeval start, end;
gettimeofday(&start, NULL);
#pragma omp parallel for num_threads(thread_count) reduction(+: count) private(x,y,z,i)
for (i=0; i<n; ++i) {
x = (double)randomU();      
y = (double)randomU();      
z = ((x*x)+(y*y));          
if (z<=1) {
++count;            
}
}
gettimeofday(&end, NULL);    
printf("\n Execution Time: %fs \n", ((end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec)/1000000.0));
pi = ((double)count/(double)(n))*4.0;
printf("With n = %lld terms and %d threads,\n", n, thread_count);
printf("   Our estimate of pi = %.14f\n", pi);
printf("                   pi = %.14f\n", 4.0*atan(1.0));
return 0;
}  
void Usage(char* prog_name) {
fprintf(stderr, "usage: %s <thread_count> <n>\n", prog_name);  
fprintf(stderr, "   thread_count is the number of threads >= 1\n");  
fprintf(stderr, "   n is the number of terms and should be >= 1\n");
exit(0);
}
