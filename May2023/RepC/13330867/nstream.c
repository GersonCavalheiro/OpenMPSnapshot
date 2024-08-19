#include <par-res-kern_general.h>
#include <par-res-kern_omp.h>
#define N   MAXLENGTH
#if STATIC_ALLOCATION
static double a[N];
#else
static double * RESTRICT a;
#endif
static double * RESTRICT b;
static double * RESTRICT c;
#define SCALAR  3.0
static int checkTRIADresults(int, long int);
int main(int argc, char **argv) 
{
long     j, iter;       
double   scalar;        
int      iterations;    
long int length,        
offset;        
double   bytes;         
size_t   space;         
double   nstream_time,  
avgtime;
int      nthread_input; 
int      nthread; 
int      num_error=0;     
printf("Parallel Research Kernels version %s\n", PRKVERSION);
printf("OpenMP stream triad: A = B + scalar*C\n");
if (argc != 5){
printf("Usage:  %s <# threads> <# iterations> <vector length> <offset>\n", *argv);
exit(EXIT_FAILURE);
}
nthread_input = atoi(*++argv);
iterations    = atoi(*++argv);
length        = atol(*++argv);
offset        = atol(*++argv);
if ((nthread_input < 1) || (nthread_input > MAX_THREADS)) {
printf("ERROR: Invalid number of threads: %d\n", nthread_input);
exit(EXIT_FAILURE);
}
if ((iterations < 1)) {
printf("ERROR: Invalid number of iterations: %d\n", iterations);
exit(EXIT_FAILURE);
}
if (length < 0) {
printf("ERROR: Invalid vector length: %ld\n", length);
exit(EXIT_FAILURE);
}
if (offset < 0) {
printf("ERROR: Incvalid array offset: %ld\n", offset);
exit(EXIT_FAILURE);
}
#if STATIC_ALLOCATION 
if ((3*length + 2*offset) > N) {
printf("ERROR: vector length/offset %ld/%ld too ", length, offset);
printf("large; increase MAXLENGTH in Makefile or decrease vector length\n");
exit(EXIT_FAILURE);
}
#endif
omp_set_num_threads(nthread_input);
#if !STATIC_ALLOCATION
space = (3*length + 2*offset)*sizeof(double);
a = (double *) prk_malloc(space);
if (!a) {
printf("ERROR: Could not allocate %ld words for vectors\n", 
3*length+2*offset);
exit(EXIT_FAILURE);
}
#endif
b = a + length + offset;
c = b + length + offset;
#pragma omp parallel private(j,iter) 
{
#pragma omp master
{
nthread = omp_get_num_threads();
if (nthread != nthread_input) {
num_error = 1;
printf("ERROR: number of requested threads %d does not equal ",
nthread_input);
printf("number of spawned threads %d\n", nthread);
} 
else {
printf("Number of threads    = %i\n",nthread_input);
printf("Vector length        = %ld\n", length);
printf("Offset               = %ld\n", offset);
printf("Number of iterations = %d\n", iterations);
#if STATIC_ALLOCATION
printf("Allocation type      = static\n");
#else
printf("Allocation type      = dynamic\n");
#endif
}
}
bail_out(num_error); 
#pragma omp for 
for (j=0; j<length; j++) {
a[j] = 0.0;
b[j] = 2.0;
c[j] = 2.0;
}
scalar = SCALAR;
for (iter=0; iter<=iterations; iter++) {
if (iter==1) {
#pragma omp barrier
#pragma omp master
{
nstream_time = wtime();
}
}
#pragma omp for 
for (j=0; j<length; j++) a[j] += b[j]+scalar*c[j];
} 
#pragma omp barrier
#pragma omp master
{
nstream_time = wtime() - nstream_time;
}
}  
bytes   = 4.0 * sizeof(double) * length;
if (checkTRIADresults(iterations, length)) {
avgtime = nstream_time/iterations;
printf("Rate (MB/s): %lf Avg time (s): %lf\n",
1.0E-06 * bytes/avgtime, avgtime);
}
else exit(EXIT_FAILURE);
return 0;
}
int checkTRIADresults (int iterations, long int length) {
double aj, bj, cj, scalar, asum;
double epsilon = 1.e-8;
int j,iter;
aj = 0.0;
bj = 2.0;
cj = 2.0;
scalar = SCALAR;
for (iter=0; iter<=iterations; iter++) {
aj += bj+scalar*cj;
}
aj = aj * (double) (length);
asum = 0.0;
for (j=0; j<length; j++) asum += a[j];
#if VERBOSE
printf ("Results Comparison: \n");
printf ("        Expected checksum: %f\n",aj);
printf ("        Observed checksum: %f\n",asum);
#endif
if (ABS(aj-asum)/asum > epsilon) {
printf ("Failed Validation on output array\n");
#if VERBOSE
printf ("        Expected checksum: %f \n",aj);
printf ("        Observed checksum: %f \n",asum);
#endif
return (0);
}
else {
printf ("Solution validates\n");
return (1);
}
}
