#include <par-res-kern_general.h>
#include <par-res-kern_omp.h>
#define A(i,j)    A[i+order*(j)]
#define B(i,j)    B[i+order*(j)]
static double test_results (size_t , double*, int);
int main(int argc, char ** argv) {
size_t order;         
size_t i, j, it, jt;  
int    Tile_order=32; 
int    iterations;    
int    iter;          
int    tiling;        
double bytes;         
double * RESTRICT A;  
double * RESTRICT B;  
double abserr;        
double epsilon=1.e-8; 
double transpose_time,
avgtime;
int    nthread_input, 
nthread;
int    num_error=0;     
printf("Parallel Research Kernels version %s\n", PRKVERSION);
printf("OpenMP Matrix transpose: B = A^T\n");
if (argc != 4 && argc != 5){
printf("Usage: %s <# threads> <# iterations> <matrix order> [tile size]\n",
*argv);
exit(EXIT_FAILURE);
}
nthread_input = atoi(*++argv); 
if ((nthread_input < 1) || (nthread_input > MAX_THREADS)) {
printf("ERROR: Invalid number of threads: %d\n", nthread_input);
exit(EXIT_FAILURE);
}
omp_set_num_threads(nthread_input);
iterations  = atoi(*++argv); 
if (iterations < 1){
printf("ERROR: iterations must be >= 1 : %d \n",iterations);
exit(EXIT_FAILURE);
}
order = atoi(*++argv); 
if (order <= 0){
printf("ERROR: Matrix Order must be greater than 0 : %zu \n", order);
exit(EXIT_FAILURE);
}
if (argc == 5) Tile_order = atoi(*++argv);
tiling = (Tile_order > 0) && ((size_t)Tile_order < order);
if (!tiling) Tile_order = order;
A   = (double *)prk_malloc(order*order*sizeof(double));
if (A == NULL){
printf(" ERROR: cannot allocate space for input matrix: %ld\n", 
order*order*sizeof(double));
exit(EXIT_FAILURE);
}
B  = (double *)prk_malloc(order*order*sizeof(double));
if (B == NULL){
printf(" ERROR: cannot allocate space for output matrix: %ld\n", 
order*order*sizeof(double));
exit(EXIT_FAILURE);
}
bytes = 2.0 * sizeof(double) * order * order;
#pragma omp parallel private (i, j, it, jt, iter)
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
printf("Number of threads     = %i;\n",nthread_input);
printf("Matrix order          = %ld\n", order);
printf("Number of iterations  = %d\n", iterations);
if (tiling) {
printf("Tile size             = %d\n", Tile_order);
#if COLLAPSE
printf("Loop collapse         = on\n");
#else
printf("Loop collapse         = off\n");
#endif
}
else                   
printf("Untiled\n");
}
}
bail_out(num_error);
if (tiling) {
#if COLLAPSE
#pragma omp for collapse(2)
#else
#pragma omp for
#endif
for (j=0; j<order; j+=Tile_order) 
for (i=0; i<order; i+=Tile_order) 
for (jt=j; jt<MIN(order,j+Tile_order);jt++)
for (it=i; it<MIN(order,i+Tile_order); it++){
A(it,jt) = (double) (order*jt + it);
B(it,jt) = 0.0;
}
}
else {
#pragma omp for
for (j=0;j<order;j++) 
for (i=0;i<order; i++) {
A(i,j) = (double) (order*j + i);
B(i,j) = 0.0;
}
}
for (iter = 0; iter<=iterations; iter++){
if (iter == 1) { 
#pragma omp barrier
#pragma omp master
{
transpose_time = wtime();
}
}
if (!tiling) {
#pragma omp for 
for (i=0;i<order; i++) 
for (j=0;j<order;j++) { 
B(j,i) += A(i,j);
A(i,j) += 1.0;
}
}
else {
#if COLLAPSE
#pragma omp for collapse(2)
#else
#pragma omp for
#endif
for (i=0; i<order; i+=Tile_order) 
for (j=0; j<order; j+=Tile_order) 
for (it=i; it<MIN(order,i+Tile_order); it++) 
for (jt=j; jt<MIN(order,j+Tile_order);jt++) {
B(jt,it) += A(it,jt);
A(it,jt) += 1.0;
} 
}	
}  
#pragma omp barrier
#pragma omp master
{
transpose_time = wtime() - transpose_time;
}
} 
abserr =  test_results (order, B, iterations);
prk_free(B);
prk_free(A);
if (abserr < epsilon) {
printf("Solution validates\n");
avgtime = transpose_time/iterations;
printf("Rate (MB/s): %lf Avg time (s): %lf\n",
1.0E-06 * bytes/avgtime, avgtime);
#if VERBOSE
printf("Squared errors: %f \n", abserr);
#endif
exit(EXIT_SUCCESS);
}
else {
printf("ERROR: Aggregate squared error %lf exceeds threshold %e\n",
abserr, epsilon);
exit(EXIT_FAILURE);
}
}  
double test_results (size_t order, double *B, int iterations) {
double abserr=0.0;
size_t i, j;
double addit = ((double)(iterations+1) * (double) (iterations))/2.0;
#pragma omp parallel for reduction(+:abserr)
for (j=0;j<order;j++) {
for (i=0;i<order; i++) {
abserr += ABS(B(i,j) - ((i*order + j)*(iterations+1L)+addit));
}
}
#if VERBOSE
#pragma omp master 
{
printf(" Squared sum of differences: %f\n",abserr);
}
#endif   
return abserr;
}
