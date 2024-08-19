#include <par-res-kern_general.h>
#include <par-res-kern_omp.h>
#if DOUBLE
#define DTYPE   double
#define EPSILON 1.e-8
#define COEFX   1.0
#define COEFY   1.0
#define FSTR    "%lf"
#else
#define DTYPE   float
#define EPSILON 0.0001f
#define COEFX   1.0f
#define COEFY   1.0f
#define FSTR    "%f"
#endif
#define IN(i,j)       in[i+(j)*(n)]
#define OUT(i,j)      out[i+(j)*(n)]
#define WEIGHT(ii,jj) weight[ii+RADIUS][jj+RADIUS]
int main(int argc, char ** argv) {
long   n;               
int    i, j, ii, jj, it, jt, iter;  
DTYPE  norm,            
reference_norm;
DTYPE  f_active_points; 
DTYPE  flops;           
int    iterations;      
double stencil_time,    
avgtime;
int    stencil_size;    
int    nthread_input,   
nthread;
DTYPE  * RESTRICT in;   
DTYPE  * RESTRICT out;  
long   total_length;    
int    num_error=0;     
DTYPE  weight[2*RADIUS+1][2*RADIUS+1]; 
printf("Parallel Research Kernels version %s\n", PRKVERSION);
printf("OpenMP stencil execution on 2D grid\n");
if (argc != 4){
printf("Usage: %s <# threads> <# iterations> <array dimension>\n",
*argv);
return(EXIT_FAILURE);
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
n  = atol(*++argv);
if (n < 1){
printf("ERROR: grid dimension must be positive: %ld\n", n);
exit(EXIT_FAILURE);
}
if (RADIUS < 1) {
printf("ERROR: Stencil radius %d should be positive\n", RADIUS);
exit(EXIT_FAILURE);
}
if (2*RADIUS +1 > n) {
printf("ERROR: Stencil radius %d exceeds grid size %ld\n", RADIUS, n);
exit(EXIT_FAILURE);
}
total_length = n*n*sizeof(DTYPE);
in  = (DTYPE *) prk_malloc(total_length);
out = (DTYPE *) prk_malloc(total_length);
if (!in || !out) {
printf("ERROR: could not allocate space for input or output array: %ld\n",
total_length);
exit(EXIT_FAILURE);
}
for (jj=-RADIUS; jj<=RADIUS; jj++) for (ii=-RADIUS; ii<=RADIUS; ii++)
WEIGHT(ii,jj) = (DTYPE) 0.0;
#if STAR
stencil_size = 4*RADIUS+1;
for (ii=1; ii<=RADIUS; ii++) {
WEIGHT(0, ii) = WEIGHT( ii,0) =  (DTYPE) (1.0/(2.0*ii*RADIUS));
WEIGHT(0,-ii) = WEIGHT(-ii,0) = -(DTYPE) (1.0/(2.0*ii*RADIUS));
}
#else
stencil_size = (2*RADIUS+1)*(2*RADIUS+1);
for (jj=1; jj<=RADIUS; jj++) {
for (ii=-jj+1; ii<jj; ii++) {
WEIGHT(ii,jj)  =  (DTYPE) (1.0/(4.0*jj*(2.0*jj-1)*RADIUS));
WEIGHT(ii,-jj) = -(DTYPE) (1.0/(4.0*jj*(2.0*jj-1)*RADIUS));
WEIGHT(jj,ii)  =  (DTYPE) (1.0/(4.0*jj*(2.0*jj-1)*RADIUS));
WEIGHT(-jj,ii) = -(DTYPE) (1.0/(4.0*jj*(2.0*jj-1)*RADIUS));
}
WEIGHT(jj,jj)    =  (DTYPE) (1.0/(4.0*jj*RADIUS));
WEIGHT(-jj,-jj)  = -(DTYPE) (1.0/(4.0*jj*RADIUS));
}
#endif
norm = (DTYPE) 0.0;
f_active_points = (DTYPE) (n-2*RADIUS)*(DTYPE) (n-2*RADIUS);
#pragma omp parallel private(i, j, ii, jj, it, jt, iter)
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
printf("Number of threads    = %d\n",nthread_input);
printf("Grid size            = %ld\n", n);
printf("Radius of stencil    = %d\n", RADIUS);
printf("Number of iterations = %d\n", iterations);
#if STAR
printf("Type of stencil      = star\n");
#else
printf("Type of stencil      = compact\n");
#endif
#if DOUBLE
printf("Data type            = double precision\n");
#else
printf("Data type            = single precision\n");
#endif
#if RESTRICT_KEYWORD
printf("No aliasing          = on\n");
#else
printf("No aliasing          = off\n");
#endif
#if LOOPGEN
printf("Script used to expand stencil loop body\n");
#else
printf("Compact representation of stencil loop body\n");
#endif
#if !PARALLELFOR
printf("Parallel regions     = fused (omp for)\n");
#else
printf("Parallel regions     = split (omp parallel for)\n");
#endif
}
}
bail_out(num_error);
#if PARALLELFOR
}
#endif
#if PARALLELFOR
#pragma omp parallel for private(i)
#else
#pragma omp for
#endif
for (j=0; j<n; j++) for (i=0; i<n; i++)
IN(i,j) = COEFX*i+COEFY*j;
#if PARALLELFOR
#pragma omp parallel for private(i)
#else
#pragma omp for
#endif
for (j=RADIUS; j<n-RADIUS; j++) for (i=RADIUS; i<n-RADIUS; i++)
OUT(i,j) = (DTYPE)0.0;
for (iter = 0; iter<=iterations; iter++){
if (iter == 1) {
#if !PARALLELFOR
#pragma omp barrier
#pragma omp master
#endif
{
stencil_time = wtime();
}
}
#if PARALLELFOR
#pragma omp parallel for private(i, ii, jj)
#else
#pragma omp for
#endif
for (j=RADIUS; j<n-RADIUS; j++) {
for (i=RADIUS; i<n-RADIUS; i++) {
#if STAR
#if LOOPGEN
#include "loop_body_star.incl"
#else
for (jj=-RADIUS; jj<=RADIUS; jj++)  OUT(i,j) += WEIGHT(0,jj)*IN(i,j+jj);
for (ii=-RADIUS; ii<0; ii++)        OUT(i,j) += WEIGHT(ii,0)*IN(i+ii,j);
for (ii=1; ii<=RADIUS; ii++)        OUT(i,j) += WEIGHT(ii,0)*IN(i+ii,j);
#endif
#else
#if LOOPGEN
#include "loop_body_compact.incl"
#else
for (jj=-RADIUS; jj<=RADIUS; jj++)
for (ii=-RADIUS; ii<=RADIUS; ii++)  OUT(i,j) += WEIGHT(ii,jj)*IN(i+ii,j+jj);
#endif
#endif
}
}
#if PARALLELFOR
#pragma omp parallel for private(i)
#else
#pragma omp for
#endif
for (j=0; j<n; j++) for (i=0; i<n; i++) IN(i,j)+= 1.0;
} 
#if !PARALLELFOR
#pragma omp barrier
#pragma omp master
#endif
{
stencil_time = wtime() - stencil_time;
}
#if PARALLELFOR
#pragma omp parallel for reduction(+:norm), private (i)
#else
#pragma omp for reduction(+:norm)
#endif
for (j=RADIUS; j<n-RADIUS; j++) for (i=RADIUS; i<n-RADIUS; i++) {
norm += (DTYPE)ABS(OUT(i,j));
}
#if !PARALLELFOR
} 
#endif
norm /= f_active_points;
prk_free(out);
prk_free(in);
reference_norm = (DTYPE) (iterations+1) * (COEFX + COEFY);
if (ABS(norm-reference_norm) > EPSILON) {
printf("ERROR: L1 norm = "FSTR", Reference L1 norm = "FSTR"\n",
norm, reference_norm);
exit(EXIT_FAILURE);
}
else {
printf("Solution validates\n");
#if VERBOSE
printf("Reference L1 norm = "FSTR", L1 norm = "FSTR"\n",
reference_norm, norm);
#endif
}
flops = (DTYPE) (2*stencil_size+1) * f_active_points;
avgtime = stencil_time/iterations;
printf("Rate (MFlops/s): "FSTR"  Avg time (s): %lf\n",
1.0E-06 * flops/avgtime, avgtime);
exit(EXIT_SUCCESS);
}
