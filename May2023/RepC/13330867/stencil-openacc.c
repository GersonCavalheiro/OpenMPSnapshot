#include <openacc.h>
#include "prk_util.h"
typedef void (*stencil_t)(const int, const double * restrict, double * restrict);
void nothing(const int n, const double * restrict in, double * restrict out)
{
printf("You are trying to use a stencil that does not exist.\n");
printf("Please generate the new stencil using the code generator.\n");
if (n==0) printf("%p %p\n", in, out);
abort();
}
#include "stencil_openacc.h"
int main(int argc, char * argv[])
{
printf("Parallel Research Kernels version %d\n", PRKVERSION);
printf("C11/OpenACC Stencil execution on 2D grid\n");
if (argc < 3){
printf("Usage: <# iterations> <array dimension> [<star/grid> <radius>]\n");
return 1;
}
int iterations = atoi(argv[1]);
if (iterations < 1) {
printf("ERROR: iterations must be >= 1\n");
return 1;
}
int n  = atoi(argv[2]);
if (n < 1) {
printf("ERROR: grid dimension must be positive\n");
return 1;
} else if (n > floor(sqrt(INT_MAX))) {
printf("ERROR: grid dimension too large - overflow risk\n");
return 1;
}
bool star = true;
if (argc > 3) {
char* pattern = argv[3];
star = (0==strncmp(pattern,"star",4)) ? true : false;
}
int radius = 2;
if (argc > 4) {
radius = atoi(argv[4]);
}
if ( (radius < 1) || (2*radius+1 > n) ) {
printf("ERROR: Stencil radius negative or too large\n");
return 1;
}
printf("Number of iterations      = %d\n", iterations);
printf("Grid sizes                = %d\n", n);
printf("Type of stencil           = %s\n", (star ? "star" : "grid") );
printf("Radius of stencil         = %d\n", radius );
stencil_t stencil = nothing;
if (star) {
switch (radius) {
case 1: stencil = star1; break;
case 2: stencil = star2; break;
case 3: stencil = star3; break;
case 4: stencil = star4; break;
case 5: stencil = star5; break;
case 6: stencil = star6; break;
case 7: stencil = star7; break;
case 8: stencil = star8; break;
case 9: stencil = star9; break;
}
} else {
switch (radius) {
case 1: stencil = grid1; break;
case 2: stencil = grid2; break;
case 3: stencil = grid3; break;
case 4: stencil = grid4; break;
case 5: stencil = grid5; break;
case 6: stencil = grid6; break;
case 7: stencil = grid7; break;
case 8: stencil = grid8; break;
case 9: stencil = grid9; break;
}
}
double stencil_time = 0.0;
size_t bytes = n*n*sizeof(double);
double * restrict in  = acc_malloc(bytes);
double * restrict out = acc_malloc(bytes);
{
#pragma acc parallel loop collapse(2) deviceptr(in,out)
for (int i=0; i<n; i++) {
for (int j=0; j<n; j++) {
in[i*n+j]  = (double)(i+j);
out[i*n+j] = 0.0;
}
}
for (int iter = 0; iter<=iterations; iter++) {
if (iter==1) stencil_time = prk_wtime();
stencil(n, in, out);
#pragma acc parallel loop collapse(2) deviceptr(in,out)
for (int i=0; i<n; i++) {
for (int j=0; j<n; j++) {
in[i*n+j] += 1.0;
}
}
}
stencil_time = prk_wtime() - stencil_time;
}
size_t active_points = (n-2*radius)*(n-2*radius);
double norm = 0.0;
#pragma acc parallel loop reduction( +:norm ) deviceptr(out)
for (int i=radius; i<n-radius; i++) {
for (int j=radius; j<n-radius; j++) {
norm += fabs(out[i*n+j]);
}
}
norm /= active_points;
acc_free(in);
acc_free(out);
const double epsilon = 1.0e-8;
double reference_norm = 2.*(iterations+1.);
if (fabs(norm-reference_norm) > epsilon) {
printf("ERROR: L1 norm = %lf Reference L1 norm = %lf\n", norm, reference_norm);
return 1;
} else {
printf("Solution validates\n");
#ifdef VERBOSE
printf("L1 norm = %lf Reference L1 norm = %lf\n", norm, reference_norm);
#endif
const int stencil_size = star ? 4*radius+1 : (2*radius+1)*(2*radius+1);
size_t flops = (2*stencil_size+1) * active_points;
double avgtime = stencil_time/iterations;
printf("Rate (MFlops/s): %lf Avg time (s): %lf\n", 1.0e-6 * (double)flops/avgtime, avgtime );
}
return 0;
}
