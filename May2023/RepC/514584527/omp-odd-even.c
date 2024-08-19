#include "hpc.h"
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
void cmp_and_swap( int* a, int* b )
{
if ( *a > *b ) {
int tmp = *a;
*a = *b;
*b = tmp;
}
}
void fill( int* v, int n )
{
int i;
int up = n-1, down = 0;
for ( i=0; i<n; i++ ) {
v[i] = ( i % 2 == 0 ? up-- : down++ );
}
}
void odd_even_sort_nopool( int* v, int n )
{
int phase, i;
for (phase = 0; phase < n; phase++) {
if ( phase % 2 == 0 ) {
#pragma omp parallel for default(none) shared(n,v)
for (i=0; i<n-1; i += 2 ) {
cmp_and_swap( &v[i], &v[i+1] );
}
} else {
#pragma omp parallel for default(none) shared(n,v)
for (i=1; i<n-1; i += 2 ) {
cmp_and_swap( &v[i], &v[i+1] );
}
}
}
}
void odd_even_sort_pool( int* v, int n )
{
int phase, i;
#pragma omp parallel default(none) private(phase) shared(n,v)
for (phase = 0; phase < n; phase++) {
if ( phase % 2 == 0 ) {
#pragma omp for
for (i=0; i<n-1; i += 2 ) {
cmp_and_swap( &v[i], &v[i+1] );
}
} else {
#pragma omp for
for (i=1; i<n-1; i += 2 ) {
cmp_and_swap( &v[i], &v[i+1] );
}
}
}
}
void check( int* v, int n )
{
int i;
for (i=0; i<n-1; i++) {
if ( v[i] != i ) {
printf("Check failed: v[%d]=%d, expected %d\n",
i, v[i], i );
abort();
}
}
printf("Check ok!\n");
}
int main( int argc, char* argv[] )
{
int n = 100000;
int *v;
int r;
const int NREPS = 5;
double tstart, tstop;
if ( argc > 1 ) {
n = atoi(argv[1]);
}
v = (int*)malloc(n*sizeof(v[0]));
fill(v,n);
printf("Without thread pool recycling: \t");
tstart = hpc_gettime();
for (r=0; r<NREPS; r++) {        
odd_even_sort_nopool(v,n);
}
tstop = hpc_gettime();
printf("Mean elapsed time %f\n", (tstop - tstart)/NREPS);
check(v,n);
fill(v,n);
printf("With thread pool recycling: \t");
tstart = hpc_gettime();
for (r=0; r<NREPS; r++) {
odd_even_sort_pool(v,n);
}
tstop = hpc_gettime();
printf("Mean elapsed time %f\n", (tstop - tstart)/NREPS);
check(v,n);
free(v);
return EXIT_SUCCESS;
}
