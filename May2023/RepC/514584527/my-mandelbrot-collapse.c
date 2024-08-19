#include "hpc.h"
#include <stdio.h>
#include <stdlib.h>
const int MAXIT = 10000;
void fill( int* M, int m, int n )
{
int i, j;
for ( i = 0; i < m; i++ ) {
for ( j = 0; j < n; j++ ) {
M[i*m + j] = (int)(0);
}
}
}
int iterate( float cx, float cy )
{
float x = 0.0f, y = 0.0f, xnew, ynew;
int it;
for ( it = 0; (it < MAXIT) && (x*x + y*y <= 2.0*2.0); it++ ) {
xnew = x*x - y*y + cx;
ynew = 2.0*x*y + cy;
x = xnew;
y = ynew;
}
return it;
}
int main( int argc, char *argv[] )
{
int x, y;
int *matrix;
int x_size = 1024, y_size = 768;
if ( argc > 3 ) {
printf("Usage: %s [x_size y_size]\n", argv[0]);
return EXIT_FAILURE;
}
if ( argc == 3 ) {
x_size = atoi(argv[1]);
y_size = atoi(argv[2]);
}
const double x_min = -2.5, x_max = 1.5;
const double y_min = -1.5, y_max = 1.5;
matrix = (int*) malloc( y_size * x_size * sizeof(int) );
fill(matrix, y_size, x_size);
const double tstart = hpc_gettime(); 
#if __GNUC__ < 9
#pragma omp parallel for collapse(2) schedule(runtime)
#else
#pragma omp parallel for collapse(2) shared(x_size,x_min,x_max,y_size,y_min,y_max,matrix,MAXIT) schedule(runtime)
#endif
for ( y = 0; y < y_size; y++ ) {
for ( x = 0; x < x_size; x++ ) {
const double re = x_min + (x_max - x_min) * (float)(x) / (x_size - 1);
const double im = y_max - (y_max - y_min) * (float)(y) / (y_size - 1);
const int it = iterate(re, im);
#pragma omp critical
if ( it < MAXIT ) {
matrix[y*y_size + x] = it;
}
}
}
const double elapsed = hpc_gettime() - tstart;
printf ("Elapsed time: %f\n", elapsed);
char filepath[256];
snprintf ( filepath, sizeof(filepath), "./data/mandelbrot/matrix_%dx%d.csv", x_size, y_size );
FILE *fpt = fopen ( filepath, "w" );
for ( y = 0; y < y_size; y++ ) {
for ( x = 0; x < x_size; x++ ) {
if ( x == x_size - 1 ) {
fprintf ( fpt, "%d\n", matrix[y*y_size + x] );
} else {
fprintf ( fpt, "%d,", matrix[y*y_size + x] );
}
}
}
fclose(fpt);
return EXIT_SUCCESS;
}
