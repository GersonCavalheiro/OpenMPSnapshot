#include "hpc.h"
#include <stdio.h>
#include <stdlib.h>
const int MAXIT = 500000; 
const int XSIZE = 78, YSIZE = 62;
const double XMIN = -2.5, XMAX = 1.0;
const double YMIN = -1.0, YMAX = 1.0;
int iterate( float cx, float cy )
{
float x = 0.0, y = 0.0, xnew, ynew;
int it;
for ( it = 0; (it < MAXIT) && (x*x + y*y < 2*2); it++ ) {
xnew = x*x - y*y + cx;
ynew = 2.0f*x*y + cy;
x = xnew;
y = ynew;
}
return it;
}
int main( int argc, char *argv[] )
{
int x, y;
float tstart, elapsed;
const char charset[] = ".,c8M@jawrpogOQEPGJ";
tstart = hpc_gettime();
#if __GNUC__ < 9
#pragma omp parallel for default(none) collapse(2) ordered
#else
#pragma omp parallel for default(none) collapse(2) shared(XSIZE,YSIZE,XMIN,XMAX,YMIN,YMAX,charset,MAXIT) ordered
#endif
for ( y = 0; y < YSIZE; y++ ) {
for ( x = 0; x < XSIZE; x++ ) {
const double cx = XMIN + (XMAX - XMIN) * (float)(x) / (XSIZE - 1);
const double cy = YMAX - (YMAX - YMIN) * (float)(y) / (YSIZE - 1);
const int v = iterate(cx, cy);
#pragma omp ordered 
{
char c = ' ';
if (v < MAXIT) {
c = charset[v % (sizeof(charset)-1)];
}
putchar(c);
if (x+1 == XSIZE) puts("|");
}            
}
}
elapsed = hpc_gettime() - tstart;
printf("Elapsed time %f\n", elapsed);
return EXIT_SUCCESS;
}
