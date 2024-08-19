#include "hpc.h"
#include <stdio.h>
#include <stdlib.h>
#include "gfx.h"
const int MAXIT = 10000;
const int XSIZE = 1024, YSIZE = 768;
const double XMIN = -2.5, XMAX = 1.0;
const double YMIN = -1.0, YMAX = 1.0;
typedef struct {
int r, g, b;
} pixel_t;
const pixel_t colors[] = {
{66, 30, 15}, 
{25, 7, 26},
{9, 1, 47},
{4, 4, 73},
{0, 7, 100},
{12, 44, 138},    
{24, 82, 177},
{57, 125, 209},
{134, 181, 229},
{211, 236, 248},
{241, 233, 191},
{248, 201, 95},
{255, 170, 0},
{204, 128, 0},
{153, 87, 0},
{106, 52, 3} };
const int NCOLORS = sizeof(colors)/sizeof(colors[0]);
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
void drawpixel( int x, int y, int it )
{
if (it < MAXIT) {
gfx_color( colors[it % NCOLORS].r,
colors[it % NCOLORS].g,
colors[it % NCOLORS].b );            
} else {
gfx_color( 0, 0, 0 );                        
}
gfx_point( x, y );
}
int main( int argc, char *argv[] )
{
int x, y;
gfx_open(XSIZE, YSIZE, "Mandelbrot Set");
const double tstart = hpc_gettime();
#if __GNUC__ < 9
#pragma omp parallel for default(none) private(x) schedule(runtime)
#else
#pragma omp parallel for default(none) shared(XSIZE,YSIZE,XMIN,XMAX,YMIN,YMAX) private(x) schedule(runtime)
#endif
for ( y = 0; y < YSIZE; y++ ) {
for ( x = 0; x < XSIZE; x++ ) {
const double re = XMIN + (XMAX - XMIN) * (float)(x) / (XSIZE - 1);
const double im = YMAX - (YMAX - YMIN) * (float)(y) / (YSIZE - 1);
const int v = iterate(re, im);
#pragma omp critical
drawpixel( x, y, v);
}
}
const double elapsed = hpc_gettime() - tstart;
printf("Elapsed time %f\n", elapsed);
printf("Click to finish\n");
gfx_wait();
return EXIT_SUCCESS;
}
