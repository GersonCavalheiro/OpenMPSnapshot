
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/time.h>
#ifdef WITH_DISPLAY
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/Xos.h>
#include "mandelbrot-gui.h"     
#endif

#define N           2           
#define NPIXELS     800         


typedef struct {
double r, i;
} complex ;


typedef unsigned int uint;
typedef unsigned long ulong;



int main(int argc, char *argv[]) {
uint maxiter;
double r_min = -N;
double r_max = N;
double i_min = -N;
double i_max = N;
uint width = NPIXELS;         
uint height = NPIXELS;
#ifdef WITH_DISPLAY
Display *display;
Window win;
GC gc;
int setup_return;
#endif
ulong min_color, max_color;
double scale_r, scale_i, scale_color;
uint col, row, k;
complex z, c;
double lengthsq, temp;


if ((argc < 2) || ((argc > 2) && (argc < 5))) {
fprintf(stderr, "usage:  %s maxiter [x0 y0 size]\n", argv[0]);
return EXIT_FAILURE;
}


maxiter = atoi(argv[1]);
if (argc > 2) {
double x0 = atof(argv[2]);
double y0 = atof(argv[3]);
double size = atof(argv[4]);
r_min = x0 - size;
r_max = x0 + size;
i_min = y0 - size;
i_max = y0 + size;
}

#ifdef WITH_DISPLAY

setup_return =
setup(width, height, &display, &win, &gc, &min_color, &max_color);
if (setup_return != EXIT_SUCCESS) {
fprintf(stderr, "Unable to initialize display, continuing\n");
abort();
}

#else
min_color=0;
max_color=16777215;
#endif

struct timeval startingTime, endingTime;
gettimeofday(&startingTime, NULL);




scale_r = (double) (r_max - r_min) / (double) width;
scale_i = (double) (i_max - i_min) / (double) height;


scale_color = (double) (max_color - min_color) / (double) (maxiter - 1);


#pragma omp parallel for private(col,c, k, temp, z, lengthsq) schedule(dynamic)
for (row = 0; row < height; ++row) {
ulong couleur[width];

for (col = 0; col < width; ++col) {
z.r = z.i = 0;


c.r = r_min + ((double) col * scale_r);
c.i = i_min + ((double) (height-1-row) * scale_i);



k = 0;
do  {
temp = z.r*z.r - z.i*z.i + c.r;
z.i = 2*z.r*z.i + c.i;
z.r = temp;
lengthsq = z.r*z.r + z.i*z.i;
++k;
} while (lengthsq < (N*N) && k < maxiter);


couleur[col] = (ulong) ((k-1) * scale_color) + min_color;
}

#ifdef WITH_DISPLAY
#pragma omp critical
for(col = 0; col<width; col++) {
XSetForeground (display, gc, couleur[col]);
XDrawPoint (display, win, gc, col, row);
XFlush(display);
}
#endif
}

#ifdef WITH_DISPLAY

XFlush (display);
#endif

gettimeofday(&endingTime, NULL);
double workingTime = ((endingTime.tv_sec-startingTime.tv_sec)*1e6 + (endingTime.tv_usec-startingTime.tv_usec))/1e6;



fprintf(stdout, "\n");
fprintf(stdout, "center = (%g, %g), size = %g\n",
(r_max + r_min)/2, (i_max + i_min)/2,
(r_max - r_min)/2);
fprintf(stdout, "maximum iterations = %d\n", maxiter);
printf("working time: %g s\n",workingTime);
fprintf(stdout, "\n");

#ifdef WITH_DISPLAY

interact(display, &win, width, height,
r_min, r_max, i_min, i_max);
#endif
return EXIT_SUCCESS;
}

