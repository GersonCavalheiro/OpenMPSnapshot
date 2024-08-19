#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <malloc.h>
#if _DISPLAY_
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/Xos.h>
#endif
#include <sys/time.h>
double getusec_() {
struct timeval time;
gettimeofday(&time, NULL);
return ((double)time.tv_sec * (double)1e6 + (double)time.tv_usec);
}
#define START_COUNT_TIME stamp = getusec_();
#define STOP_COUNT_TIME(_m) stamp = getusec_() - stamp;\
stamp = stamp/1e6;\
printf ("%s: %0.6fs\n",(_m), stamp);
#define N           2           
#define NPIXELS     800         
int row, col; 
typedef struct {
double real, imag;
} complex;
#if _DISPLAY_
#include "mandelbrot-gui.h"     
#endif
void mandelbrot(int height, 
int width, 
double real_min, 
double imag_min,
double scale_real, 
double scale_imag, 
int maxiter, 
#if _DISPLAY_
int setup_return,
Display *display, 
Window win, 
GC gc, 
double scale_color,
double min_color)
#else
int ** output) 
#endif
{
#pragma omp parallel private(row,col)
for (row = 0; row < height; ++row) {
#pragma omp for schedule(runtime)
for (col = 0; col < width; ++col) {
complex z, c;
z.real = z.imag = 0;
c.real = real_min + ((double) col * scale_real);
c.imag = imag_min + ((double) (height-1-row) * scale_imag);
int k = 0;
double lengthsq, temp;
do  {
temp = z.real*z.real - z.imag*z.imag + c.real;
z.imag = 2*z.real*z.imag + c.imag;
z.real = temp;
lengthsq = z.real*z.real + z.imag*z.imag;
++k;
} while (lengthsq < (N*N) && k < maxiter);
#if _DISPLAY_
long color = (long) ((k-1) * scale_color) + min_color;
if (setup_return == EXIT_SUCCESS) {
#pragma omp critical
{
XSetForeground (display, gc, color);
XDrawPoint (display, win, gc, col, row);
}
}
#else
output[row][col]=k;
#endif
}
}
}
int main(int argc, char *argv[]) {
int maxiter = 1000;
double real_min;
double real_max;
double imag_min;
double imag_max;
int width = NPIXELS;         
int height = NPIXELS;
double size=N, x0 = 0, y0 = 0;
#if _DISPLAY_
Display *display;
Window win;
GC gc;
int setup_return;
long min_color = 0, max_color = 0;
double scale_color;
#else 
int ** output;
FILE *fp = NULL;
#endif
double scale_real, scale_imag;
for (int i=1; i<argc; i++) {
if (strcmp(argv[i], "-i")==0) {
maxiter = atoi(argv[++i]);
}
else if (strcmp(argv[i], "-w")==0) {
width = atoi(argv[++i]);
height = width;
}
else if (strcmp(argv[i], "-s")==0) {
size = atof(argv[++i]);
}
#if !_DISPLAY_
else if (strcmp(argv[i], "-o")==0) {
if((fp=fopen("mandel.out", "wb"))==NULL) { 
fprintf(stderr, "Unable to open file\n"); 
return EXIT_FAILURE; 
}
}
#endif
else if (strcmp(argv[i], "-c")==0) {
x0 = atof(argv[++i]); 
y0 = atof(argv[++i]);
}
else {
#if _DISPLAY_
fprintf(stderr, "Usage: %s [-i maxiter -w windowsize -c x0 y0 -s size]\n", argv[0]);
#else
fprintf(stderr, "Usage: %s [-o -i maxiter -w windowsize -c x0 y0 -s size]\n", argv[0]);
fprintf(stderr, "       -o to write computed image to disk (default no file generated)\n");
#endif
fprintf(stderr, "       -i to specify maximum number of iterations at each point (default 1000)\n");
#if _DISPLAY_
fprintf(stderr, "       -w to specify the size of the display window (default 800x800 pixels)\n");
#else
fprintf(stderr, "       -w to specify the size of the image to compute (default 800x800 elements)\n");
#endif
fprintf(stderr, "       -c to specify the center x0+iy0 of the square to compute (default origin)\n");
fprintf(stderr, "       -s to specify the size of the square to compute (default 2, i.e. size 4 by 4)\n");
return EXIT_FAILURE;
}
}
real_min = x0 - size;
real_max = x0 + size;
imag_min = y0 - size;
imag_max = y0 + size;
fprintf(stdout, "\n");
fprintf(stdout, "Mandelbrot program\n");
fprintf(stdout, "center = (%g, %g), size = %g\n",
(real_max + real_min)/2, (imag_max + imag_min)/2,
(real_max - real_min)/2);
fprintf(stdout, "maximum iterations = %d\n", maxiter);
fprintf(stdout, "\n");
#if _DISPLAY_
setup_return = 
setup(width, height, &display, &win, &gc, &min_color, &max_color);
if (setup_return != EXIT_SUCCESS) {
fprintf(stderr, "Unable to initialize display, continuing\n");
return EXIT_FAILURE;
}
#else
output = malloc(height*sizeof(int *));
for (int row = 0; row < height; ++row)
output[row] = malloc(width*sizeof(int));
#endif
scale_real = (double) (real_max - real_min) / (double) width;
scale_imag = (double) (imag_max - imag_min) / (double) height;
#if _DISPLAY_
scale_color = (double) (max_color - min_color) / (double) (maxiter - 1);
#endif
double stamp;
START_COUNT_TIME;
#if _DISPLAY_
mandelbrot(height,width,real_min, imag_min, scale_real, scale_imag, maxiter, 
setup_return, display, win, gc, scale_color, min_color); 
#else
mandelbrot(height,width,real_min, imag_min, scale_real, scale_imag, maxiter, 
output); 
#endif
STOP_COUNT_TIME("Total execution time");
#if _DISPLAY_
if (setup_return == EXIT_SUCCESS) {
XFlush (display);
}
#else
if (fp != NULL)
{
for (int row = 0; row < height; ++row)
if(fwrite(output[row], sizeof(int), width, fp) != width) { 
fprintf(stderr, "Output file not written correctly\n"); 
}
}
#endif
#if _DISPLAY_
if (setup_return == EXIT_SUCCESS) {
interact(display, &win, width, height,
real_min, real_max, imag_min, imag_max);
}
return EXIT_SUCCESS;
#endif
}
