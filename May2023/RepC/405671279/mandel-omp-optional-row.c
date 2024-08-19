#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <malloc.h>
#include "omp.h"
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/Xos.h>
#include <sys/time.h>
double getusec_() {
struct timeval time;
gettimeofday(&time, NULL);
return ((double)time.tv_sec * (double)1e6 + (double)time.tv_usec);
}
double stamp;
#define START_COUNT_TIME stamp = getusec_();
#define STOP_COUNT_TIME(_m) stamp = getusec_() - stamp;\
stamp = stamp/1e6;\
printf ("%s %0.6f\n",(_m), stamp);
#define N           2           
#define NPIXELS     800         
typedef struct {
double real, imag;
} complex;
#include "mandelbrot-gui.h"     
int output2file = 0;
FILE *fp = NULL;
int output2display = 0;
Display *display;
Window win;
GC gc;
int setup_return;
long min_color = 0, max_color = 0;
double scale_color;
double scale_real, scale_imag;
int output2histogram = 0;
int * histogram;
int user_param = 1;
void mandelbrot(int height, int width, double real_min, double imag_min,
double scale_real, double scale_imag, int maxiter, int **output) {
#pragma omp parallel
#pragma omp single
#pragma omp taskloop num_tasks(user_param)
for (int row = 0; row < height; ++row) {
for (int col = 0; col < width; ++col) {
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
output[row][col]=k;
if (output2histogram)
{
#pragma omp atomic
histogram[k-1]++;
}
if (output2display) {
long color = (long) ((k-1) * scale_color) + min_color;
{
if (setup_return == EXIT_SUCCESS) {
#pragma omp critical
{
XSetForeground (display, gc, color);
XDrawPoint (display, win, gc, col, row);
}
}	
}
}
}
}
int main(int argc, char *argv[]) {
int maxiter = 1000;
double real_min;
double real_max;
double imag_min;
double imag_max;
int width  = NPIXELS;         
int height = NPIXELS;
double size = N, x0 = 0, y0 = 0;
int ** output;
char filename[32];
#pragma omp parallel
;
for (int i=1; i<argc; i++) {
if (strcmp(argv[i], "-d")==0) {
output2display = 1;
}
else if (strcmp(argv[i], "-h")==0) {
output2histogram = 1;
}
else if (strcmp(argv[i], "-i")==0) {
maxiter = atoi(argv[++i]);
}
else if (strcmp(argv[i], "-w")==0) {
width = atoi(argv[++i]);
height = width;
}
else if (strcmp(argv[i], "-c")==0) {
x0 = atof(argv[++i]);
y0 = atof(argv[++i]);
}
else if (strcmp(argv[i], "-u")==0) {
user_param = atof(argv[++i]);
}
else if (strcmp(argv[i], "-s")==0) {
size = atof(argv[++i]);
}
else if (strcmp(argv[i], "-o")==0) {
output2file = 1;
sprintf(filename, "output_omp_%d.out", omp_get_max_threads());
if((fp=fopen(filename, "wb"))==NULL) {
fprintf(stderr, "Unable to open file\n");
return EXIT_FAILURE;
}
}
else {
fprintf(stderr, "Usage: %s [-o -h -d -i maxiter -w windowsize -c x0 y0 -s size]\n", argv[0]);
fprintf(stderr, "       -o to write computed image and histogram to disk (default no file generated)\n");
fprintf(stderr, "       -h to produce histogram of values in computed image (default no histogream)\n");
fprintf(stderr, "       -d to display computed image (default no display)\n");
fprintf(stderr, "       -i to specify maximum number of iterations at each point (default 1000)\n");
fprintf(stderr, "       -w to specify the size of the image to compute (default 800x800 elements)\n");
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
fprintf(stdout, "Computation of the Mandelbrot set with:\n");
fprintf(stdout, "    center = (%g, %g) \n    size = %g\n    maximum iterations = %d\n",
(real_max + real_min)/2, (imag_max + imag_min)/2,
(real_max - real_min)/2,
maxiter);
fprintf(stdout, "\n");
output = malloc(height*sizeof(int *));
for (int row = 0; row < height; ++row)
output[row] = malloc(width*sizeof(int));
if (output2histogram) histogram = calloc(maxiter, sizeof(int));
if (output2display) {
setup_return =
setup(width, height, &display, &win, &gc, &min_color, &max_color);
if (setup_return != EXIT_SUCCESS) {
fprintf(stderr, "Unable to initialize display, continuing\n");
return EXIT_FAILURE;
}
}
scale_real = (double) (real_max - real_min) / (double) width;
scale_imag = (double) (imag_max - imag_min) / (double) height;
if (output2display) {
scale_color = (double) (max_color - min_color) / (double) (maxiter - 1);
}
if (!output2display) {
START_COUNT_TIME;
}
mandelbrot(height, width, real_min, imag_min, scale_real, scale_imag, maxiter, output);
if (!output2display) {
STOP_COUNT_TIME("Total execution time (in seconds):");
fprintf(stdout, "\n");
}
fprintf(stdout, "Mandelbrot set: Computed\n");
if (output2histogram) fprintf(stdout, "Histogram for Mandelbrot set: Computed\n");
else fprintf(stdout, "Histogram for Mandelbrot set: Not computed\n");
if ((output2display) && (setup_return == EXIT_SUCCESS)) XFlush (display);
if ((output2file) && (fp != NULL)) {
fprintf(stdout, "Writing output file to disk: %s\n", filename);
for (int row = 0; row < height; ++row)
if(fwrite(output[row], sizeof(int), width, fp) != width)
fprintf(stderr, "Error when writing output to file\n");
if (output2histogram)
if(fwrite(histogram, sizeof(int), maxiter, fp) != maxiter)
fprintf(stderr, "Error when writing histogram to file\n");
}
if (output2display) {
if (setup_return == EXIT_SUCCESS) {
interact(display, &win, width, height,
real_min, real_max, imag_min, imag_max);
}
return EXIT_SUCCESS;
}
#pragma omp parallel
;
}