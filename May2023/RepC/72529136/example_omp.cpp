#include <complex>
#include <cstdio>
#include <cstdlib>
#include <omp.h>
typedef std::complex<double> complex;
int MandelbrotCalculate(complex c, int maxiter)
{
complex z = c;
int n=0;
for(; n<maxiter; ++n)
{
if( std::abs(z) >= 2.0) break;
z = z*z + c;
}
return n;
}
int main(int argc, char **argv)
{
const int thread_cnt = atoi(argv[1]);
const int width = 78, height = 44, num_pixels = width*height;
int mandel_vals[height][width];
const complex center(-.7, 0), span(2.7, -(4/3.0)*2.7*height/width);
const complex begin = center-span/2.0;
const int maxiter = 100000;
double start, finish;
start = omp_get_wtime();
#pragma omp parallel for schedule(runtime) num_threads(thread_cnt)
for(int pix=0; pix<num_pixels; ++pix)
{
const int x = pix%width, y = pix/width;
complex c = begin + complex(x * span.real() / (width +1.0),
y * span.imag() / (height+1.0));
int n = MandelbrotCalculate(c, maxiter);
if(n == maxiter) n = 0;
mandel_vals[y][x] = n;
}
finish = omp_get_wtime();
printf("Elapsed Time : %.4f(s)\n", finish - start);
}