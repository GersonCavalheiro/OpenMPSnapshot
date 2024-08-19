#include <omp.h>
#define N 2     
typedef struct {
double real, imag;
} complex;
#if 1
void mandelbrot1(int height,
int width,
double real_min,
double imag_min,
double scale_real,
double scale_imag,
int maxiter, 
int ** output)
{
int col;
for (int row = 0; row < height; ++row)
{
#pragma analysis_check assert correctness_race() correctness_incoherent_fp(col)
#pragma omp task
for (col = 0; col < width; ++col) 
{
complex z, c;
z.real = z.imag = 0;
c.real = real_min + ((double) col * scale_real);
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
}
}
}
#endif
int row2, col2; 
#if 1
void mandelbrot2(int height,
int width,
double real_min,
double imag_min,
double scale_real,
double scale_imag,
int maxiter,
int ** output)
{
complex z, c;
for (row2 = 0; row2 < height; ++row2) {
#pragma analysis_check assert correctness_race() correctness_incoherent_fp(col2, z, c)
#pragma omp task firstprivate(row2) firstprivate(col2)
for (col2 = 0; col2 < width; ++col2) {
z.real = z.imag = 0;
c.real = real_min + ((double) col2 * scale_real);
c.imag = imag_min + ((double) (height-1-row2) * scale_imag);
int k = 0;
double lengthsq, temp;
do  {
temp = z.real*z.real - z.imag*z.imag + c.real;
z.imag = 2*z.real*z.imag + c.imag;
z.real = temp;
lengthsq = z.real*z.real + z.imag*z.imag;
++k;
} while (lengthsq < (N*N) && k < maxiter);
output[row2][col2]=k;
}
}
}
#endif
#if 1
void mandelbrot3(int height,
int width,
double real_min,
double imag_min,
double scale_real,
double scale_imag,
int maxiter,
int ** output)
{
complex z, c;
for (row2 = 0; row2 < height; ++row2) {
#pragma analysis_check assert correctness_race(row2, col2) correctness_incoherent_fp(z, c)
#pragma omp task
for (col2 = 0; col2 < width; ++col2) {
z.real = z.imag = 0;
c.real = real_min + ((double) col2 * scale_real);
c.imag = imag_min + ((double) (height-1-row2) * scale_imag);
int k = 0;
double lengthsq, temp;
do  {
temp = z.real*z.real - z.imag*z.imag + c.real;
z.imag = 2*z.real*z.imag + c.imag;
z.real = temp;
lengthsq = z.real*z.real + z.imag*z.imag;
++k;
} while (lengthsq < (N*N) && k < maxiter);
output[row2][col2]=k;
}
}
}
#endif
#if 1
void mandelbrot4(int height,
int width,
double real_min,
double imag_min,
double scale_real,
double scale_imag,
int maxiter,
int ** output)
{
complex z, c;
#pragma omp parallel private(row2)
for (row2 = 0; row2 < height; ++row2) {
#pragma analysis_check assert correctness_race(row2, col2, z, c)
#pragma omp task
for (col2 = 0; col2 < width; ++col2) {
z.real = z.imag = 0;
c.real = real_min + ((double) col2 * scale_real);
c.imag = imag_min + ((double) (height-1-row2) * scale_imag);
int k = 0;
double lengthsq, temp;
do  {
temp = z.real*z.real - z.imag*z.imag + c.real;
z.imag = 2*z.real*z.imag + c.imag;
z.real = temp;
lengthsq = z.real*z.real + z.imag*z.imag;
++k;
} while (lengthsq < (N*N) && k < maxiter);
output[row2][col2]=k;
}
}
}
#endif
