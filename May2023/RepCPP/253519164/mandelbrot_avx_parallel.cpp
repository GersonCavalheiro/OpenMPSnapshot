#include "Mandelbrotters/mandelbrot_avx_parallel.h"

Mandelbrot_AVX_Parallel::Mandelbrot_AVX_Parallel(int width, int height) : MandelbrotCalculator(width, height)
{
temporaryResultsParallelAVX = (double**)malloc(height*sizeof(double*));
for(int i = 0; i < height; i++)
temporaryResultsParallelAVX[i] = (double*)_aligned_malloc(4*sizeof(double), 32);
}

Mandelbrot_AVX_Parallel::Mandelbrot_AVX_Parallel(const MandelbrotCalculator &obj) : MandelbrotCalculator(obj)
{
temporaryResultsParallelAVX = (double**)malloc(height*sizeof(double*));
for(unsigned int i = 0; i < height; i++)
temporaryResultsParallelAVX[i] = (double*)_aligned_malloc(4*sizeof(double), 32);
}

Mandelbrot_AVX_Parallel::~Mandelbrot_AVX_Parallel()
{
for(unsigned int i = 0; i < height; i++)
_aligned_free(temporaryResultsParallelAVX[i]);
free(temporaryResultsParallelAVX);
}


unsigned int* Mandelbrot_AVX_Parallel::calculate(unsigned int numberOfIterations, double upperLeftX, double upperLeftY, double downRightX, double downRightY)
{


double incrementX = (downRightX - upperLeftX) / (double)width;
double incrementY = (upperLeftY - downRightY) / (double)height;
qInfo() << "Width: " << width;
qInfo() << "Height: " << height;

__m256d _upperLeftX,_four, _two, _incrementX;



_upperLeftX = _mm256_set1_pd(upperLeftX);
_four = _mm256_set1_pd(4.0);
_two = _mm256_set1_pd(2.0);
_incrementX = _mm256_set1_pd(incrementX);


unsigned int wholeParts = width / 4; 

#pragma omp parallel for
for(int y = 0; y < height; y++)
{
__m256d divergenceIterations, groupOfFour, imaginary, _secondaryReal, _secondaryImaginary;
__m256d _incrementor = _mm256_set_pd(3, 2, 1, 0); 


double* temporaryResult = temporaryResultsParallelAVX[y];


double imaginaryComponent = upperLeftY - y*incrementY;
imaginary = _mm256_set1_pd(imaginaryComponent);

for(int x = 0; x < wholeParts*4; x += 4)
{
divergenceIterations = _mm256_setzero_pd();
__m256d diverged = _mm256_castsi256_pd(_mm256_set1_epi64x(-1)); 
groupOfFour = _mm256_fmadd_pd(_incrementor, _incrementX, _upperLeftX);
_secondaryImaginary = _mm256_setzero_pd();
_secondaryReal = _mm256_setzero_pd();

for(unsigned int i = 0; i < numberOfIterations; i++)
{
__m256d currentIteration = _mm256_castsi256_pd(_mm256_set1_epi64x((long long)i));
__m256d a2 = _mm256_mul_pd(_secondaryReal, _secondaryReal); 
__m256d b2 = _mm256_mul_pd(_secondaryImaginary, _secondaryImaginary); 


__m256d moduloSquare = _mm256_add_pd(a2, b2); 
__m256d comparisonMask = _mm256_cmp_pd(moduloSquare, _four, _CMP_LE_OQ);
groupOfFour = _mm256_and_pd(groupOfFour, comparisonMask);


divergenceIterations =_mm256_or_pd(divergenceIterations, _mm256_and_pd(currentIteration, _mm256_andnot_pd(comparisonMask, diverged))); 
diverged = _mm256_and_pd(diverged, comparisonMask);

if(_mm256_movemask_pd(diverged) == 0)
break;


__m256d tempReal = _mm256_add_pd(_mm256_sub_pd(a2, b2), groupOfFour); 
_secondaryImaginary = _mm256_fmadd_pd(_mm256_mul_pd(_secondaryReal, _secondaryImaginary), _two, imaginary); 
_secondaryReal = tempReal;
}

_mm256_store_pd(temporaryResult, divergenceIterations);


unsigned int first = *((unsigned int*)(temporaryResult));
unsigned int second = *((unsigned int*)(temporaryResult + 1));
unsigned int third = *((unsigned int*)(temporaryResult + 2));
unsigned int fourth = *((unsigned int*)(temporaryResult + 3));

escapeCounts[y*width + x] = first;
escapeCounts[y*width + x+1] = second;
escapeCounts[y*width + x+2] = third;
escapeCounts[y*width + x+3] = fourth;

_incrementor = _mm256_add_pd(_incrementor, _four);
}

if((wholeParts*4) != width)
{
double realValue = upperLeftX + incrementX*(wholeParts*4);
int counter = 0;
for(unsigned int x = wholeParts*4; x < height; x++)
escapeCounts[y*width + x] = escapeTime(realValue + incrementX*(counter++), imaginaryComponent, numberOfIterations);
}
}

return escapeCounts;
}
