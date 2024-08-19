#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../../Libraries/C++/stb_image_write.h"
#include <omp.h>
#include <unistd.h>
#include <chrono>
#include <iostream>
#include "../../Libraries/Timer.h"
#define ALLOC_ALIGN 64
#define ALLOC_TRANSFER_ALIGN 4096
#define FLT_MIN 0.00000001f
#define ALLOC alloc_if(1) free_if(0)
#define FREE alloc_if(0) free_if(1)
#define REUSE alloc_if(0) free_if(0)
#define ALLOCANDFREE alloc_if(1) free_if(1)
#pragma offload_attribute (push, target(mic))
struct Complex {
float x, y;
Complex():y(0),x(0){};
Complex(float r, float i):y(i),x(r){};
};
inline Complex Add(const Complex left, const Complex right) {
return Complex(left.x+ right.x, left.y+ right.y);
}
inline Complex Multiply(const Complex left, const Complex right) {
return Complex(left.x* right.x - left.y* right.y, left.x* right.y + left.y* right.x);
}
inline float Magnitude(const Complex value) {
return sqrt(value.x*value.x + value.y*value.y);
}
inline int NormalizedIterations(const Complex value, const int n, const float bailout) {
return n + (log(log(bailout)) - log(log(Magnitude(value)))) / log(2.0f);
}
inline int CalcuateSinglePoint(Complex point, int bailout)
{
Complex startingPoint = Complex(0.0f,0.0f);
for (int k = 0 ; k < bailout; ++k) {
if (Magnitude(startingPoint) > 2.0f)
{
float returnValue = NormalizedIterations(startingPoint, k, bailout);
return returnValue;
}
startingPoint = Multiply(startingPoint, startingPoint);
startingPoint = Add(startingPoint, point);
}
return FLT_MIN;
}
#pragma offload_attribute (pop)
void Mandelbrot(char* output, unsigned int offset, int width, int height)
{
#pragma offload target(mic) mandatory out(output:length(width*height) ALLOCANDFREE)
#pragma omp parallel
{
Complex starting;
#pragma omp for collapse(2)
for (int y = 0; y < height; y++)
{
for (int x = 0; x < width; x++)
{
starting.x = (-2.5f + 3.5f*(x / (float)width));
starting.y = -1.25f + 2.5f*((y + offset) / (float)height);
output[width * y + x] = (char)(CalcuateSinglePoint(starting, 200));
}
}
}
}
int main(int argc, char** argv)
{
int width, height, numberOfThreads;
Timer timer(Timer::Mode::Single);
timer.Start();
if(argc != 3)
{
printf("%d", argc);
printf("Wrong number of arguments, correct number is: 1- width, 2 height\n");
return 0;
}
width = atoi(argv[1]);
height = width;
numberOfThreads = atoi(argv[2]);
char* results = (char*)_mm_malloc(sizeof(char)*width * height, ALLOC_TRANSFER_ALIGN);
Mandelbrot(results, 0, width, height);
_mm_free(results);
timer.Stop();
printf("OpenMP,%d,%d,%lu,%lu", numberOfThreads, width, timer.Get(), timer.Get());
return 0;
}