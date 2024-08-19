#include <omp.h>
#include <iostream>
#include <cmath>
#include "Bitmap.h"
#include "Logger.h"
#include "Filter.h"
#include "Kernels.h"

#define OPEN_MP_TEST

int main() {
double currentTime = 0;
Logger* logger = new Logger();

Bitmap* original = new Bitmap("./samples/original/hummingbird-squared.bmp");
Bitmap* negative = original->Clone();
Bitmap* grayscale = original->Clone();
Bitmap* gaussian = original->Clone();
Bitmap* laplace = original->Clone();
Bitmap* sobel = original->Clone();
Bitmap* prewitt = original->Clone();
Bitmap* scharr = original->Clone();

Filter filter;

#ifdef OPEN_MP_TEST
for (size_t i = 0; i <= 1; i++)
{
negative = original->Clone();
grayscale = original->Clone();

int thread = pow(2, i);

printf("Thread number [%d]: \n", thread);
omp_set_num_threads(thread);

currentTime = logger->CetCurrentTime();

#pragma omp parallel
{
#pragma omp sections
{
#pragma omp section
{
filter.Grayscale(*grayscale);
}

#pragma omp section
{
filter.Negative(*negative);
}
}
}

logger->GetElapsedTime(currentTime);
}

negative->Save("./samples/negative.bmp");
grayscale->Save("./samples/grayscale.bmp");

#endif 

omp_set_num_threads(1);
printf("Applying filter: Gaussian (5x)\n");
currentTime = logger->CetCurrentTime();
for (size_t i = 0; i < 5; i++)
filter.Convolve(*gaussian, gaussian5x5, 1 / 230.f, 0, false);
logger->GetElapsedTime(currentTime);

printf("Applying filter: Laplacian\n");
currentTime = logger->CetCurrentTime();
filter.Convolve(*laplace, laplacian5x5, 1, 0, false);
logger->GetElapsedTime(currentTime);

printf("Applying filter: Sobel\n");
currentTime = logger->CetCurrentTime();
filter.Convolve(*sobel, sobelx, sobely, 1, 0, false);
logger->GetElapsedTime(currentTime);

printf("Applying filter: Prewitt\n");
currentTime = logger->CetCurrentTime();
filter.Convolve(*prewitt, prewittx, prewitty, 1, 0, true);
logger->GetElapsedTime(currentTime);

printf("Applying filter: Scharr\n");
currentTime = logger->CetCurrentTime();
filter.Convolve(*scharr, scharrx, scharry, 1, 0, true);
logger->GetElapsedTime(currentTime);

printf("Writing image files\n");
original->Save("./samples/original.bmp");
gaussian->Save("./samples/gaussian.bmp");
laplace->Save("./samples/laplacian.bmp");
sobel->Save("./samples/sobel.bmp");
prewitt->Save("./samples/prewitt.bmp");
scharr->Save("./samples/scharr.bmp");

return 0;
}