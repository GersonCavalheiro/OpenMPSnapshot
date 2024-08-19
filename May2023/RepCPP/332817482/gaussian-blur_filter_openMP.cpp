#include "../../utils/filter.h"
#include <omp.h>

static const float kernel[3][3] = {{1. / 16., 2. / 16., 1. / 16.},
{2. / 16., 4. / 16., 2. / 16.},
{1. / 16., 2. / 16., 1. / 16.}};


void GaussianBlurFilter::applyFilter(Image *image, Image *newImage) {
#pragma omp parallel for
for (unsigned int i = 1; i < image->height - 1; ++i) {
for (unsigned int j = 1; j < image->width - 1; ++j) {
Pixel newPixel;
float red, green, blue;
red = green = blue = 0;
newPixel.a = image->matrix[i][j].a;
newPixel.r = newPixel.b = newPixel.g = 0;

for (int ki = -1; ki <= 1; ++ki) {
for (int kj = -1; kj <= 1; ++kj) {
red   += static_cast<float>(image->matrix[i + ki][j + kj].r) * kernel[ki + 1][kj + 1];
green += static_cast<float>(image->matrix[i + ki][j + kj].g) * kernel[ki + 1][kj + 1];
blue  += static_cast<float>(image->matrix[i + ki][j + kj].b) * kernel[ki + 1][kj + 1];
}
}

red = (red < 0) ? 0 : red;
green = (green < 0) ? 0 : green;
blue = (blue < 0) ? 0 : blue;
newPixel.r = (red > 255) ? 255 : red;
newPixel.g = (green > 255) ? 255 : green;
newPixel.b = (blue > 255) ? 255 : blue;
newImage->matrix[i][j] = newPixel;
}
}
}
