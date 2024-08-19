#include "../../utils/filter.h"
#include <omp.h>


void BrightnessFilter::applyFilter(Image *image, Image *newImage) {
float brightness = this->brightness;

#pragma omp parallel for
for (unsigned int i = 1; i < image->height - 1; ++i) {
for (unsigned int j = 1; j < image->width - 1; ++j) {
newImage->matrix[i][j] = Pixel(
image->matrix[i][j].r * brightness,
image->matrix[i][j].g * brightness,
image->matrix[i][j].b * brightness,
image->matrix[i][j].a);
}
}
}
