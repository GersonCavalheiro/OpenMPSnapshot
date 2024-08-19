#include "../../utils/filter.h"
#include <omp.h>


void ConstrastFilter::applyFilter(Image *image, Image *newImage) {
float factor = (259. * (this->contrast + 255.)) / (255. * (259. - this->contrast));

#pragma omp parallel for
for (unsigned int i = 1; i < image->height - 1; ++i) {
for (unsigned int j = 1; j < image->width - 1; ++j) {
Pixel newPixel;
float tempColor;

newPixel.a = image->matrix[i][j].a;
tempColor = factor * (image->matrix[i][j].r - 128) + 128;
tempColor = (tempColor < 0) ? 0 : tempColor;
newPixel.r = (tempColor > 255) ? 255 : tempColor;
tempColor = factor * (image->matrix[i][j].g - 128) + 128;
tempColor = (tempColor < 0) ? 0 : tempColor;
newPixel.g = (tempColor > 255) ? 255 : tempColor;
tempColor = factor * (image->matrix[i][j].b - 128) + 128;
tempColor = (tempColor < 0) ? 0 : tempColor;
newPixel.b = (tempColor > 255) ? 255 : tempColor;

newImage->matrix[i][j] = newPixel;
}
}
}
