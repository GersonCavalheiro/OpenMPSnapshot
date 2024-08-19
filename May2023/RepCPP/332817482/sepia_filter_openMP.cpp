#include "../../utils/filter.h"
#include <omp.h>


void SepiaFilter::applyFilter(Image *image, Image *newImage) {
#pragma omp parallel for
for (unsigned int i = 1; i < image->height - 1; ++i) {
for (unsigned int j = 1; j < image->width - 1; ++j) {
Pixel newPixel;
int tempColor;

newPixel.a = image->matrix[i][j].a;
tempColor = (image->matrix[i][j].r * 0.393) + (image->matrix[i][j].g * 0.769) + (image->matrix[i][j].b * 0.189);
tempColor = (tempColor < 0) ? 0 : tempColor;
newPixel.r = (tempColor > 255) ? 255 : tempColor;
tempColor = (image->matrix[i][j].r * 0.349) + (image->matrix[i][j].g * 0.686) + (image->matrix[i][j].b * 0.168);
tempColor = (tempColor < 0) ? 0 : tempColor;
newPixel.g = (tempColor > 255) ? 255 : tempColor;
tempColor = (image->matrix[i][j].r * 0.272) + (image->matrix[i][j].g * 0.534) + (image->matrix[i][j].b * 0.131);
tempColor = (tempColor < 0) ? 0 : tempColor;
newPixel.b = (tempColor > 255) ? 255 : tempColor;

newImage->matrix[i][j] = newPixel;
}
}
}
