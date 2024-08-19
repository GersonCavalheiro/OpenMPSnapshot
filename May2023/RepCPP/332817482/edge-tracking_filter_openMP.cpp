#include "../../utils/filter.h"
#include <omp.h>


void EdgeTrackingFilter::applyFilter(Image *image, Image *newImage) {
const int weak = 100;
const int strong = 255;

#pragma omp parallel for
for (unsigned int i = 1; i < image->height - 1; ++i) {
for (unsigned int j = 1; j < image->width - 1; ++j) {
if (image->matrix[i][j].r == weak) {
if (image->matrix[i - 1][j - 1].r == strong || image->matrix[i - 1][j].r == strong ||
image->matrix[i - 1][j + 1].r  == strong || image->matrix[i][j - 1].r == strong ||
image->matrix[i][j + 1].r == strong || image->matrix[i + 1][j - 1].r == strong ||
image->matrix[i + 1][j].r == strong || image->matrix[i + 1][j + 1].r == strong) {

newImage->matrix[i][j] = Pixel(strong, strong, strong, image->matrix[i][j].a);
} else {
newImage->matrix[i][j] = Pixel(0, 0, 0, image->matrix[i][j].a);
}
} else {
newImage->matrix[i][j] = image->matrix[i][j];
}
}
}
}
