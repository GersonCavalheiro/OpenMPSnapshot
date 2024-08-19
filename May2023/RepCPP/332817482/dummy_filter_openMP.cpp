#include "../../utils/filter.h"
#include <omp.h>
#include <iostream>


void DummyFilter::applyFilter(Image *image, Image *newImage) {
#pragma omp parallel for
for (unsigned int i = 1; i < image->height - 1; ++i) {
for (unsigned int j = 1; j < image->width - 1; ++j) {
newImage->matrix[i][j] = image->matrix[i][j];
}
}
}
