#pragma once
#include <stdlib.h>
#include "common.h"



void filter_gauss_2d_flt(float *data, float *data_copy, float *data_row,
float *data_col, const size_t size_x,
const size_t size_y, const size_t n_iter,
const size_t filter_radius) {
const size_t size_xy = size_x * size_y;
float *ptr = data + size_xy;
float *ptr2;

while (ptr > data) {
ptr -= size_x;
for (size_t i = n_iter; i--;)
filter_boxcar_1d_flt(ptr, data_row, size_x, filter_radius);
}

for (size_t x = size_x; x--;) {
ptr = data + size_xy - size_x + x;
ptr2 = data_copy + size_y;
while (ptr2-- > data_copy) {
*ptr2 = *ptr;
ptr -= size_x;
}

for (size_t i = n_iter; i--;)
filter_boxcar_1d_flt(data_copy, data_col, size_y, filter_radius);

ptr = data + size_xy - size_x + x;
ptr2 = data_copy + size_y;
while (ptr2-- > data_copy) {
*ptr = *ptr2;
ptr -= size_x;
}
}

return;
}
