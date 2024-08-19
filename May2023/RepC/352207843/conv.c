#include <omp.h>
#include <stdlib.h>
#include <stdio.h>
#include "include/image.h"
#include "include/conv.h"
static char conv2D_region(struct raw_image *img, int offset, const char *kernel, int n) {
int i, j, k_offset = 0;
unsigned char acc = 0;
for (i = 0; i < n; i++) {
for (j = 0; j < n; j++) {
acc += (img->data[offset + j] * kernel[k_offset + j]);
}
offset += img->width;
k_offset += n;
}
return acc;
}
struct raw_image *conv_2d(struct raw_image *in, const char *kernel, int n) {
int row, col;
struct raw_image *out;
int in_offset, out_offset;
out = raw_create(in->width - (2 * (n/2)),
in->height - (2 * (n/2)),
in->nchannels);
in_offset = 0;
out_offset = 0;
for (row = 0; row < out->height; row++) {
for (col = 0; col < out->width; col++) {
out->data[out_offset + col] = conv2D_region(in, in_offset + col, kernel, n);
}
in_offset += in->width;
out_offset += out->width;
}
return out;
}
struct raw_image *conv_2d_parallel(struct raw_image *in, const char *kernel, int n) {
int row, col;
struct raw_image *out;
int in_offset, out_offset;
out = raw_create(in->width - (2 * (n/2)),
in->height - (2 * (n/2)),
in->nchannels);
#pragma omp parallel for private(in_offset, out_offset) shared(in, out)
for (row = 0; row < out->height; row++) {
in_offset = row * in->width;
out_offset = row * out->width;
for (col = 0; col < out->width; col++) {
out->data[out_offset + col] = conv2D_region(in, in_offset + col, kernel, n);
}
}
return out;
}
