#include "df.h"
#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

static float parabola_intersect(float* restrict f, size_t p, size_t q) {
float p1_x = (float)p;
float p2_x = (float)q;
float p1_y = f[p];
float p2_y = f[q];
return ((p2_y - p1_y) + ((p2_x * p2_x) - (p1_x * p1_x))) / (2 * (p2_x - p1_x));
}

static void dist_transform_1d(float* restrict img_row, size_t w, size_t y, size_t* restrict v, float* restrict h,
float* restrict z, float* restrict img_tpose_out, bool do_sqrt) {
if (w <= 1) {
img_tpose_out[0] = img_row[0];
return;
}

size_t offset = 0;
while (isinf(img_row[offset]) && offset < w) ++offset;

if (offset == w) {
for (size_t i = 0; i < w; ++i) {
size_t tpose_idx = y + w * i;
img_tpose_out[tpose_idx] = INFINITY;
}
return;
}

v[0] = offset;
h[0] = img_row[offset];

size_t k = 0;
for (size_t q = offset + 1; q < w; ++q) {
if (isinf(img_row[q])) continue;

float s = parabola_intersect(img_row, v[k], q);

while (k > 0 && s <= z[k - 1]) {
--k;
s = parabola_intersect(img_row, v[k], q);
}
z[k] = s;
++k;
v[k] = q;
h[k] = img_row[q];
}

size_t j = 0;
for (size_t q = 0; q < w; ++q) {
while (j < k && z[j] < (float)q) ++j;

size_t v_j = v[j];
float displacement = (float)q - (float)v_j;

size_t tpose_idx = y + w * q;
img_tpose_out[tpose_idx] = displacement * displacement + h[j];

if (do_sqrt) img_tpose_out[tpose_idx] = sqrtf(img_tpose_out[tpose_idx]);
}
}

static void dist_transform_axis(float* restrict img, size_t w, size_t h, float* restrict img_tpose_out, bool do_sqrt) {
#pragma omp parallel
{
ptrdiff_t y;
size_t* v = malloc(sizeof(size_t) * (size_t)(w));
float* p = malloc(sizeof(float) * (size_t)(w));
float* z = malloc(sizeof(float) * (size_t)(w - 1));

#pragma omp for schedule(static)
for (y = 0; y < (ptrdiff_t)(h); ++y) {
float* img_slice = img + ((size_t)y * w);
dist_transform_1d(img_slice, w, (size_t)y, v, p, z, img_tpose_out, do_sqrt);
}

free(z);
free(p);
free(v);
}
}

void dist_transform_2d(float* img, size_t w, size_t h) {
float* img_tpose = malloc(w * h * sizeof(float));

dist_transform_axis(img, w, h, img_tpose, false);

dist_transform_axis(img_tpose, h, w, img, true);

free(img_tpose);
}
