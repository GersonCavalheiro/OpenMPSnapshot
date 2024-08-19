#define PNG_NO_SETJMP

#include <assert.h>
#include <png.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <immintrin.h>
#include <omp.h>

#define MAX_ITER 10000

inline void write_png(const char *filename, const int width, const int height, const int *buffer)
{
FILE *fp = fopen(filename, "wb");
assert(fp);
png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
assert(png_ptr);
png_infop info_ptr = png_create_info_struct(png_ptr);
assert(info_ptr);
png_init_io(png_ptr, fp);
png_set_IHDR(png_ptr, info_ptr, width, height, 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
png_write_info(png_ptr, info_ptr);
size_t row_size = 3 * width * sizeof(png_byte);
png_bytep row = (png_bytep)malloc(row_size);
for (int y = 0; y < height; ++y)
{
memset(row, 0, row_size);
#pragma omp parallel for schedule(static)
for (int x = 0; x < width; ++x)
{
int p = buffer[(height - 1 - y) * width + x];
png_bytep color = row + x * 3;
if (p != MAX_ITER)
{
if (p & 16)
{
color[0] = 240;
color[1] = color[2] = p % 16 * 16;
}
else
{
color[0] = p % 16 * 16;
}
}
}
png_write_row(png_ptr, row);
}
free(row);
png_write_end(png_ptr, NULL);
png_destroy_write_struct(&png_ptr, &info_ptr);
fclose(fp);
}

inline void in_set(double *x0, double *y0, int *ret_val)
{
int repeats = 0;
bool finished[2] = {0};

__m128d xm = _mm_set_pd(0, 0);
__m128d ym = _mm_set_pd(0, 0);
__m128d x2m = _mm_set_pd(0, 0);
__m128d y2m = _mm_set_pd(0, 0);
__m128d x0m = _mm_load_pd(x0);
__m128d y0m = _mm_load_pd(y0);

while (repeats < MAX_ITER)
{
ym = _mm_mul_pd(xm, ym);
ym = _mm_add_pd(ym, ym);
ym = _mm_add_pd(ym, y0m);
xm = _mm_sub_pd(x2m, y2m);
xm = _mm_add_pd(xm, x0m);
x2m = _mm_mul_pd(xm, xm);
y2m = _mm_mul_pd(ym, ym);
repeats++;
__m128d lenm = _mm_add_pd(x2m, y2m);
double len[2];
_mm_store_pd(len, lenm);
for (int i = 0; i < 2; i++)
{
if (!finished[i])
{
ret_val[i] = repeats;
if (len[i] > 4)
finished[i] = 1;
}
}
if (finished[0] && finished[1])
break;
}
}

int main(int argc, char **argv)
{

assert(argc == 9);
int num_threads = strtol(argv[1], 0, 10);
double left = strtod(argv[2], 0);
double right = strtod(argv[3], 0);
double lower = strtod(argv[4], 0);
double upper = strtod(argv[5], 0);
int width = strtol(argv[6], 0, 10);
int height = strtol(argv[7], 0, 10);
const char *filename = argv[8];


int *image = new int[width * height * 2];

double unit_y = ((upper - lower) / height);
double unit_x = ((right - left) / width);

#pragma omp parallel for schedule(dynamic)
for (int j = 0; j < height * width; j += 2)
{
double *y0 = new double[2];
double *x0 = new double[2];
y0[0] = ((j) / width) * unit_y + lower;
x0[0] = ((j) % width) * unit_x + left;
y0[1] = ((j + 1) / width) * unit_y + lower;
x0[1] = ((j + 1) % width) * unit_x + left;
in_set(x0, y0, image + j);
delete y0;
delete x0;
}


write_png(filename, width, height, image);
delete image;
}
