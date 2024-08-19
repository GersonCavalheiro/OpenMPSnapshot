#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../lib/imagelib/imagelib.h"
float **read_kernel(char *kernel_path, size_t *H, size_t *W) {
FILE *file = fopen(kernel_path, "r");
char *line = NULL;
size_t length = 0;
ssize_t read;
read = getline(&line, &length, file);
size_t width = atoi(line);
read = getline(&line, &length, file);
size_t height = atoi(line);
float **K = calloc(height, sizeof(float *));
size_t row = 0;
while ((read = getline(&line, &length, file)) != -1) {
K[row] = calloc(width, sizeof(float));
size_t col = 0;
char *pch = strtok(line, " ");
while (pch != NULL) {
K[row][col] = atof(pch);
col += 1;
pch = strtok(NULL, " ");
}
row += 1;
}
fclose(file);
free(line);
(*H) = height;
(*W) = width;
return K;
}
Image *create_image(size_t height, size_t width) {
Image *img = malloc(sizeof(Image));
img->width = width;
img->height = height;
img->pixels = calloc(height, sizeof(Color *));
for (size_t row = 0; row < height; row++) {
img->pixels[row] = calloc(width, sizeof(Color));
}
return img;
}
int main(int argc, char *argv[]) {
char *img_in_path = argv[1];
char *kernel_path = argv[2];
size_t passes = atoi(argv[3]);
char *img_out_path = argv[4];
size_t k_height = 0;
size_t k_width = 0;
float **K = read_kernel(kernel_path, &k_height, &k_width);
size_t k_width_center = floor(k_width / 2);
size_t k_height_center = floor(k_height / 2);
size_t k_width_radius = k_width - k_width_center - 1;
size_t k_height_radius = k_height - k_height_center - 1;
Image *src = img_png_read_from_file(img_in_path);
Image *out = NULL;
size_t pass;
size_t row;
size_t col;
size_t i;
ssize_t row_start;
size_t j;
ssize_t col_start;
float acc_R = 0.0;
float acc_G = 0.0;
float acc_B = 0.0;
Color I_ij;
float K_ij = 0;
for (pass = 0; pass < passes; pass++) {
out = create_image(src->height, src->width);
#pragma omp parallel for private(                                              row, col, i, j, acc_R, acc_G, acc_B, row_start, col_start, K_ij, I_ij)     shared(pass, src, out, K, k_height, k_width, k_width_center,           k_height_center, k_width_radius, k_height_radius) collapse(2)
for (row = 0; row < src->height; row++) {
for (col = 0; col < src->width; col++) {
acc_R = acc_G = acc_B = 0.0;
i = 0;
for (row_start = row - k_height_radius;
row_start <= row + k_height_radius; row_start++) {
j = 0;
for (col_start = col - k_width_radius;
col_start <= col + k_width_radius; col_start++) {
if (row_start < 0 || row_start >= src->height) {
} else if (col_start < 0 || col_start >= src->width) {
} else {
K_ij = K[i][j];
I_ij = src->pixels[row_start][col_start];
acc_R += K_ij * I_ij.R;
acc_G += K_ij * I_ij.G;
acc_B += K_ij * I_ij.B;
}
j += 1;
}
i += 1;
}
out->pixels[row][col].R = acc_R;
out->pixels[row][col].G = acc_G;
out->pixels[row][col].B = acc_B;
}
}
img_destroy(src);
src = out;
}
img_png_write_to_file(out, img_out_path);
img_destroy(out);
return 0;
}
