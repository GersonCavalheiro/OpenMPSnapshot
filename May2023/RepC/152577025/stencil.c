#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#define OUTPUT_FILE "stencil.pgm"
void stencil(const int nx, const int ny, float * restrict image, float * restrict tmp_image);
void init_image(const int nx, const int ny, float * image, float * tmp_image);
void output_image(const char * file_name, const int nx, const int ny, float *image);
double wtime(void);
int main(int argc, char *argv[]) {
if (argc != 4) {
fprintf(stderr, "Usage: %s nx ny niters\n", argv[0]);
exit(EXIT_FAILURE);
}
int nx = atoi(argv[1]);
int ny = atoi(argv[2]);
int niters = atoi(argv[3]);
float *image     = _mm_malloc(sizeof(float)*(nx+2)*(ny+2), 64);
float *tmp_image = _mm_malloc(sizeof(float)*(nx+2)*(ny+2), 64);
init_image(nx+2, ny+2, image, tmp_image);
double tic = wtime();
for (int t = 0; t < niters; ++t) {
stencil(nx+2, ny+2, image, tmp_image);
stencil(nx+2, ny+2, tmp_image, image);
}
double toc = wtime();
printf("----------------------------------------\n");
printf(" runtime:          %lf s\n", toc-tic);
printf(" memory bandwidth: %lf GB/s\n", (4 * 6 * (nx / 1024) * (ny / 1024) * 2 * niters) / ((toc - tic) * 1024) );
printf("----------------------------------------\n");
output_image(OUTPUT_FILE, nx+2, ny+2, image);
_mm_free(image);
}
void stencil(const int nx, const int ny, float * restrict image, float * restrict tmp_image) {
for( int i = 1; i < nx-1; i++ ) {
__assume_aligned(tmp_image, 64);
__assume_aligned(    image, 64);
__assume( ((i - 1) * ny) % 16 == 0 );
__assume( ((i + 1) * ny) % 16 == 0 );
__assume( (-1 + i * ny)  % 16 == 0 );
__assume( ( 1 + i * ny)  % 16 == 0 );
#pragma simd
#pragma unroll (4)
for( int j = 1; j < ny-1; j++ ) {
tmp_image[ j + i * ny ]  = image[ j + i * ny ] * 0.6f +
( image[ j - 1 + i * ny ]    +
image[ j + 1 + i * ny ]    +
image[ j + (i - 1) * ny ]  +
image[ j + (i + 1) * ny ] ) * 0.1f;
}
}
}
void init_image(const int nx, const int ny, float * image, float * tmp_image) {
for (int j = 0; j < ny; ++j) {
for (int i = 0; i < nx; ++i) {
image[j+i*ny] = 0.0;
}
}
for (int i = 0; i < ny; ++i)
for (int j = 0; j < nx; ++j)
tmp_image[j+i*ny] = 0.0;
for (int j = 0; j < 8; ++j) {
for (int i = 0; i < 8; ++i) {
for (int jj = j*(ny-2)/8; jj < (j+1)*(ny-2)/8; ++jj) {
for (int ii = i*(nx-2)/8; ii < (i+1)*(nx-2)/8; ++ii) {
if ((i+j)%2)
image[(jj+1)+(ii+1)*ny] = 100.0;
}
}
}
}
}
void output_image(const char * file_name, const int nx, const int ny, float *image) {
FILE *fp = fopen(file_name, "w");
if (!fp) {
fprintf(stderr, "Error: Could not open %s\n", OUTPUT_FILE);
exit(EXIT_FAILURE);
}
fprintf(fp, "P5 %d %d 255\n", nx-2, ny-2);
float maximum = 0.0;
for (int j = 1; j < ny-1; ++j) {
for (int i = 1; i < nx-1; ++i) {
if (image[j+i*ny] > maximum)
maximum = image[j+i*ny];
}
}
for (int j = 1; j < ny-1; ++j) {
for (int i = 1; i < nx-1; ++i) {
fputc((char)(255.0*image[j+i*ny]/maximum), fp);
}
}
fclose(fp);
}
double wtime(void) {
struct timeval tv;
gettimeofday(&tv, NULL);
return tv.tv_sec + tv.tv_usec*1e-6;
}
