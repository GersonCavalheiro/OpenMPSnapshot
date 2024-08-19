#include "pgm.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <omp.h>
#include <string.h>
#define THRESHOLD(a,max) ((a > max)? max: 0)
pgm_t *sobel_filter(const pgm_t *image)
{	
register int x_sum, y_sum;
const register int height = image->height, width = image->width;
const int maxval = image->maxval;
pgm_t *new_image = new_pgm_image(width, height, image->maxval);
#pragma omp parallel for private(x_sum, y_sum) schedule(static, 1)
for (register int x = 1; x < height - 1; x++) {
for (register int y = 1; y < width - 1; y++) {
x_sum = (
image->pixels[(x + 1)*width + (y + 1)] -
image->pixels[(x + 1)*width + (y - 1)] +
(image->pixels[   (x)*width + (y + 1)] << 1) -
(image->pixels[   (x)*width + (y - 1)] << 1) +
image->pixels[(x - 1)*width + (y + 1)] -
image->pixels[(x - 1)*width + (y - 1)]
);
y_sum = (
image->pixels[ (x + 1)*width + (y + 1)] +
(image->pixels[(x + 1)*width + (y)    ] << 1) +
image->pixels[ (x + 1)*width + (y - 1)] -
image->pixels[ (x - 1)*width + (y + 1)] -
(image->pixels[(x - 1)*width + (y)    ] << 1) -
image->pixels[ (x - 1)*width + (y - 1)]
);
new_image->pixels[x * width + y] = THRESHOLD(abs(x_sum) + abs(y_sum), maxval);
}
}
return new_image;
}
int main(int argc, char **argv) {
if (argc < 3) {
printf("Invalid Arguments!\n");
return 1;
}
pgm_t *image;
if (strcmp(argv[1], "-r") == 0) {
image = new_pgm_image(1024, 1024, 8);
rand_pgm_image(image);
}
else if (strcmp(argv[1], "--std-test") == 0) {
image = new_pgm_image(256, 256, 255);
fill_pgm_image(image);
pgm_t *new_image = sobel_filter(image); 
store_pgm_image(new_image, argv[2]);	
bool checked = check_pgm_image(new_image);
printf(checked
? "The image was filtered correctly!\n"
: "The image was not filtered correctly!\n");
}
else {
image = load_pgm_image(argv[1]); 
}
pgm_t *new_image = sobel_filter(image); 
store_pgm_image(new_image, argv[2]);	
return 0;
}
