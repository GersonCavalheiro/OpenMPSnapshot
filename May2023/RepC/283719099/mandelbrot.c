#include <math.h>
#include <stdlib.h>
#include <pixmap.h>
#include <palette.h>
#include <stdbool.h>
#define ROUND(x)	((int) round(x))
#define MIN(x,y)	(((x)<(y))?(x):(y))
#define MAX(x,y)	(((x)>(y))?(x):(y))
void colormap_init   (pixel_t *, int);
void draw_image      (pixmap_t *, pixmap_t *, pixel_t *, int, int, double, double, double);
void rough_sketch    (pixmap_t *,             pixel_t *, int,      double, double, double);
int  mandelbrot      (int, double, double);
bool detect_edge     (pixmap_t *, pixel_t  *, int, int);
bool equivalent_color(pixel_t, pixel_t);
int main(int argc, char **argv)
{
pixmap_t image, sketch;
pixel_t  colormap[ITER_MAX];
pixmap_create(&image , WIDTH, HEIGHT);
pixmap_create(&sketch, WIDTH, HEIGHT);
colormap_init(colormap, ITER_MAX);
draw_image(&image, &sketch, colormap, ITER_MAX, AALEV, CENTER_R, CENTER_I, RADIUS);
pixmap_write_ppmfile(&image, "output.ppm");
pixmap_destroy(&sketch);
pixmap_destroy(&image );
return 0;
}
void colormap_init(pixel_t *colormap, int iter_max)
{
int colormap_mask = COLORMAP_CYCLE - 1;
colormap[0] = pixel_set_rgb(0x00, 0x00, 0x00);
for (int i = 1; i < iter_max; i++)
#ifdef REVERSE_COLORMAP
colormap[i] = palette(COLORMAP_TYPE, 0x00, colormap_mask,
colormap_mask - (i & colormap_mask));
#else
colormap[i] = palette(COLORMAP_TYPE, 0x00, colormap_mask,
i & colormap_mask);
#endif
return;
}
void draw_image(pixmap_t *image, pixmap_t *sketch, pixel_t *colormap,
int iter_max, int sampling, double c_r, double c_i, double radius)
{				
int iter_mask = iter_max - 1;
int width, height;
double d;
pixmap_get_size(image, &width, &height);
d = 2.0 * radius / (sampling * MIN(width, height));
rough_sketch(sketch, colormap, iter_max, c_r, c_i, radius);
#pragma omp parallel for schedule(static,1) collapse(2)
for (int j = 0; j < height; j++) {
for (int i = 0; i < width; i++) {
pixel_t pixel;
if (detect_edge(sketch, &pixel, i, j)) {	
int sum_r = 0, sum_g = 0, sum_b = 0;
for (int n = j * sampling; n < (j + 1) * sampling; n++)
for (int m = i * sampling; m < (i + 1) * sampling; m++) {
double p_r = c_r + d * (m - sampling * width  / 2),
p_i = c_i + d * (sampling * height / 2 - n);
int   iter = mandelbrot(iter_max, p_r, p_i);
sum_r += pixel_get_r(colormap[iter & iter_mask]);
sum_g += pixel_get_g(colormap[iter & iter_mask]);
sum_b += pixel_get_b(colormap[iter & iter_mask]);
}
pixel = pixel_set_rgb(ROUND((double) sum_r / (sampling * sampling)),
ROUND((double) sum_g / (sampling * sampling)),
ROUND((double) sum_b / (sampling * sampling)));
}
pixmap_put_pixel(image, pixel, i, j);
}
}
return;
}
void rough_sketch(pixmap_t *sketch, pixel_t *colormap,
int iter_max, double c_r, double c_i, double radius)
{
int iter_mask = iter_max - 1;
int width, height;
double d;
pixmap_get_size(sketch, &width, &height);
d = 2.0 * radius / MIN(width, height);
#pragma omp parallel for schedule(static,1) collapse(2)
for (int j = 0; j < height; j++) {
for (int i = 0; i < width; i++) {
double p_r = c_r + d * (i - width  / 2),
p_i = c_i + d * (height / 2 - j);
int   iter = mandelbrot(iter_max, p_r, p_i);
pixmap_put_pixel(sketch, colormap[iter & iter_mask], i, j);
}
}
return;
}
int mandelbrot(int iter_max, double p_r, double p_i)
{				
int i;
double z_r, z_i, work;
z_r  = p_r;
z_i  = p_i;
work = 2.0 * z_r * z_i;
for (i = 1; i < iter_max && (z_r *= z_r) +
(z_i *= z_i) < 4.0; i++) {
z_r += p_r - z_i ;
z_i  = p_i + work;
work = 2.0 * z_r * z_i;
}
return i;
}
bool detect_edge(pixmap_t *pixmap, pixel_t *pixel, int x, int y)
{
int width, height;
pixmap_get_size (pixmap, &width, &height);
pixmap_get_pixel(pixmap, pixel, x, y);
for (int j = MAX(0, y - 1); j <= MIN(height - 1, y + 1); j++)
for (int i = MAX(0, x - 1); i <= MIN(width - 1, x + 1); i++)
if (i != x || j != y) {
pixel_t p;
pixmap_get_pixel(pixmap, &p, i, j);
if (!equivalent_color(*pixel, p))
return true;
}
return false;
}
bool equivalent_color(pixel_t p, pixel_t q)
#ifdef USE_SAME_COLOR
{
return pixel_get_r(p) == pixel_get_r(q) &&
pixel_get_g(p) == pixel_get_g(q) &&
pixel_get_b(p) == pixel_get_b(q);
}
#else				
{
return 3 * abs(pixel_get_r(p) - pixel_get_r(q)) +
6 * abs(pixel_get_g(p) - pixel_get_g(q)) +
1 * abs(pixel_get_b(p) - pixel_get_b(q)) < 15;
}
#endif
