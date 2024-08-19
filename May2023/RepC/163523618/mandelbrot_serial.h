#pragma once
#include <stdio.h>
#include "complex_number.h"
void generate_fractal_serial(const int max_iteration, const int width, const int height)
{
const int num_pixels = width * height;
const complexx center = { .real = -0.7,.imag = 0 }, span = { .real = 2.7,.imag = -(4 / 3.0)*2.7*height / width };
const complexx begin = complex_div(complex_subtract(center, span), 2.0);
for (int pix = 0; pix < num_pixels; ++pix)
{
const int x = pix % width, y = pix / width;
const complexx value = { .real = x * get_complex_real(span) / (width + 1.0),
.imag = y * get_complex_imag(span) / (height + 1.0) };
complexx c = complex_add(begin, value);
int n = mandelbrot_calculate(c, max_iteration);
if (n == max_iteration)
{
n = 0;
}
char xter = ' ';
if (n > 0)
{
static const char charset[] = ".,c8M@jawrpogOQEPGJ";
xter = charset[n % (sizeof(charset) - 1)];
}
putchar(xter);
if (x + 1 == width) puts("|");
}
}