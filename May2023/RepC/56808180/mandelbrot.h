#pragma once
#include <iostream>
#include <fstream>
#include <chrono>
#include <cstdlib>
#include "Complex.h"
#include <mm_malloc.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../../../Libraries/C++/stb_image_write.h"
#define ALLOC_ALIGN 64
#define ALLOC_TRANSFER_ALIGN 4096
#define MAX_COLOR_VALUE 200
unsigned char GrayValue(float n) {
return (char)(n);
}
void Mandelbrot(char* output, unsigned int offset, unsigned int width, unsigned int height)
{
Complex *complex = (Complex*)_mm_malloc(sizeof(Complex), ALLOC_ALIGN);
Complex c;
float count;
for (int y = 0; y < height; ++y)
{
for (int x = 0 ; x < width; ++x)
{
c.Real = (-2.5f + 3.5f*(x / (float)width));
c.Imaginary = -1.25f + 2.5f*((y + offset) / (float)height);
count = c.BoundedOrbit(complex, 2.0f, MAX_COLOR_VALUE);
output[width * y + x] = GrayValue(count);
}
}
_mm_free(complex);
}