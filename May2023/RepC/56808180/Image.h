#ifndef GENETICALGORITHM_IMAGE_H
#define GENETICALGORITHM_IMAGE_H
#include "Pixel.h"
#pragma offload_attribute(push, target(mic))
struct Image {
Pixel* Area;
static Image* CreateImage(int height, int width);
static void InitImage(Image* image, int height, int width);
};
#pragma offload_attribute(pop)
#endif 
