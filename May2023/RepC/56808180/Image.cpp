#include "Image.h"
#define ALLOC_ALLIGN 64
#pragma offload_attribute(push, target(mic))
Image *Image::CreateImage(int height, int width) {
Image* image = (Image*)_mm_malloc(sizeof(image), 64);
image->Area = (Pixel*)_mm_malloc(height*width*sizeof(Pixel), ALLOC_ALLIGN);
return image;
}
void Image::InitImage(Image *image, int height, int width) {
image->Area = (Pixel*)_mm_malloc(height*width*sizeof(Pixel), ALLOC_ALLIGN);
}
#pragma offload_attribute(pop)
