#pragma once
#include "colorstuff.h"
#ifdef __cplusplus
extern "C" {
#endif
void sdls_init(unsigned int width, unsigned int height);
void sdls_cleanup(void);
void sdls_blitrectangle_rgba(unsigned int x, unsigned int y, unsigned int width, unsigned int height, const rgba * src);
void sdls_blitrectangle_grayscale(unsigned int x, unsigned int y, unsigned int width, unsigned int height, const grayscale * src);
void sdls_draw(void);
rgba * sdls_loadimage_rgba(const char * file, size_t * width, size_t * height);
grayscale * sdls_loadimage_grayscale(const char * file, size_t * width, size_t * height);
#ifdef __cplusplus
}
#endif
