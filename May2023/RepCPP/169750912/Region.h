#pragma once

#include "Fractal.h"


struct Region {


precision_t minReal, maxImaginary;


precision_t maxReal, minImaginary;


unsigned int width, height;


unsigned short int maxIteration;


int validation;


unsigned int guaranteedDivisor;


int hOffset, vOffset;


enum fractal_type fractal;


unsigned short int regionCount;

unsigned int getPixelCount() {
return width * height;
}


long double projectReal(int pixelX) {
return minReal + (static_cast<long double>(pixelX)
* (maxReal - minReal)) / width;
}

long double projectImag(int pixelY) {
return minImaginary + (static_cast<long double>(pixelY)
* (maxImaginary - minImaginary)) / height;
}
};