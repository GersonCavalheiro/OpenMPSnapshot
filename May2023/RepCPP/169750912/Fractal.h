#pragma once

typedef long double precision_t;

enum fractal_type {
mandelbrot,
mandelbrot32,
mandelbrot64,
mandelbrotSIMD32,
mandelbrotSIMD64,
mandelbrotOpenMP,
mandelbrotOpenMP32,
mandelbrotOpenMP64,
mandelbrotOpenMPSIMD32,
mandelbrotOpenMPSIMD64,
};

class Fractal {
public:

virtual void calculateFractal(precision_t* cReal, precision_t* cImaginary, unsigned short int maxIteration, unsigned int vectorLength, unsigned short int* dest) = 0;
virtual ~Fractal();


static precision_t deltaReal(precision_t maxReal, precision_t minReal, int xRes);


static precision_t deltaImaginary(precision_t maxImaginary, precision_t minImaginary, int yRes);
};
