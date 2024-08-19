

#pragma once

#include <string>

namespace tdogl {


class Bitmap {
public:

enum Format {
Format_Grayscale = 1, 
Format_GrayscaleAlpha = 2, 
Format_RGB = 3, 
Format_RGBA = 4 
};


Bitmap(unsigned width, 
unsigned height, 
Format format,
const unsigned char* pixels = NULL);
~Bitmap();


static Bitmap bitmapFromFile(std::string filePath);


unsigned width() const;


unsigned height() const;


Format format() const;


unsigned char* pixelBuffer() const;


unsigned char* getPixel(unsigned int column, unsigned int row) const;


void setPixel(unsigned int column, unsigned int row, const unsigned char* pixel);


void flipVertically();


void rotate90CounterClockwise();


void copyRectFromBitmap(const Bitmap& src, 
unsigned srcCol, 
unsigned srcRow, 
unsigned destCol, 
unsigned destRow,
unsigned width,
unsigned height);


Bitmap(const Bitmap& other);


Bitmap& operator = (const Bitmap& other);

private:
Format _format;
unsigned _width;
unsigned _height;
unsigned char* _pixels;

void _set(unsigned width, unsigned height, Format format, const unsigned char* pixels);
static void _getPixelOffset(unsigned col, unsigned row, unsigned width, unsigned height, Format format);
};

}
