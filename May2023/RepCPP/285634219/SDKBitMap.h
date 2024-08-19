
#ifndef SDKBITMAP_H_
#define SDKBITMAP_H_


#include <cstdlib>
#include <iostream>
#include <string.h>
#include <stdio.h>

#define SDK_SUCCESS 0
#define SDK_FAILURE 1

static const short bitMapID = 19778;



#pragma pack(push,1)

#ifdef _OPENMP

typedef struct __attribute__((__aligned__(4)))
{
unsigned char x;
unsigned char y;
unsigned char z;
unsigned char w;
} uchar4;

typedef struct __attribute__((__aligned__(16)))
{
float x;
float y;
float z;
float w;
} float4;
#endif


typedef uchar4 ColorPalette;


typedef struct
{
short id;
int size;
short reserved1;
short reserved2;
int offset;
} BitMapHeader;



typedef struct
{
int sizeInfo;
int width;
int height;
short planes;
short bitsPerPixel;
unsigned compression;
unsigned imageSize;
int xPelsPerMeter;
int yPelsPerMeter;
int clrUsed;
int clrImportant;
} BitMapInfoHeader;


class SDKBitMap : public BitMapHeader, public BitMapInfoHeader
{
private:
uchar4 * pixels_;               
int numColors_;                 
ColorPalette * colors_;         
bool isLoaded_;                 
void releaseResources(void)     
{
if (pixels_ != NULL)
{
delete[] pixels_;
}
if (colors_ != NULL)
{
delete[] colors_;
}
pixels_    = NULL;
colors_    = NULL;
isLoaded_  = false;
}
int colorIndex(uchar4 color)    
{
for (int i = 0; i < numColors_; i++)
{
#if defined(SYCL_LANGUAGE_VERSION)
if (colors_[i].x() == color.x() && colors_[i].y() == color.y() &&
colors_[i].z() == color.z() && colors_[i].w() == color.w())
#else
if (colors_[i].x == color.x && colors_[i].y == color.y &&
colors_[i].z == color.z && colors_[i].w == color.w)
#endif
{
return i;
}
}
return SDK_SUCCESS;
}
public:


SDKBitMap() :
pixels_(NULL), numColors_(0), colors_(NULL), isLoaded_(false) {}


SDKBitMap(const char * filename) :
pixels_(NULL), numColors_(0), colors_(NULL), isLoaded_(false) 
{
load(filename);
}


SDKBitMap(const SDKBitMap& rhs)
{
*this = rhs;
}


~SDKBitMap()
{
releaseResources();
}


SDKBitMap& operator=(const SDKBitMap& rhs)
{
if (this == &rhs)
{
return *this;
}
id             = rhs.id;
size           = rhs.size;
reserved1      = rhs.reserved1;
reserved2      = rhs.reserved2;
offset         = rhs.offset;
sizeInfo       = rhs.sizeInfo;
width          = rhs.width;
height         = rhs.height;
planes         = rhs.planes;
bitsPerPixel   = rhs.bitsPerPixel;
compression    = rhs.compression;
imageSize      = rhs.imageSize;
xPelsPerMeter  = rhs.xPelsPerMeter;
yPelsPerMeter  = rhs.yPelsPerMeter;
clrUsed        = rhs.clrUsed;
clrImportant   = rhs.clrImportant;
numColors_     = rhs.numColors_;
isLoaded_      = rhs.isLoaded_;
pixels_        = NULL;
colors_        = NULL;
if (isLoaded_)
{
if (rhs.colors_ != NULL)
{
colors_ = new ColorPalette[numColors_];
if (colors_ == NULL)
{
isLoaded_ = false;
return *this;
}
memcpy(colors_, rhs.colors_, numColors_ * sizeof(ColorPalette));
}
pixels_ = new uchar4[width * height];
if (pixels_ == NULL)
{
delete[] colors_;
colors_   = NULL;
isLoaded_ = false;
return *this;
}
memcpy(pixels_, rhs.pixels_, width * height * sizeof(uchar4));
}
return *this;
}


void
load(const char * filename)
{
size_t val;
releaseResources();
FILE * fd = fopen(filename, "rb");
if (fd != NULL)
{
val = fread((BitMapHeader *)this, sizeof(BitMapHeader), 1, fd);
if (val != 1) 
{
fclose(fd);
return;
}
if (id != bitMapID)
{
fclose(fd);
return;
}
val = fread((BitMapInfoHeader *)this, sizeof(BitMapInfoHeader), 1, fd);
if (val != 1) 
{
fclose(fd);
return;
}

if (compression)
{
fclose(fd);
return;
}
if (bitsPerPixel < 8)
{
fclose(fd);
return;
}
numColors_ = 1 << bitsPerPixel;
if(bitsPerPixel == 8)
{
colors_ = new ColorPalette[numColors_];
if (colors_ == NULL)
{
fclose(fd);
return;
}
val  = fread(
(char *)colors_,
numColors_ * sizeof(ColorPalette),
1,
fd);

if (val != 1) 
{
fclose(fd);
return;
}

}
unsigned int sizeBuffer = size - offset;
unsigned char * tmpPixels = new unsigned char[sizeBuffer];
if (tmpPixels == NULL)
{
delete colors_;
colors_ = NULL;
fclose(fd);
return;
}
val = fread(tmpPixels, sizeBuffer * sizeof(unsigned char), 1, fd);
if (val != 1) 
{
delete colors_;
colors_ = NULL;
delete[] tmpPixels;
fclose(fd);
return;
}
pixels_ = new uchar4[width * height];
if (pixels_ == NULL)
{
delete colors_;
colors_ = NULL;
delete[] tmpPixels;
fclose(fd);
return;
}
memset(pixels_, 0xff, width * height * sizeof(uchar4));
unsigned int index = 0;
for(int y = 0; y < height; y++)
{
for(int x = 0; x < width; x++)
{
if (bitsPerPixel == 8)
{
pixels_[(y * width + x)] = colors_[tmpPixels[index++]];
}
else   
{
#if defined(SYCL_LANGUAGE_VERSION)
pixels_[(y * width + x)].z() = tmpPixels[index++];
pixels_[(y * width + x)].y() = tmpPixels[index++];
pixels_[(y * width + x)].x() = tmpPixels[index++];
#else
pixels_[(y * width + x)].z = tmpPixels[index++];
pixels_[(y * width + x)].y = tmpPixels[index++];
pixels_[(y * width + x)].x = tmpPixels[index++];
#endif
}
}
for(int x = 0; x < (4 - (3 * width) % 4) % 4; x++)
{
index++;
}
}
fclose(fd);
delete[] tmpPixels;
isLoaded_  = true;
}
else 
{
fprintf(stderr, "Failed to load file %s\n", filename);
}
}


bool
write(const char * filename)
{
if (!isLoaded_)
{
return false;
}
FILE * fd = fopen(filename, "wb");
if (fd != NULL)
{
fwrite((BitMapHeader *)this, sizeof(BitMapHeader), 1, fd);
if (ferror(fd))
{
fclose(fd);
return false;
}
fwrite((BitMapInfoHeader *)this, sizeof(BitMapInfoHeader), 1, fd);
if (ferror(fd))
{
fclose(fd);
return false;
}
if(bitsPerPixel == 8)
{
fwrite(
(char *)colors_,
numColors_ * sizeof(ColorPalette),
1,
fd);
if (ferror(fd))
{
fclose(fd);
return false;
}
}
for(int y = 0; y < height; y++)
{
for(int x = 0; x < width; x++)
{
if (bitsPerPixel == 8)
{
fputc(
colorIndex(
pixels_[(y * width + x)]),
fd);
}
else   
{
#if defined(SYCL_LANGUAGE_VERSION)
fputc(pixels_[(y * width + x)].z(), fd);
fputc(pixels_[(y * width + x)].y(), fd);
fputc(pixels_[(y * width + x)].x(), fd);
#else
fputc(pixels_[(y * width + x)].z, fd);
fputc(pixels_[(y * width + x)].y, fd);
fputc(pixels_[(y * width + x)].x, fd);
#endif
if (ferror(fd))
{
fclose(fd);
return false;
}
}
}
for(int x = 0; x < (4 - (3 * width) % 4) % 4; x++)
{
fputc(0, fd);
}
}
return true;
}
return false;
}

bool
write(const char * filename, int width, int height, unsigned int *ptr)
{
FILE * fd = fopen(filename, "wb");
int alignSize  = width * 4;
alignSize ^= 0x03;
alignSize ++;
alignSize &= 0x03;
int rowLength = width * 4 + alignSize;
if (fd != NULL)
{
BitMapHeader *bitMapHeader = new BitMapHeader;
bitMapHeader->id = bitMapID;
bitMapHeader->offset = sizeof(BitMapHeader) + sizeof(BitMapInfoHeader);
bitMapHeader->reserved1 = 0x0000;
bitMapHeader->reserved2 = 0x0000;
bitMapHeader->size = sizeof(BitMapHeader) + sizeof(BitMapInfoHeader) + rowLength
* height;
fwrite(bitMapHeader, sizeof(BitMapHeader), 1, fd);
if (ferror(fd))
{
fclose(fd);
return false;
}
BitMapInfoHeader *bitMapInfoHeader = new BitMapInfoHeader;
bitMapInfoHeader->bitsPerPixel = 32;
bitMapInfoHeader->clrImportant = 0;
bitMapInfoHeader->clrUsed = 0;
bitMapInfoHeader->compression = 0;
bitMapInfoHeader->height = height;
bitMapInfoHeader->imageSize = rowLength * height;
bitMapInfoHeader->planes = 1;
bitMapInfoHeader->sizeInfo = sizeof(BitMapInfoHeader);
bitMapInfoHeader->width = width;
bitMapInfoHeader->xPelsPerMeter = 0;
bitMapInfoHeader->yPelsPerMeter = 0;
fwrite(bitMapInfoHeader, sizeof(BitMapInfoHeader), 1, fd);
if (ferror(fd))
{
fclose(fd);
return false;
}
unsigned char buffer[4];
int x, y;
for (y = 0; y < height; y++)
{
for (x = 0; x < width; x++, ptr++)
{
if( 4 != fwrite(ptr, 1, 4, fd))
{
fclose(fd);
return false;
}
}
memset( buffer, 0x00, 4 );
fwrite( buffer, 1, alignSize, fd );
}
fclose( fd );
return true;
}
return false;
}


int
getWidth(void) const
{
if (isLoaded_)
{
return width;
}
else
{
return -1;
}
}



int getNumChannels()
{
if (isLoaded_)
{
return bitsPerPixel / 8;
}
else
{
return SDK_FAILURE;
}
}


int
getHeight(void) const
{
if (isLoaded_)
{
return height;
}
else
{
return -1;
}
}


uchar4 * getPixels(void) const
{
return pixels_;
}


bool
isLoaded(void) const
{
return isLoaded_;
}

};
#pragma pack(pop)
#endif 
