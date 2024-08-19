#include <iostream>
#include <memory.h>
#include <cassert>
#include <iomanip>
#include <cmath>
#include "FreeImagePlus.h"
#include "Stopwatch.h"

using namespace std;

#ifndef WIN32
typedef DWORD COLORREF;


#endif

const int bypp = 4;
const int fSize = 3;
const int fSize2 = fSize / 2;
const int hFilter[][fSize] = {
{ 1, 1, 1 },
{ 0, 0, 0 },
{ -1,-1,-1 }
};
const int vFilter[][fSize] = {
{ 1, 0,-1 },
{ 1, 0,-1 },
{ 1, 0,-1 }
};

static BYTE dist(int x, int y) {
int d = (int)sqrt(x*x + y*y);
return (d < 256) ? d : 255;
}

static void processSerial(const fipImage& input, fipImage& output) {
assert(input.getWidth() == output.getWidth() && input.getHeight() == output.getHeight() && input.getImageSize() == output.getImageSize());
assert(input.getBitsPerPixel() == 32);

for (unsigned int v = fSize2; v < output.getHeight() - fSize2; v++) {
for (unsigned int u = fSize2; u < output.getWidth() - fSize2; u++) {
RGBQUAD iC;
int hC[3] = { 0, 0, 0 };
int vC[3] = { 0, 0, 0 };

for (unsigned int j = 0; j < fSize; j++) {
for (unsigned int i = 0; i < fSize; i++) {
input.getPixelColor(u + i - fSize2, v + j - fSize2, &iC);
hC[0] += hFilter[j][i]*iC.rgbBlue;
vC[0] += vFilter[j][i]*iC.rgbBlue;
hC[1] += hFilter[j][i]*iC.rgbGreen;
vC[1] += vFilter[j][i]*iC.rgbGreen;
hC[2] += hFilter[j][i]*iC.rgbRed;
vC[2] += vFilter[j][i]*iC.rgbRed;
}
}
RGBQUAD oC = { dist(hC[0], vC[0]), dist(hC[1], vC[1]), dist(hC[2], vC[2]), 255 };
output.setPixelColor(u, v, &oC);
}
}
}

static void processSerialOpt(const fipImage& input, fipImage& output) {

assert(input.getWidth() == output.getWidth() && input.getHeight() == output.getHeight() && input.getImageSize() == output.getImageSize());
assert(input.getBitsPerPixel() == bypp * 8);

const size_t stride = input.getScanWidth();

BYTE *iRow = input.getScanLine(fSize2) + bypp*fSize2;
BYTE *oRow = output.getScanLine(fSize2) + bypp*fSize2;

for (size_t v = fSize2; v < output.getHeight() - fSize2; v++) {
BYTE *iCenter = iRow;
BYTE *oPos = oRow;

for (size_t u = fSize2; u < output.getWidth() - fSize2; u++) {
int hC[3] = { 0, 0, 0 };
int vC[3] = { 0, 0, 0 };

BYTE *iPos = iCenter - fSize2*stride - bypp*fSize2;

for (size_t j = 0; j < fSize; j++) {
for (size_t i = 0; i < fSize; i++) {
RGBQUAD *iC = reinterpret_cast<RGBQUAD*>(iPos);
hC[0] += hFilter[j][i] * iC->rgbBlue;
vC[0] += vFilter[j][i] * iC->rgbBlue;
hC[1] += hFilter[j][i] * iC->rgbGreen;
vC[1] += vFilter[j][i] * iC->rgbGreen;
hC[2] += hFilter[j][i] * iC->rgbRed;
vC[2] += vFilter[j][i] * iC->rgbRed;
iPos += bypp;
}
iPos += stride - bypp*fSize;
}

RGBQUAD *oC = reinterpret_cast<RGBQUAD*>(oPos);
oC->rgbBlue = dist(hC[0], vC[0]);
oC->rgbGreen = dist(hC[1], vC[1]);
oC->rgbRed = dist(hC[2], vC[2]);
oC->rgbReserved = 255;
iCenter += bypp;
oPos += bypp;
}
iRow += stride;
oRow += stride;
}
}

static void processParallel(const fipImage& input, fipImage& output) {
const int bypp = 4;
assert(input.getWidth() == output.getWidth() && input.getHeight() == output.getHeight() && input.getImageSize() == output.getImageSize());
assert(input.getBitsPerPixel() == bypp * 8);

const size_t stride = input.getScanWidth();
const int fSize = 3;
const int fSize2 = fSize / 2;
const int hFilter[][fSize] = {
{ 1, 1, 1 },
{ 0, 0, 0 },
{ -1,-1,-1 }
};
const int vFilter[][fSize] = {
{ 1, 0,-1 },
{ 1, 0,-1 },
{ 1, 0,-1 }
};

#pragma omp parallel for default(none)
for (int v = fSize2; v < (int)output.getHeight() - fSize2; v++) {
BYTE *iCenter = input.getScanLine(v) + bypp*fSize2;
BYTE *oPos = output.getScanLine(v) + bypp*fSize2;

for (size_t u = fSize2; u < output.getWidth() - fSize2; u++) {
int hC[3] = { 0, 0, 0 };
int vC[3] = { 0, 0, 0 };

BYTE *iPos = iCenter - fSize2*stride - bypp*fSize2;
for (size_t j = 0; j < fSize; j++) {
for (size_t i = 0; i < fSize; i++) {
RGBQUAD *iC = reinterpret_cast<RGBQUAD*>(iPos);
hC[0] += hFilter[j][i] * iC->rgbBlue;
vC[0] += vFilter[j][i] * iC->rgbBlue;
hC[1] += hFilter[j][i] * iC->rgbGreen;
vC[1] += vFilter[j][i] * iC->rgbGreen;
hC[2] += hFilter[j][i] * iC->rgbRed;
vC[2] += vFilter[j][i] * iC->rgbRed;
iPos += bypp;
}
iPos += stride - bypp*fSize;
}
RGBQUAD *oC = reinterpret_cast<RGBQUAD*>(oPos);
oC->rgbBlue = dist(hC[0], vC[0]);
oC->rgbGreen = dist(hC[1], vC[1]);
oC->rgbRed = dist(hC[2], vC[2]);
oC->rgbReserved = 255;
iCenter += bypp;
oPos += bypp;
}
}
}



static bool operator==(const fipImage& im1, const fipImage& im2) {
assert(im1.getWidth() == im2.getWidth() && im1.getHeight() == im2.getHeight() && im1.getImageSize() == im2.getImageSize());
assert(im1.getBitsPerPixel() == 32);

for(unsigned int i = 0; i < im1.getHeight(); i++) {
COLORREF *row1 = reinterpret_cast<COLORREF*>(im1.getScanLine(i));
COLORREF *row2 = reinterpret_cast<COLORREF*>(im2.getScanLine(i));
for(unsigned int j = 0; j < im1.getWidth(); j++) {
if (row1[j] != row2[j]) return false;
}
}
return true;
}

static bool operator!=(const fipImage& im1, const fipImage& im2) {
return !(im1 == im2);
}

int imageProcessing(int argc, const char* argv[]) {
cout << "Image processing started" << endl;
if (argc < 3) {
cerr << "Usage: " << argv[0] << " input-file-name output-file-name" << endl;
return -1;
}

Stopwatch sw;
fipImage image;

if (!image.load(argv[1])) {
cerr << "Image not found: " << argv[1] << endl;
return -1;
}

fipImage out1(image), out2(image), out3(image);

cout << "Start sequential process" << endl;
sw.Start();
processSerial(image, out1);
sw.Stop();
double seqTime = sw.GetElapsedTimeMilliseconds();
cout << seqTime << " ms" << endl;

cout << "Start optimized sequential process" << endl;
sw.Start();
processSerialOpt(image, out2);
sw.Stop();
double seqOptTime = sw.GetElapsedTimeMilliseconds();
cout << seqOptTime << " ms, speedup = " << seqTime/seqOptTime << endl;

cout << boolalpha << "The two operations produce the same results: " << (out1 == out2) << endl;

cout << "Start parallel process" << endl;
sw.Start();
processParallel(image, out3);
sw.Stop();
cout << sw.GetElapsedTimeMilliseconds() << " ms, speedup = " << seqOptTime/sw.GetElapsedTimeMilliseconds() << endl;

cout << boolalpha << "The two operations produce the same results: " << (out1 == out3) << endl;

if (!out1.save(argv[2])) {
cerr << "Image not saved: " << argv[2] << endl;
return -1;
}

return 0;
}