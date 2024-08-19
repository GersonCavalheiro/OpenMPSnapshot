#pragma once

#include <cstdio> 
#include <vector> 

#include "status.hxx" 

namespace bitmap {

inline void transfer4Byte(unsigned char target[4], unsigned int const source) {
for(int i = 0; i < 4; ++i) {
target[i] = (unsigned char)(source >> (8*i));
} 
} 

template <typename real_t>
status_t write_bmp_file(char const *basename
, real_t const data 
, unsigned const h, unsigned const w 
, int const stride=-1 
, float const factor=255
, char const *extension=".bmp"
, bool const invert_y=true
, int const echo=1
) {

int const s = (stride < int(w)) ? int(w) : stride;


size_t const filesize = 54 + 3*w*h; 
if (echo > 0) std::printf("# write %d x %d (stride %d) %.3f kByte\n", h, w, s, filesize*1e-3);

int constexpr MaxFilenameLenth = 1024;
char filename[MaxFilenameLenth];
std::snprintf(filename, MaxFilenameLenth, "%s%s", basename, extension);
auto const f = std::fopen(filename, "wb"); 
if (echo > 0) std::printf("# write bitmap to file '%s'\n", filename);
if (nullptr == f) return 1; 

unsigned char bmpfileheader[14] = {'B','M', 0,0,0,0, 0,0, 0,0, 54,0,0,0};
transfer4Byte(bmpfileheader + 2, filesize);
std::fwrite(bmpfileheader, 1, 14, f);

unsigned char bmpinfoheader[40] = {40,0,0,0, 0,0,0,0, 0,0,0,0, 1,0, 24,0};
transfer4Byte(bmpinfoheader + 4, w);
transfer4Byte(bmpinfoheader + 8, h);
std::fwrite(bmpinfoheader, 1, 40, f);

unsigned char const bmppad[3] = {0,0,0}; 
std::vector<unsigned char> img(w*3, 0);
for (int i = 0; i < h; ++i) {
int const y = invert_y ? ((h - 1) - i) : i;
for (int j = 0; j < w; ++j) {
int r = data[(y*s + j)*4 + 0]*factor;
int g = data[(y*s + j)*4 + 1]*factor;
int b = data[(y*s + j)*4 + 2]*factor;
if (r > 255) r=255;
if (g > 255) g=255;
if (b > 255) b=255;
img[j*3 + 2] = (unsigned char)(r);
img[j*3 + 1] = (unsigned char)(g);
img[j*3    ] = (unsigned char)(b);
} 
std::fwrite(img.data(), 3, w, f);
std::fwrite(bmppad, 1, (4-(w*3)%4)%4, f);
} 
std::fclose(f);

return 0;
} 

inline status_t test_image(int const echo=0) {
int const d=64, h=4*d, w=5*d;
std::vector<uint8_t> data(h*w*4, 0);
float const decay = -.5f/(d*d);
for (int rgb = 0; rgb < 3; ++rgb) {
auto const arg = (rgb*2*M_PI)/3.;
int const center_x = w/2 + d/2*std::sin(arg);
int const center_y = h/2 + d/2*std::cos(arg);
for (int y = 0; y < h; ++y) {
auto const y2 = (y - center_y)*(y - center_y);
for (int x = 0; x < w; ++x) {
auto const x2 = (x - center_x)*(x - center_x);
auto const value = std::exp(decay*(x2 + y2)); 
data[(y*w + x)*4 + rgb] = 255*value;
} 
} 
} 
return write_bmp_file("test_file", data, h, w, w, 1);
} 

inline status_t all_tests(int const echo=1) { return test_image(echo); }

} 
