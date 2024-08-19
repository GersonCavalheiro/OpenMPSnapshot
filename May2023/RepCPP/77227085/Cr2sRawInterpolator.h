

#pragma once

#include "adt/Array2DRef.h" 
#include <array>            
#include <cstdint>          

namespace rawspeed {

class RawImage;

class Cr2sRawInterpolator final {
const RawImage& mRaw;

const Array2DRef<const uint16_t> input;
std::array<int, 3> sraw_coeffs;
int hue;

struct YCbCr;

public:
Cr2sRawInterpolator(const RawImage& mRaw_, Array2DRef<const uint16_t> input_,
std::array<int, 3> sraw_coeffs_, int hue_)
: mRaw(mRaw_), input(input_), sraw_coeffs(sraw_coeffs_), hue(hue_) {}

void interpolate(int version);

private:
template <int version> inline void YUV_TO_RGB(const YCbCr& p, uint16_t* X);

static inline void STORE_RGB(uint16_t* X, int r, int g, int b);

template <int version> void interpolate_422_row(int row);
template <int version> void interpolate_422();

template <int version> void interpolate_420_row(int row);
template <int version> void interpolate_420();
};

} 
