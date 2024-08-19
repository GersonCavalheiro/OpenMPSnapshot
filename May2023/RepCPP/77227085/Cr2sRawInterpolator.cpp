

#include "rawspeedconfig.h" 
#include "interpolators/Cr2sRawInterpolator.h"
#include "adt/Array2DRef.h"               
#include "adt/Invariant.h"                
#include "adt/Point.h"                    
#include "common/Common.h"                
#include "common/RawImage.h"              
#include "decoders/RawDecoderException.h" 
#include <array>                          
#include <cstddef>                        
#include <cstdint>                        

namespace rawspeed {

struct Cr2sRawInterpolator::YCbCr final {
int Y = 0;
int Cb = 0;
int Cr = 0;

inline static void LoadY(YCbCr* p, const uint16_t* data) {
invariant(p);
invariant(data);

p->Y = data[0];
}

inline static void LoadCbCr(YCbCr* p, const uint16_t* data) {
invariant(p);
invariant(data);

p->Cb = data[0];
p->Cr = data[1];
}

inline static void CopyCbCr(YCbCr* p, const YCbCr& pSrc) {
invariant(p);

p->Cb = pSrc.Cb;
p->Cr = pSrc.Cr;
}

YCbCr() = default;

inline void signExtend() {
Cb -= 16384;
Cr -= 16384;
}

inline void applyHue(int hue_) {
Cb += hue_;
Cr += hue_;
}

inline void process(int hue_) {
signExtend();
applyHue(hue_);
}

inline void interpolateCbCr(const YCbCr& p0, const YCbCr& p2) {
Cb = (p0.Cb + p2.Cb) >> 1;
Cr = (p0.Cr + p2.Cr) >> 1;
}

inline void interpolateCbCr(const YCbCr& p0, const YCbCr& p1, const YCbCr& p2,
const YCbCr& p3) {
Cb = (p0.Cb + p1.Cb + p2.Cb + p3.Cb) >> 2;
Cr = (p0.Cr + p1.Cr + p2.Cr + p3.Cr) >> 2;
}
};

template <int version> void Cr2sRawInterpolator::interpolate_422_row(int row) {
const Array2DRef<uint16_t> out(mRaw->getU16DataAsUncroppedArray2DRef());

constexpr int InputComponentsPerMCU = 4;
constexpr int PixelsPerMCU = 2;
constexpr int YsPerMCU = PixelsPerMCU;
constexpr int ComponentsPerPixel = 3;
constexpr int OutputComponentsPerMCU = ComponentsPerPixel * PixelsPerMCU;

invariant(input.width % InputComponentsPerMCU == 0);
int numMCUs = input.width / InputComponentsPerMCU;
invariant(numMCUs > 1);

using MCUTy = std::array<YCbCr, PixelsPerMCU>;

auto LoadMCU = [input_ = input, row](int MCUIdx) {
MCUTy MCU;
for (int YIdx = 0; YIdx < PixelsPerMCU; ++YIdx)
YCbCr::LoadY(&MCU[YIdx],
&input_(row, InputComponentsPerMCU * MCUIdx + YIdx));
YCbCr::LoadCbCr(&MCU[0],
&input_(row, InputComponentsPerMCU * MCUIdx + YsPerMCU));
return MCU;
};
auto StoreMCU = [this, out, row](const MCUTy& MCU, int MCUIdx) {
for (int Pixel = 0; Pixel < PixelsPerMCU; ++Pixel) {
YUV_TO_RGB<version>(MCU[Pixel],
&out(row, OutputComponentsPerMCU * MCUIdx +
ComponentsPerPixel * Pixel));
}
};


int MCUIdx;
for (MCUIdx = 0; MCUIdx < numMCUs - 1; ++MCUIdx) {
invariant(MCUIdx + 1 <= numMCUs);

std::array<MCUTy, 2> MCUs;
for (size_t SubMCUIdx = 0; SubMCUIdx < MCUs.size(); ++SubMCUIdx)
MCUs[SubMCUIdx] = LoadMCU(MCUIdx + SubMCUIdx);

MCUs[0][0].process(hue);
MCUs[1][0].process(hue);
MCUs[0][1].interpolateCbCr(MCUs[0][0], MCUs[1][0]);

StoreMCU(MCUs[0], MCUIdx);
}

invariant(MCUIdx + 1 == numMCUs);


MCUTy MCU = LoadMCU(MCUIdx);

MCU[0].process(hue);
YCbCr::CopyCbCr(&MCU[1], MCU[0]);

StoreMCU(MCU, MCUIdx);
}

template <int version> void Cr2sRawInterpolator::interpolate_422() {
const Array2DRef<uint16_t> out(mRaw->getU16DataAsUncroppedArray2DRef());
invariant(out.width > 0);
invariant(out.height > 0);

for (int row = 0; row < out.height; row++)
interpolate_422_row<version>(row);
}

template <int version> void Cr2sRawInterpolator::interpolate_420_row(int row) {
const Array2DRef<uint16_t> out(mRaw->getU16DataAsUncroppedArray2DRef());

constexpr int X_S_F = 2;
constexpr int Y_S_F = 2;
constexpr int PixelsPerMCU = X_S_F * Y_S_F;
constexpr int InputComponentsPerMCU = 2 + PixelsPerMCU;

constexpr int YsPerMCU = PixelsPerMCU;
constexpr int ComponentsPerPixel = 3;
constexpr int OutputComponentsPerMCU = ComponentsPerPixel * PixelsPerMCU;

invariant(input.width % InputComponentsPerMCU == 0);
int numMCUs = input.width / InputComponentsPerMCU;
invariant(numMCUs > 1);

using MCUTy = std::array<std::array<YCbCr, X_S_F>, Y_S_F>;

auto LoadMCU = [input_ = input](int Row, int MCUIdx)
__attribute__((always_inline)) {
MCUTy MCU;
for (int MCURow = 0; MCURow < Y_S_F; ++MCURow) {
for (int MCUCol = 0; MCUCol < X_S_F; ++MCUCol) {
YCbCr::LoadY(&MCU[MCURow][MCUCol],
&input_(Row, InputComponentsPerMCU * MCUIdx +
X_S_F * MCURow + MCUCol));
}
}
YCbCr::LoadCbCr(&MCU[0][0],
&input_(Row, InputComponentsPerMCU * MCUIdx + YsPerMCU));
return MCU;
};
auto StoreMCU = [ this, out ](const MCUTy& MCU, int MCUIdx, int Row)
__attribute__((always_inline)) {
for (int MCURow = 0; MCURow < Y_S_F; ++MCURow) {
for (int MCUCol = 0; MCUCol < X_S_F; ++MCUCol) {
YUV_TO_RGB<version>(
MCU[MCURow][MCUCol],
&out(2 * Row + MCURow, ((OutputComponentsPerMCU * MCUIdx) / Y_S_F) +
ComponentsPerPixel * MCUCol));
}
}
};

invariant(row + 1 <= input.height);


int MCUIdx;
for (MCUIdx = 0; MCUIdx < numMCUs - 1; ++MCUIdx) {
invariant(MCUIdx + 1 <= numMCUs);

std::array<std::array<MCUTy, 2>, 2> MCUs;
for (int Row = 0; Row < 2; ++Row)
for (int Col = 0; Col < 2; ++Col)
MCUs[Row][Col] = LoadMCU(row + Row, MCUIdx + Col);

for (int Row = 0; Row < 2; ++Row)
for (int Col = 0; Col < 2; ++Col)
MCUs[Row][Col][0][0].process(hue);

MCUs[0][0][0][1].interpolateCbCr(MCUs[0][0][0][0], MCUs[0][1][0][0]);

MCUs[0][0][1][0].interpolateCbCr(MCUs[0][0][0][0], MCUs[1][0][0][0]);

MCUs[0][0][1][1].interpolateCbCr(MCUs[0][0][0][0], MCUs[0][1][0][0],
MCUs[1][0][0][0], MCUs[1][1][0][0]);


StoreMCU(MCUs[0][0], MCUIdx, row);
}

invariant(MCUIdx + 1 == numMCUs);


std::array<MCUTy, 2> MCUs;
for (int Row = 0; Row < 2; ++Row)
MCUs[Row] = LoadMCU(row + Row, MCUIdx);

for (int Row = 0; Row < 2; ++Row)
MCUs[Row][0][0].process(hue);

MCUs[0][1][0].interpolateCbCr(MCUs[0][0][0], MCUs[1][0][0]);

for (int Row = 0; Row < 2; ++Row)
YCbCr::CopyCbCr(&MCUs[0][Row][1], MCUs[0][Row][0]);

StoreMCU(MCUs[0], MCUIdx, row);
}

template <int version> void Cr2sRawInterpolator::interpolate_420() {
const Array2DRef<uint16_t> out(mRaw->getU16DataAsUncroppedArray2DRef());

constexpr int X_S_F = 2;
constexpr int Y_S_F = 2;
constexpr int PixelsPerMCU = X_S_F * Y_S_F;
constexpr int InputComponentsPerMCU = 2 + PixelsPerMCU;

constexpr int YsPerMCU = PixelsPerMCU;
constexpr int ComponentsPerPixel = 3;
constexpr int OutputComponentsPerMCU = ComponentsPerPixel * PixelsPerMCU;

invariant(input.width % InputComponentsPerMCU == 0);
int numMCUs = input.width / InputComponentsPerMCU;
invariant(numMCUs > 1);

using MCUTy = std::array<std::array<YCbCr, X_S_F>, Y_S_F>;

auto LoadMCU = [input_ = input](int Row, int MCUIdx)
__attribute__((always_inline)) {
MCUTy MCU;
for (int MCURow = 0; MCURow < Y_S_F; ++MCURow) {
for (int MCUCol = 0; MCUCol < X_S_F; ++MCUCol) {
YCbCr::LoadY(&MCU[MCURow][MCUCol],
&input_(Row, InputComponentsPerMCU * MCUIdx +
X_S_F * MCURow + MCUCol));
}
}
YCbCr::LoadCbCr(&MCU[0][0],
&input_(Row, InputComponentsPerMCU * MCUIdx + YsPerMCU));
return MCU;
};
auto StoreMCU = [ this, out ](const MCUTy& MCU, int MCUIdx, int Row)
__attribute__((always_inline)) {
for (int MCURow = 0; MCURow < Y_S_F; ++MCURow) {
for (int MCUCol = 0; MCUCol < X_S_F; ++MCUCol) {
YUV_TO_RGB<version>(
MCU[MCURow][MCUCol],
&out(2 * Row + MCURow, ((OutputComponentsPerMCU * MCUIdx) / Y_S_F) +
ComponentsPerPixel * MCUCol));
}
}
};

int row = 0;
#ifdef HAVE_OPENMP
#pragma omp parallel for default(none) schedule(static)                        \
num_threads(rawspeed_get_number_of_processor_cores()) firstprivate(out)    \
lastprivate(row)
#endif
for (row = 0; row < input.height - 1; ++row)
interpolate_420_row<version>(row);

invariant(row + 1 == input.height);


int MCUIdx;
for (MCUIdx = 0; MCUIdx < numMCUs - 1; ++MCUIdx) {
invariant(MCUIdx + 1 < numMCUs);

std::array<std::array<MCUTy, 2>, 1> MCUs;
for (int Row = 0; Row < 1; ++Row)
for (int Col = 0; Col < 2; ++Col)
MCUs[Row][Col] = LoadMCU(row + Row, MCUIdx + Col);

for (int Row = 0; Row < 1; ++Row)
for (int Col = 0; Col < 2; ++Col)
MCUs[Row][Col][0][0].process(hue);

MCUs[0][0][0][1].interpolateCbCr(MCUs[0][0][0][0], MCUs[0][1][0][0]);

for (int Col = 0; Col < 2; ++Col)
YCbCr::CopyCbCr(&MCUs[0][0][1][Col], MCUs[0][0][0][Col]);

StoreMCU(MCUs[0][0], MCUIdx, row);
}

invariant(MCUIdx + 1 == numMCUs);


MCUTy MCU = LoadMCU(row, MCUIdx);

MCU[0][0].process(hue);

for (int Row = 0; Row < 2; ++Row)
for (int Col = 0; Col < 2; ++Col)
YCbCr::CopyCbCr(&MCU[Row][Col], MCU[0][0]);

StoreMCU(MCU, MCUIdx, row);
}

inline void Cr2sRawInterpolator::STORE_RGB(uint16_t* X, int r, int g, int b) {
invariant(X);

X[0] = clampBits(r >> 8, 16);
X[1] = clampBits(g >> 8, 16);
X[2] = clampBits(b >> 8, 16);
}

template <>

inline void Cr2sRawInterpolator::YUV_TO_RGB<0>(const YCbCr& p, uint16_t* X) {
invariant(X);

int r = sraw_coeffs[0] * (p.Y + p.Cr - 512);
int g = sraw_coeffs[1] * (p.Y + ((-778 * p.Cb - (p.Cr * 2048)) >> 12) - 512);
int b = sraw_coeffs[2] * (p.Y + (p.Cb - 512));
STORE_RGB(X, r, g, b);
}

template <>
inline void Cr2sRawInterpolator::YUV_TO_RGB<1>(const YCbCr& p, uint16_t* X) {
invariant(X);

int r = sraw_coeffs[0] * (p.Y + ((50 * p.Cb + 22929 * p.Cr) >> 12));
int g = sraw_coeffs[1] * (p.Y + ((-5640 * p.Cb - 11751 * p.Cr) >> 12));
int b = sraw_coeffs[2] * (p.Y + ((29040 * p.Cb - 101 * p.Cr) >> 12));
STORE_RGB(X, r, g, b);
}

template <>

inline void Cr2sRawInterpolator::YUV_TO_RGB<2>(const YCbCr& p, uint16_t* X) {
invariant(X);

int r = sraw_coeffs[0] * (p.Y + p.Cr);
int g = sraw_coeffs[1] * (p.Y + ((-778 * p.Cb - (p.Cr * 2048)) >> 12));
int b = sraw_coeffs[2] * (p.Y + p.Cb);
STORE_RGB(X, r, g, b);
}

void Cr2sRawInterpolator::interpolate(int version) {
invariant(version >= 0 && version <= 2);

const auto& subSampling = mRaw->metadata.subsampling;
if (subSampling.y == 1 && subSampling.x == 2) {
switch (version) {
case 0:
interpolate_422<0>();
break;
case 1:
interpolate_422<1>();
break;
case 2:
interpolate_422<2>();
break;
default:
__builtin_unreachable();
}
} else if (subSampling.y == 2 && subSampling.x == 2) {
switch (version) {
case 1:
interpolate_420<1>();
break;
case 2:
interpolate_420<2>();
break;
default:
__builtin_unreachable();
}
} else
ThrowRDE("Unknown subsampling: (%i; %i)", subSampling.x, subSampling.y);
}

} 
