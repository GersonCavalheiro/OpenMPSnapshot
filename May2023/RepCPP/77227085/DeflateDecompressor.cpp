

#include "rawspeedconfig.h"        
#include "adt/CroppedArray1DRef.h" 
#include "adt/CroppedArray2DRef.h" 
#include "adt/Invariant.h"         
#include "common/Common.h"         
#include <array>                   
#include <climits>                 

#ifdef HAVE_ZLIB

#include "adt/Point.h"                    
#include "common/FloatingPoint.h"         
#include "decoders/RawDecoderException.h" 
#include "decompressors/DeflateDecompressor.h"
#include "io/Endianness.h" 
#include <cstdint>         
#include <cstdio>          
#include <zlib.h>          

namespace rawspeed {

DeflateDecompressor::DeflateDecompressor(Buffer bs, const RawImage& img,
int predictor, int bps_)
: input(bs), mRaw(img), bps(bps_) {
switch (predictor) {
case 3:
predFactor = 1;
break;
case 34894:
predFactor = 2;
break;
case 34895:
predFactor = 4;
break;
default:
ThrowRDE("Unsupported predictor %i", predictor);
}
predFactor *= mRaw->getCpp();
}

static inline void decodeDeltaBytes(unsigned char* src, size_t realTileWidth,
unsigned int bytesps, int factor) {
for (size_t col = factor; col < realTileWidth * bytesps; ++col) {
src[col] = static_cast<unsigned char>(src[col] + src[col - factor]);
}
}

template <typename T> struct StorageType {};
template <> struct StorageType<ieee_754_2008::Binary16> {
using type = uint16_t;
static constexpr int padding_bytes = 0;
};
template <> struct StorageType<ieee_754_2008::Binary24> {
using type = uint32_t;
static constexpr int padding_bytes = 1;
};
template <> struct StorageType<ieee_754_2008::Binary32> {
using type = uint32_t;
static constexpr int padding_bytes = 0;
};

template <typename T>
static inline void decodeFPDeltaRow(const unsigned char* src,
size_t realTileWidth,
CroppedArray1DRef<float> out) {
using storage_type = typename StorageType<T>::type;
constexpr unsigned storage_bytes = sizeof(storage_type);
constexpr unsigned bytesps = T::StorageWidth / 8;

for (int col = 0; col < out.size(); ++col) {
std::array<unsigned char, storage_bytes> bytes;
for (int c = 0; c != bytesps; ++c)
bytes[c] = src[col + c * realTileWidth];

auto tmp = getBE<storage_type>(bytes.data());
tmp >>= CHAR_BIT * StorageType<T>::padding_bytes;

uint32_t tmp_expanded;
switch (bytesps) {
case 2:
case 3:
tmp_expanded = extendBinaryFloatingPoint<T, ieee_754_2008::Binary32>(tmp);
break;
case 4:
tmp_expanded = tmp;
break;
}

out(col) = bit_cast<float>(tmp_expanded);
}
}

void DeflateDecompressor::decode(
std::unique_ptr<unsigned char[]>* uBuffer, 
iPoint2D maxDim, iPoint2D dim, iPoint2D off) {
int bytesps = bps / 8;
invariant(bytesps >= 2 && bytesps <= 4);

uLongf dstLen = bytesps * maxDim.area();

if (!*uBuffer)
*uBuffer =
std::unique_ptr<unsigned char[]>(new unsigned char[dstLen]); 

if (int err =
uncompress(uBuffer->get(), &dstLen, input.begin(), input.getSize());
err != Z_OK) {
ThrowRDE("failed to uncompress tile: %d (%s)", err, zError(err));
}

const auto out = CroppedArray2DRef(
mRaw->getF32DataAsUncroppedArray2DRef(), off.x,
off.y, dim.x, dim.y);

for (int row = 0; row < out.croppedHeight; ++row) {
unsigned char* src = uBuffer->get() + row * maxDim.x * bytesps;

decodeDeltaBytes(src, maxDim.x, bytesps, predFactor);

switch (bytesps) {
case 2:
decodeFPDeltaRow<ieee_754_2008::Binary16>(src, maxDim.x, out[row]);
break;
case 3:
decodeFPDeltaRow<ieee_754_2008::Binary24>(src, maxDim.x, out[row]);
break;
case 4:
decodeFPDeltaRow<ieee_754_2008::Binary32>(src, maxDim.x, out[row]);
break;
}
}
}

} 

#else

#pragma message                                                                \
"ZLIB is not present! Deflate compression will not be supported!"

#endif
