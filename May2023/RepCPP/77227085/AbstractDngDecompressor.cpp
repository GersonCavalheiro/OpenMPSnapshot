

#include "rawspeedconfig.h" 
#include "decompressors/AbstractDngDecompressor.h"
#include "adt/Invariant.h"                          
#include "adt/Point.h"                              
#include "common/Common.h"                          
#include "common/RawImage.h"                        
#include "decoders/RawDecoderException.h"           
#include "decompressors/DeflateDecompressor.h"      
#include "decompressors/JpegDecompressor.h"         
#include "decompressors/LJpegDecoder.h"             
#include "decompressors/UncompressedDecompressor.h" 
#include "decompressors/VC5Decompressor.h"          
#include "io/ByteStream.h"                          
#include "io/Endianness.h"                          
#include "io/IOException.h"                         
#include <limits>                                   
#include <memory>                                   
#include <string>                                   
#include <vector>                                   

namespace rawspeed {

template <> void AbstractDngDecompressor::decompressThread<1>() const noexcept {
#ifdef HAVE_OPENMP
#pragma omp for schedule(static)
#endif
for (auto e = slices.cbegin(); e < slices.cend(); ++e) {

iPoint2D tileSize(e->width, e->height);
iPoint2D pos(e->offX, e->offY);

bool big_endian = e->bs.getByteOrder() == Endianness::big;

switch (mBps) {
case 8:
case 16:
case 32:
break;
default:
if (mRaw->getDataType() == RawImageType::UINT16)
big_endian = true;
break;
}

try {
const uint32_t inputPixelBits = mRaw->getCpp() * mBps;

if (e->dsc.tileW > std::numeric_limits<int>::max() / inputPixelBits)
ThrowIOE("Integer overflow when calculating input pitch");

const int inputPitchBits = inputPixelBits * e->dsc.tileW;
invariant(inputPitchBits > 0);

if (inputPitchBits % 8 != 0) {
ThrowRDE("Bad combination of cpp (%u), bps (%u) and width (%u), the "
"pitch is %u bits, which is not a multiple of 8 (1 byte)",
mRaw->getCpp(), mBps, e->width, inputPitchBits);
}

const int inputPitch = inputPitchBits / 8;
if (inputPitch == 0)
ThrowRDE("Data input pitch is too short. Can not decode!");

UncompressedDecompressor decompressor(
e->bs, mRaw, iRectangle2D(pos, tileSize), inputPitch, mBps,
big_endian ? BitOrder::MSB : BitOrder::LSB);
decompressor.readUncompressedRaw();
} catch (const RawDecoderException& err) {
mRaw->setError(err.what());
} catch (const IOException& err) {
mRaw->setError(err.what());
}
}
}

template <> void AbstractDngDecompressor::decompressThread<7>() const noexcept {
#ifdef HAVE_OPENMP
#pragma omp for schedule(static)
#endif
for (auto e = slices.cbegin(); e < slices.cend(); ++e) {
try {
LJpegDecoder d(e->bs, mRaw);
d.decode(e->offX, e->offY, e->width, e->height, mFixLjpeg);
} catch (const RawDecoderException& err) {
mRaw->setError(err.what());
} catch (const IOException& err) {
mRaw->setError(err.what());
}
}
}

#ifdef HAVE_ZLIB
template <> void AbstractDngDecompressor::decompressThread<8>() const noexcept {
std::unique_ptr<unsigned char[]> uBuffer; 

#ifdef HAVE_OPENMP
#pragma omp for schedule(static)
#endif
for (auto e = slices.cbegin(); e < slices.cend(); ++e) {
DeflateDecompressor z(e->bs.peekBuffer(e->bs.getRemainSize()), mRaw,
mPredictor, mBps);
try {
z.decode(&uBuffer, iPoint2D(mRaw->getCpp() * e->dsc.tileW, e->dsc.tileH),
iPoint2D(mRaw->getCpp() * e->width, e->height),
iPoint2D(mRaw->getCpp() * e->offX, e->offY));
} catch (const RawDecoderException& err) {
mRaw->setError(err.what());
} catch (const IOException& err) {
mRaw->setError(err.what());
}
}
}
#endif

template <> void AbstractDngDecompressor::decompressThread<9>() const noexcept {
#ifdef HAVE_OPENMP
#pragma omp for schedule(static)
#endif
for (auto e = slices.cbegin(); e < slices.cend(); ++e) {
try {
VC5Decompressor d(e->bs, mRaw);
d.decode(e->offX, e->offY, e->width, e->height);
} catch (const RawDecoderException& err) {
mRaw->setError(err.what());
} catch (const IOException& err) {
mRaw->setError(err.what());
}
}
}

#ifdef HAVE_JPEG
template <>
void AbstractDngDecompressor::decompressThread<0x884c>() const noexcept {
#ifdef HAVE_OPENMP
#pragma omp for schedule(static)
#endif
for (auto e = slices.cbegin(); e < slices.cend(); ++e) {
JpegDecompressor j(e->bs.peekBuffer(e->bs.getRemainSize()), mRaw);
try {
j.decode(e->offX, e->offY);
} catch (const RawDecoderException& err) {
mRaw->setError(err.what());
} catch (const IOException& err) {
mRaw->setError(err.what());
}
}
}
#endif

void AbstractDngDecompressor::decompressThread() const noexcept {
invariant(mRaw->dim.x > 0);
invariant(mRaw->dim.y > 0);
invariant(mRaw->getCpp() > 0 && mRaw->getCpp() <= 4);
invariant(mBps > 0 && mBps <= 32);

if (compression == 1) {

decompressThread<1>();
} else if (compression == 7) {

decompressThread<7>();
} else if (compression == 8) {

#ifdef HAVE_ZLIB
decompressThread<8>();
#else
#pragma message                                                                \
"ZLIB is not present! Deflate compression will not be supported!"
mRaw->setError("deflate support is disabled.");
#endif
} else if (compression == 9) {

decompressThread<9>();
} else if (compression == 0x884c) {

#ifdef HAVE_JPEG
decompressThread<0x884c>();
#else
#pragma message "JPEG is not present! Lossy JPEG DNG will not be supported!"
mRaw->setError("jpeg support is disabled.");
#endif
} else
mRaw->setError("AbstractDngDecompressor: Unknown compression");
}

void AbstractDngDecompressor::decompress() const {
#ifdef HAVE_OPENMP
#pragma omp parallel default(none) num_threads(                                \
rawspeed_get_number_of_processor_cores()) if (slices.size() > 1)
#endif
decompressThread();

std::string firstErr;
if (mRaw->isTooManyErrors(1, &firstErr)) {
ThrowRDE("Too many errors encountered. Giving up. First Error:\n%s",
firstErr.c_str());
}
}

} 
