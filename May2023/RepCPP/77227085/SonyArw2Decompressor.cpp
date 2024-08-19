

#include "rawspeedconfig.h" 
#include "decompressors/SonyArw2Decompressor.h"
#include "adt/Array2DRef.h"               
#include "adt/Invariant.h"                
#include "adt/Point.h"                    
#include "common/Common.h"                
#include "common/RawImage.h"              
#include "decoders/RawDecoderException.h" 
#include "io/BitPumpLSB.h"                
#include <cstdint>                        
#include <string>                         

namespace rawspeed {

SonyArw2Decompressor::SonyArw2Decompressor(const RawImage& img,
ByteStream input_)
: mRaw(img) {
if (mRaw->getCpp() != 1 || mRaw->getDataType() != RawImageType::UINT16 ||
mRaw->getBpp() != sizeof(uint16_t))
ThrowRDE("Unexpected component count / data type");

if (!mRaw->dim.hasPositiveArea() || mRaw->dim.x % 32 != 0 ||
mRaw->dim.x > 9600 || mRaw->dim.y > 6376)
ThrowRDE("Unexpected image dimensions found: (%u; %u)", mRaw->dim.x,
mRaw->dim.y);

input = input_.peekStream(mRaw->dim.x * mRaw->dim.y);
}

void SonyArw2Decompressor::decompressRow(int row) const {
const Array2DRef<uint16_t> out(mRaw->getU16DataAsUncroppedArray2DRef());
invariant(out.width > 0);
invariant(out.width % 32 == 0);

auto& rawdata = reinterpret_cast<RawImageDataU16&>(*mRaw);

ByteStream rowBs = input;
rowBs.skipBytes(row * out.width);
rowBs = rowBs.peekStream(out.width);

BitPumpLSB bits(rowBs);

uint32_t random = bits.peekBits(24);

for (int col = 0; col < out.width; col += ((col & 1) != 0) ? 31 : 1) {
int _max = bits.getBits(11);
int _min = bits.getBits(11);
int _imax = bits.getBits(4);
int _imin = bits.getBits(4);

if (_imax == _imin)
ThrowRDE("ARW2 invariant failed, same pixel is both min and max");

int sh = 0;
while ((sh < 4) && ((0x80 << sh) <= (_max - _min)))
sh++;

for (int i = 0; i < 16; i++) {
int p;
if (i == _imax)
p = _max;
else {
if (i == _imin)
p = _min;
else {
p = (bits.getBits(7) << sh) + _min;
if (p > 0x7ff)
p = 0x7ff;
}
}
rawdata.setWithLookUp(
p << 1, reinterpret_cast<uint8_t*>(&out(row, col + i * 2)), &random);
}
}
}

void SonyArw2Decompressor::decompressThread() const noexcept {
invariant(mRaw->dim.x > 0);
invariant(mRaw->dim.x % 32 == 0);
invariant(mRaw->dim.y > 0);

#ifdef HAVE_OPENMP
#pragma omp for schedule(static)
#endif
for (int y = 0; y < mRaw->dim.y; y++) {
try {
decompressRow(y);
} catch (const RawspeedException& err) {
mRaw->setError(err.what());
#ifdef HAVE_OPENMP
#pragma omp cancel for
#endif
}
}
}

void SonyArw2Decompressor::decompress() const {
#ifdef HAVE_OPENMP
#pragma omp parallel default(none)                                             \
num_threads(rawspeed_get_number_of_processor_cores())
#endif
decompressThread();

std::string firstErr;
if (mRaw->isTooManyErrors(1, &firstErr)) {
ThrowRDE("Too many errors encountered. Giving up. First Error:\n%s",
firstErr.c_str());
}
}

} 
