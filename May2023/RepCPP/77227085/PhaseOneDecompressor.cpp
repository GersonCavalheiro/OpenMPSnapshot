

#include "rawspeedconfig.h" 
#include "decompressors/PhaseOneDecompressor.h"
#include "adt/Array2DRef.h"               
#include "adt/Invariant.h"                
#include "adt/Point.h"                    
#include "common/Common.h"                
#include "common/RawImage.h"              
#include "decoders/RawDecoderException.h" 
#include "io/BitPumpMSB32.h"              
#include <algorithm>                      
#include <array>                          
#include <cstdint>                        
#include <memory>                         
#include <string>                         
#include <utility>                        
#include <vector>                         

namespace rawspeed {

PhaseOneDecompressor::PhaseOneDecompressor(const RawImage& img,
std::vector<PhaseOneStrip>&& strips_)
: mRaw(img), strips(std::move(strips_)) {
if (mRaw->getDataType() != RawImageType::UINT16)
ThrowRDE("Unexpected data type");

if (mRaw->getCpp() != 1 || mRaw->getBpp() != sizeof(uint16_t))
ThrowRDE("Unexpected cpp: %u", mRaw->getCpp());

if (!mRaw->dim.hasPositiveArea() || mRaw->dim.x % 2 != 0 ||
mRaw->dim.x > 11976 || mRaw->dim.y > 8854) {
ThrowRDE("Unexpected image dimensions found: (%u; %u)", mRaw->dim.x,
mRaw->dim.y);
}

prepareStrips();
}

void PhaseOneDecompressor::prepareStrips() {

if (strips.size() != static_cast<decltype(strips)::size_type>(mRaw->dim.y)) {
ThrowRDE("Height (%u) vs strip count %zu mismatch", mRaw->dim.y,
strips.size());
}

std::sort(
strips.begin(), strips.end(),
[](const PhaseOneStrip& a, const PhaseOneStrip& b) { return a.n < b.n; });
for (decltype(strips)::size_type i = 0; i < strips.size(); ++i)
if (static_cast<decltype(strips)::size_type>(strips[i].n) != i)
ThrowRDE("Strips validation issue.");
}

void PhaseOneDecompressor::decompressStrip(const PhaseOneStrip& strip) const {
const Array2DRef<uint16_t> out(mRaw->getU16DataAsUncroppedArray2DRef());

invariant(out.width > 0);
invariant(out.width % 2 == 0);

static constexpr std::array<const int, 10> length = {8,  7, 6,  9,  11,
10, 5, 12, 14, 13};

BitPumpMSB32 pump(strip.bs);

std::array<int32_t, 2> pred;
pred.fill(0);
std::array<int, 2> len;
const int row = strip.n;
for (int col = 0; col < out.width; col++) {
pump.fill(32);
if (static_cast<unsigned>(col) >=
(out.width & ~7U)) 
len[0] = len[1] = 14;
else if ((col & 7) == 0) {
for (int& i : len) {
int j = 0;

for (; j < 5; j++) {
if (pump.getBitsNoFill(1) != 0) {
if (col == 0)
ThrowRDE("Can not initialize lengths. Data is corrupt.");

break;
}
}

invariant((col == 0 && j > 0) || col != 0);
if (j > 0)
i = length[2 * (j - 1) + pump.getBitsNoFill(1)];
}
}

int i = len[col & 1];
if (i == 14)
out(row, col) = pred[col & 1] = pump.getBitsNoFill(16);
else {
pred[col & 1] +=
static_cast<signed>(pump.getBitsNoFill(i)) + 1 - (1 << (i - 1));
out(row, col) = uint16_t(pred[col & 1]);
}
}
}

void PhaseOneDecompressor::decompressThread() const noexcept {
#ifdef HAVE_OPENMP
#pragma omp for schedule(static)
#endif
for (auto strip = strips.cbegin(); strip < strips.cend(); ++strip) {
try {
decompressStrip(*strip);
} catch (const RawspeedException& err) {
mRaw->setError(err.what());
}
}
}

void PhaseOneDecompressor::decompress() const {
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
