

#include "rawspeedconfig.h" 
#include "decompressors/PanasonicV7Decompressor.h"
#include "adt/Array1DRef.h"               
#include "adt/Array2DRef.h"               
#include "adt/CroppedArray1DRef.h"        
#include "adt/Invariant.h"                
#include "adt/Point.h"                    
#include "common/Common.h"                
#include "common/RawImage.h"              
#include "decoders/RawDecoderException.h" 
#include "io/BitPumpLSB.h"                
#include <cstdint>                        

namespace rawspeed {

PanasonicV7Decompressor::PanasonicV7Decompressor(const RawImage& img,
ByteStream input_)
: mRaw(img) {
if (mRaw->getCpp() != 1 || mRaw->getDataType() != RawImageType::UINT16 ||
mRaw->getBpp() != sizeof(uint16_t))
ThrowRDE("Unexpected component count / data type");

if (!mRaw->dim.hasPositiveArea() || mRaw->dim.x % PixelsPerBlock != 0) {
ThrowRDE("Unexpected image dimensions found: (%i; %i)", mRaw->dim.x,
mRaw->dim.y);
}

const auto numBlocks = mRaw->dim.area() / PixelsPerBlock;

if (const auto haveBlocks = input_.getRemainSize() / BytesPerBlock;
haveBlocks < numBlocks)
ThrowRDE("Insufficient count of input blocks for a given image");

input = input_.peekStream(numBlocks, BytesPerBlock);
}

inline void __attribute__((always_inline))
PanasonicV7Decompressor::decompressBlock(
ByteStream block, CroppedArray1DRef<uint16_t> out) noexcept {
invariant(out.size() == PixelsPerBlock);
BitPumpLSB pump(block);
for (int pix = 0; pix < PixelsPerBlock; pix++)
out(pix) = pump.getBits(BitsPerSample);
}

void PanasonicV7Decompressor::decompressRow(int row) const noexcept {
const Array2DRef<uint16_t> out(mRaw->getU16DataAsUncroppedArray2DRef());
Array1DRef<uint16_t> outRow = out[row];

invariant(outRow.size() % PixelsPerBlock == 0);
const int blocksperrow = outRow.size() / PixelsPerBlock;
const int bytesPerRow = BytesPerBlock * blocksperrow;

ByteStream rowInput = input.getSubStream(bytesPerRow * row, bytesPerRow);
for (int rblock = 0; rblock < blocksperrow; rblock++) {
ByteStream block = rowInput.getStream(BytesPerBlock);
decompressBlock(block,
outRow.getCrop(PixelsPerBlock * rblock, PixelsPerBlock));
}
}

void PanasonicV7Decompressor::decompress() const {
#ifdef HAVE_OPENMP
#pragma omp parallel for num_threads(rawspeed_get_number_of_processor_cores()) \
schedule(static) default(none)
#endif
for (int row = 0; row < mRaw->dim.y;
++row) { 
decompressRow(row);
}
}

} 
