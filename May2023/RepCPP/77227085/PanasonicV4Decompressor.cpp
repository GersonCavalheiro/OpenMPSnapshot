

#include "rawspeedconfig.h" 
#include "decompressors/PanasonicV4Decompressor.h"
#include "adt/Array2DRef.h"               
#include "adt/Invariant.h"                
#include "adt/Mutex.h"                    
#include "adt/Point.h"                    
#include "common/Common.h"                
#include "common/RawImage.h"              
#include "decoders/RawDecoderException.h" 
#include "io/Buffer.h"                    
#include <algorithm>                      
#include <array>                          
#include <cassert>                        
#include <cstdint>                        
#include <iterator>                       
#include <limits>                         
#include <memory>                         
#include <vector>                         

namespace rawspeed {

PanasonicV4Decompressor::PanasonicV4Decompressor(const RawImage& img,
ByteStream input_,
bool zero_is_not_bad,
uint32_t section_split_offset_)
: mRaw(img), zero_is_bad(!zero_is_not_bad),
section_split_offset(section_split_offset_) {
if (mRaw->getCpp() != 1 || mRaw->getDataType() != RawImageType::UINT16 ||
mRaw->getBpp() != sizeof(uint16_t))
ThrowRDE("Unexpected component count / data type");

if (!mRaw->dim.hasPositiveArea() || mRaw->dim.x % PixelsPerPacket != 0) {
ThrowRDE("Unexpected image dimensions found: (%i; %i)", mRaw->dim.x,
mRaw->dim.y);
}

if (BlockSize < section_split_offset)
ThrowRDE("Bad section_split_offset: %u, less than BlockSize (%u)",
section_split_offset, BlockSize);

invariant(mRaw->dim.area() % PixelsPerPacket == 0);
const auto bytesTotal = (mRaw->dim.area() / PixelsPerPacket) * BytesPerPacket;
invariant(bytesTotal > 0);

const auto bufSize =
section_split_offset == 0 ? bytesTotal : roundUp(bytesTotal, BlockSize);

if (bufSize > std::numeric_limits<ByteStream::size_type>::max())
ThrowRDE("Raw dimensions require input buffer larger than supported");

input = input_.peekStream(bufSize);

chopInputIntoBlocks();
}

void PanasonicV4Decompressor::chopInputIntoBlocks() {
auto pixelToCoordinate = [width = mRaw->dim.x](unsigned pixel) {
return iPoint2D(pixel % width, pixel / width);
};

const auto blocksTotal = roundUpDivision(input.getRemainSize(), BlockSize);
invariant(blocksTotal > 0);
invariant(blocksTotal * PixelsPerBlock >= mRaw->dim.area());
blocks.reserve(blocksTotal);

unsigned currPixel = 0;
std::generate_n(
std::back_inserter(blocks), blocksTotal, [&, pixelToCoordinate]() {
invariant(input.getRemainSize() != 0);
const auto blockSize = std::min(input.getRemainSize(), BlockSize);
invariant(blockSize > 0);
invariant(blockSize % BytesPerPacket == 0);
const auto packets = blockSize / BytesPerPacket;
invariant(packets > 0);
const auto pixels = packets * PixelsPerPacket;
invariant(pixels > 0);

ByteStream bs = input.getStream(blockSize);
iPoint2D beginCoord = pixelToCoordinate(currPixel);
currPixel += pixels;
iPoint2D endCoord = pixelToCoordinate(currPixel);
return Block(bs, beginCoord, endCoord);
});
assert(blocks.size() == blocksTotal);
assert(currPixel >= mRaw->dim.area());
assert(input.getRemainSize() == 0);

blocks.back().endCoord = mRaw->dim;
blocks.back().endCoord.y -= 1;
}

class PanasonicV4Decompressor::ProxyStream {
ByteStream block;
const uint32_t section_split_offset;
std::vector<uint8_t> buf;

int vbits = 0;

void parseBlock() {
assert(buf.empty());
invariant(block.getRemainSize() <= BlockSize);
invariant(section_split_offset <= BlockSize);

Buffer FirstSection = block.getBuffer(section_split_offset);
Buffer SecondSection = block.getBuffer(block.getRemainSize());

buf.reserve(BlockSize + 1UL);

buf.insert(buf.end(), SecondSection.begin(), SecondSection.end());
buf.insert(buf.end(), FirstSection.begin(), FirstSection.end());

invariant(block.getRemainSize() == 0);

buf.emplace_back(0);
}

public:
ProxyStream(ByteStream block_, int section_split_offset_)
: block(block_), section_split_offset(section_split_offset_) {
parseBlock();
}

uint32_t getBits(int nbits) noexcept {
vbits = (vbits - nbits) & 0x1ffff;
int byte = vbits >> 3 ^ 0x3ff0;
return (buf[byte] | buf[byte + 1UL] << 8) >> (vbits & 7) & ~(-(1 << nbits));
}
};

inline void PanasonicV4Decompressor::processPixelPacket(
ProxyStream& bits, int row, int col,
std::vector<uint32_t>* zero_pos) const noexcept {
const Array2DRef<uint16_t> out(mRaw->getU16DataAsUncroppedArray2DRef());

int sh = 0;

std::array<int, 2> pred;
pred.fill(0);

std::array<int, 2> nonz;
nonz.fill(0);

int u = 0;

for (int p = 0; p < PixelsPerPacket; ++p, ++col) {
const int c = p & 1;

if (u == 2) {
sh = extractHighBits(4U, bits.getBits(2), 3);
u = -1;
}

if (nonz[c]) {
int j = bits.getBits(8);
if (j) {
pred[c] -= 0x80 << sh;
if (pred[c] < 0 || sh == 4)
pred[c] &= ~(-(1 << sh));
pred[c] += j << sh;
}
} else {
nonz[c] = bits.getBits(8);
if (nonz[c] || p > 11)
pred[c] = nonz[c] << 4 | bits.getBits(4);
}

out(row, col) = pred[c];

if (zero_is_bad && 0 == pred[c])
zero_pos->push_back((row << 16) | col);

u++;
}
}

void PanasonicV4Decompressor::processBlock(
const Block& block, std::vector<uint32_t>* zero_pos) const noexcept {
ProxyStream bits(block.bs, section_split_offset);

for (int row = block.beginCoord.y; row <= block.endCoord.y; row++) {
int col = 0;
if (block.beginCoord.y == row)
col = block.beginCoord.x;

int endCol = mRaw->dim.x;
if (block.endCoord.y == row)
endCol = block.endCoord.x;

invariant(col % PixelsPerPacket == 0);
invariant(endCol % PixelsPerPacket == 0);

for (; col < endCol; col += PixelsPerPacket)
processPixelPacket(bits, row, col, zero_pos);
}
}

void PanasonicV4Decompressor::decompressThread() const noexcept {
std::vector<uint32_t> zero_pos;

assert(!blocks.empty());

#ifdef HAVE_OPENMP
#pragma omp for schedule(static)
#endif
for (auto block = blocks.cbegin(); block < blocks.cend(); ++block)
processBlock(*block, &zero_pos);

if (zero_is_bad && !zero_pos.empty()) {
MutexLocker guard(&mRaw->mBadPixelMutex);
mRaw->mBadPixelPositions.insert(mRaw->mBadPixelPositions.end(),
zero_pos.begin(), zero_pos.end());
}
}

void PanasonicV4Decompressor::decompress() const noexcept {
assert(!blocks.empty());
#ifdef HAVE_OPENMP
#pragma omp parallel default(none)                                             \
num_threads(rawspeed_get_number_of_processor_cores())
#endif
decompressThread();
}

} 
