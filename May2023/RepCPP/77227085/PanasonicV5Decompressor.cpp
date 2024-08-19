

#include "rawspeedconfig.h" 
#include "decompressors/PanasonicV5Decompressor.h"
#include "adt/Array2DRef.h"               
#include "adt/Invariant.h"                
#include "adt/Point.h"                    
#include "common/Common.h"                
#include "common/RawImage.h"              
#include "decoders/RawDecoderException.h" 
#include "io/BitPumpLSB.h"                
#include "io/Buffer.h"                    
#include "io/Endianness.h"                
#include <algorithm>                      
#include <cassert>                        
#include <cstdint>                        
#include <iterator>                       
#include <memory>                         
#include <vector>                         

namespace rawspeed {

struct PanasonicV5Decompressor::PacketDsc {
Buffer::size_type bps;
int pixelsPerPacket;

constexpr PacketDsc();
explicit constexpr PacketDsc(int bps_)
: bps(bps_),
pixelsPerPacket(PanasonicV5Decompressor::bitsPerPacket / bps) {
}
};

constexpr PanasonicV5Decompressor::PacketDsc
PanasonicV5Decompressor::TwelveBitPacket =
PanasonicV5Decompressor::PacketDsc(12);
constexpr PanasonicV5Decompressor::PacketDsc
PanasonicV5Decompressor::FourteenBitPacket =
PanasonicV5Decompressor::PacketDsc(14);

PanasonicV5Decompressor::PanasonicV5Decompressor(const RawImage& img,
ByteStream input_,
uint32_t bps_)
: mRaw(img), bps(bps_) {
if (mRaw->getCpp() != 1 || mRaw->getDataType() != RawImageType::UINT16 ||
mRaw->getBpp() != sizeof(uint16_t))
ThrowRDE("Unexpected component count / data type");

const PacketDsc* dsc = nullptr;
switch (bps) {
case 12:
dsc = &TwelveBitPacket;
break;
case 14:
dsc = &FourteenBitPacket;
break;
default:
ThrowRDE("Unsupported bps: %u", bps);
}

if (!mRaw->dim.hasPositiveArea() || mRaw->dim.x % dsc->pixelsPerPacket != 0) {
ThrowRDE("Unexpected image dimensions found: (%i; %i)", mRaw->dim.x,
mRaw->dim.y);
}

invariant(mRaw->dim.area() % dsc->pixelsPerPacket == 0);
const auto numPackets = mRaw->dim.area() / dsc->pixelsPerPacket;
invariant(numPackets > 0);

numBlocks = roundUpDivision(numPackets, PacketsPerBlock);
invariant(numBlocks > 0);

if (const auto haveBlocks = input_.getRemainSize() / BlockSize;
haveBlocks < numBlocks)
ThrowRDE("Insufficient count of input blocks for a given image");

input = input_.peekStream(numBlocks, BlockSize);

chopInputIntoBlocks(*dsc);
}

void PanasonicV5Decompressor::chopInputIntoBlocks(const PacketDsc& dsc) {
auto pixelToCoordinate = [width = mRaw->dim.x](unsigned pixel) {
return iPoint2D(pixel % width, pixel / width);
};

invariant(numBlocks * BlockSize == input.getRemainSize());
blocks.reserve(numBlocks);

const auto pixelsPerBlock = dsc.pixelsPerPacket * PacketsPerBlock;
invariant((numBlocks - 1U) * pixelsPerBlock < mRaw->dim.area());
invariant(numBlocks * pixelsPerBlock >= mRaw->dim.area());

unsigned currPixel = 0;
std::generate_n(std::back_inserter(blocks), numBlocks,
[&, pixelToCoordinate, pixelsPerBlock]() {
ByteStream bs = input.getStream(BlockSize);
iPoint2D beginCoord = pixelToCoordinate(currPixel);
currPixel += pixelsPerBlock;
iPoint2D endCoord = pixelToCoordinate(currPixel);
return Block(bs, beginCoord, endCoord);
});
assert(blocks.size() == numBlocks);
invariant(currPixel >= mRaw->dim.area());
invariant(input.getRemainSize() == 0);

blocks.back().endCoord = mRaw->dim;
blocks.back().endCoord.y -= 1;
}

class PanasonicV5Decompressor::ProxyStream {
ByteStream block;
std::vector<uint8_t> buf;
ByteStream input;

void parseBlock() {
assert(buf.empty());
invariant(block.getRemainSize() == BlockSize);

static_assert(BlockSize > sectionSplitOffset);

Buffer FirstSection = block.getBuffer(sectionSplitOffset);
Buffer SecondSection = block.getBuffer(block.getRemainSize());
invariant(FirstSection.getSize() < SecondSection.getSize());

buf.reserve(BlockSize);

buf.insert(buf.end(), SecondSection.begin(), SecondSection.end());
buf.insert(buf.end(), FirstSection.begin(), FirstSection.end());

assert(buf.size() == BlockSize);
invariant(block.getRemainSize() == 0);

input = ByteStream(
DataBuffer(Buffer(buf.data(), buf.size()), Endianness::little));
}

public:
explicit ProxyStream(ByteStream block_) : block(block_) {}

ByteStream& getStream() {
parseBlock();
return input;
}
};

template <const PanasonicV5Decompressor::PacketDsc& dsc>
inline void PanasonicV5Decompressor::processPixelPacket(BitPumpLSB& bs, int row,
int col) const {
static_assert(dsc.pixelsPerPacket > 0, "dsc should be compile-time const");
static_assert(dsc.bps > 0 && dsc.bps <= 16);

const Array2DRef<uint16_t> out(mRaw->getU16DataAsUncroppedArray2DRef());

invariant(bs.getFillLevel() == 0);

for (int p = 0; p < dsc.pixelsPerPacket;) {
bs.fill();
for (; bs.getFillLevel() >= dsc.bps; ++p, ++col)
out(row, col) = bs.getBitsNoFill(dsc.bps);
}
bs.skipBitsNoFill(bs.getFillLevel()); 
}

template <const PanasonicV5Decompressor::PacketDsc& dsc>
void PanasonicV5Decompressor::processBlock(const Block& block) const {
static_assert(dsc.pixelsPerPacket > 0, "dsc should be compile-time const");
static_assert(BlockSize % bytesPerPacket == 0);

ProxyStream proxy(block.bs);
BitPumpLSB bs(proxy.getStream());

for (int row = block.beginCoord.y; row <= block.endCoord.y; row++) {
int col = 0;
if (block.beginCoord.y == row)
col = block.beginCoord.x;

int endx = mRaw->dim.x;
if (block.endCoord.y == row)
endx = block.endCoord.x;

invariant(col % dsc.pixelsPerPacket == 0);
invariant(endx % dsc.pixelsPerPacket == 0);

for (; col < endx; col += dsc.pixelsPerPacket)
processPixelPacket<dsc>(bs, row, col);
}
}

template <const PanasonicV5Decompressor::PacketDsc& dsc>
void PanasonicV5Decompressor::decompressInternal() const noexcept {
#ifdef HAVE_OPENMP
#pragma omp parallel for num_threads(rawspeed_get_number_of_processor_cores()) \
schedule(static) default(none)
#endif
for (auto block = blocks.cbegin(); block < blocks.cend();
++block) { 
processBlock<dsc>(*block);
}
}

void PanasonicV5Decompressor::decompress() const noexcept {
switch (bps) {
case 12:
decompressInternal<TwelveBitPacket>();
break;
case 14:
decompressInternal<FourteenBitPacket>();
break;
default:
__builtin_unreachable();
}
}

} 
