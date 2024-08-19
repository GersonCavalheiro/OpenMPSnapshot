

#pragma once

#include "adt/Point.h"                          
#include "common/RawImage.h"                    
#include "decompressors/AbstractDecompressor.h" 
#include "io/BitPumpLSB.h"                      
#include "io/ByteStream.h"                      
#include <cstddef>                              
#include <cstdint>                              
#include <utility>                              
#include <vector>                               

namespace rawspeed {

class PanasonicV5Decompressor final : public AbstractDecompressor {
static constexpr uint32_t BlockSize = 0x4000;

static constexpr uint32_t sectionSplitOffset = 0x1FF8;

static constexpr uint32_t bytesPerPacket = 16;
static constexpr uint32_t bitsPerPacket = 8 * bytesPerPacket;
static_assert(BlockSize % bytesPerPacket == 0);
static constexpr uint32_t PacketsPerBlock = BlockSize / bytesPerPacket;

struct PacketDsc;

static const PacketDsc TwelveBitPacket;
static const PacketDsc FourteenBitPacket;

class ProxyStream;

RawImage mRaw;

ByteStream input;

const uint32_t bps;

size_t numBlocks;

struct Block {
ByteStream bs;
iPoint2D beginCoord;
iPoint2D endCoord;

Block() = default;
Block(ByteStream bs_, iPoint2D beginCoord_, iPoint2D endCoord_)
: bs(bs_), beginCoord(beginCoord_), endCoord(endCoord_) {}
};

std::vector<Block> blocks;

void chopInputIntoBlocks(const PacketDsc& dsc);

template <const PacketDsc& dsc>
inline void processPixelPacket(BitPumpLSB& bs, int row, int col) const;

template <const PacketDsc& dsc> void processBlock(const Block& block) const;

template <const PacketDsc& dsc> void decompressInternal() const noexcept;

public:
PanasonicV5Decompressor(const RawImage& img, ByteStream input_,
uint32_t bps_);

void decompress() const noexcept;
};

} 
