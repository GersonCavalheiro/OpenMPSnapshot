

#pragma once

#include "adt/Point.h"                          
#include "common/RawImage.h"                    
#include "decompressors/AbstractDecompressor.h" 
#include "io/ByteStream.h"                      
#include <cstdint>                              
#include <utility>                              
#include <vector>                               

namespace rawspeed {

class PanasonicV4Decompressor final : public AbstractDecompressor {
static constexpr uint32_t BlockSize = 0x4000;

static constexpr int PixelsPerPacket = 14;

static constexpr uint32_t BytesPerPacket = 16;

static constexpr uint32_t PacketsPerBlock = BlockSize / BytesPerPacket;

static constexpr uint32_t PixelsPerBlock = PixelsPerPacket * PacketsPerBlock;

class ProxyStream;

RawImage mRaw;
ByteStream input;
bool zero_is_bad;

uint32_t section_split_offset;

struct Block {
ByteStream bs;
iPoint2D beginCoord;
iPoint2D endCoord;

Block() = default;
Block(ByteStream bs_, iPoint2D beginCoord_, iPoint2D endCoord_)
: bs(bs_), beginCoord(beginCoord_), endCoord(endCoord_) {}
};

std::vector<Block> blocks;

void chopInputIntoBlocks();

inline void
processPixelPacket(ProxyStream& bits, int row, int col,
std::vector<uint32_t>* zero_pos) const noexcept;

void processBlock(const Block& block,
std::vector<uint32_t>* zero_pos) const noexcept;

void decompressThread() const noexcept;

public:
PanasonicV4Decompressor(const RawImage& img, ByteStream input_,
bool zero_is_not_bad, uint32_t section_split_offset_);

void decompress() const noexcept;
};

} 
