

#pragma once

#include "common/RawImage.h"                    
#include "decompressors/AbstractDecompressor.h" 
#include "io/ByteStream.h"                      
#include <cstdint>                              

namespace rawspeed {

class PanasonicV6Decompressor final : public AbstractDecompressor {
RawImage mRaw;

ByteStream input;

struct BlockDsc;

static const BlockDsc TwelveBitBlock;
static const BlockDsc FourteenBitBlock;

const uint32_t bps;

template <const BlockDsc& dsc>
inline void __attribute__((always_inline))
decompressBlock(ByteStream& rowInput, int row, int col) const noexcept;

template <const BlockDsc& dsc> void decompressRow(int row) const noexcept;

template <const BlockDsc& dsc> void decompressInternal() const noexcept;

public:
PanasonicV6Decompressor(const RawImage& img, ByteStream input_,
uint32_t bps_);

void decompress() const noexcept;
};

} 
