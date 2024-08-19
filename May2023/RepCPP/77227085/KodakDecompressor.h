

#pragma once

#include "common/RawImage.h"                    
#include "decompressors/AbstractDecompressor.h" 
#include "io/ByteStream.h"                      
#include <array>                                
#include <cstdint>                              

namespace rawspeed {

class KodakDecompressor final : public AbstractDecompressor {
RawImage mRaw;
ByteStream input;
int bps;
bool uncorrectedRawValues;

static constexpr int segment_size = 256; 
using segment = std::array<int16_t, segment_size>;

segment decodeSegment(uint32_t bsize);

public:
KodakDecompressor(const RawImage& img, ByteStream bs, int bps,
bool uncorrectedRawValues_);

void decompress();
};

} 
