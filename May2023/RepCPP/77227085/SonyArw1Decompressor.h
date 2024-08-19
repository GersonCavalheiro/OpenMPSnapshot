

#pragma once

#include "common/RawImage.h"                    
#include "decompressors/AbstractDecompressor.h" 
#include "io/BitPumpMSB.h"                      
#include <cstdint>                              

namespace rawspeed {

class ByteStream;

class SonyArw1Decompressor final : public AbstractDecompressor {
RawImage mRaw;

inline static int getDiff(BitPumpMSB& bs, uint32_t len);

public:
explicit SonyArw1Decompressor(const RawImage& img);
void decompress(ByteStream input) const;
};

} 
