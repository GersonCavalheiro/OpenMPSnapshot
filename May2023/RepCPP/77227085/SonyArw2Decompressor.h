

#pragma once

#include "common/RawImage.h"                    
#include "decompressors/AbstractDecompressor.h" 
#include "io/ByteStream.h"                      

namespace rawspeed {

class SonyArw2Decompressor final : public AbstractDecompressor {
void decompressRow(int row) const;
void decompressThread() const noexcept;

RawImage mRaw;
ByteStream input;

public:
SonyArw2Decompressor(const RawImage& img, ByteStream input);
void decompress() const;
};

} 
