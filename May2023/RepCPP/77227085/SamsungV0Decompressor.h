

#pragma once

#include "decompressors/AbstractSamsungDecompressor.h" 
#include "io/BitPumpMSB32.h"                           
#include "io/ByteStream.h"                             
#include <cstdint>                                     
#include <vector>                                      

namespace rawspeed {

class RawImage;

class SamsungV0Decompressor final : public AbstractSamsungDecompressor {
std::vector<ByteStream> stripes;

void computeStripes(ByteStream bso, ByteStream bsr);

void decompressStrip(int row, ByteStream bs) const;

static int32_t calcAdj(BitPumpMSB32& bits, int b);

public:
SamsungV0Decompressor(const RawImage& image, ByteStream bso, ByteStream bsr);

void decompress() const;
};

} 
