

#pragma once

#include "decompressors/AbstractSamsungDecompressor.h" 
#include "io/BitPumpMSB.h"                             
#include "io/ByteStream.h"                             
#include <cstdint>                                     
#include <vector>                                      

namespace rawspeed {

class ByteStream;
class RawImage;

class SamsungV1Decompressor final : public AbstractSamsungDecompressor {
struct encTableItem;

static inline int32_t samsungDiff(BitPumpMSB& pump,
const std::vector<encTableItem>& tbl);

ByteStream bs;
static constexpr int bits = 12;

public:
SamsungV1Decompressor(const RawImage& image, ByteStream bs_, int bit);

void decompress() const;
};

} 
