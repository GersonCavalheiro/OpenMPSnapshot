

#pragma once

#include "adt/Point.h"                          
#include "common/Common.h"                      
#include "common/RawImage.h"                    
#include "decompressors/AbstractDecompressor.h" 
#include "io/ByteStream.h"                      
#include "io/Endianness.h"                      
#include <cstdint>                              
#include <utility>                              

namespace rawspeed {

class iPoint2D;

class UncompressedDecompressor final : public AbstractDecompressor {
ByteStream input;
RawImage mRaw;

const iPoint2D size;
const iPoint2D offset;
int inputPitchBytes;
int bitPerPixel;
BitOrder order;

uint32_t skipBytes;

void sanityCheck(const uint32_t* h, int bytesPerLine) const;

void sanityCheck(uint32_t w, const uint32_t* h, int bpp) const;

static int bytesPerLine(int w, bool skips);

template <typename Pump, typename NarrowFpType>
void decodePackedFP(int rows, int row) const;

template <typename Pump> void decodePackedInt(int rows, int row) const;

public:
UncompressedDecompressor(ByteStream input, const RawImage& img,
const iRectangle2D& crop, int inputPitchBytes,
int bitPerPixel, BitOrder order);








void readUncompressedRaw();


template <bool uncorrectedRawValues> void decode8BitRaw();


template <Endianness e> void decode12BitRawWithControl();


template <Endianness e> void decode12BitRawUnpackedLeftAligned();
};

extern template void UncompressedDecompressor::decode8BitRaw<false>();
extern template void UncompressedDecompressor::decode8BitRaw<true>();

extern template void
UncompressedDecompressor::decode12BitRawWithControl<Endianness::little>();
extern template void
UncompressedDecompressor::decode12BitRawWithControl<Endianness::big>();

extern template void
UncompressedDecompressor::decode12BitRawUnpackedLeftAligned<
Endianness::little>();
extern template void
UncompressedDecompressor::decode12BitRawUnpackedLeftAligned<Endianness::big>();

} 
