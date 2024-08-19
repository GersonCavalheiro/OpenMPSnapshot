

#pragma once

#include "common/RawImage.h"                    
#include "decompressors/AbstractDecompressor.h" 
#include "io/ByteStream.h"                      
#include <climits>                              
#include <cstdint>                              
#include <functional>

namespace rawspeed {
template <class T> class CroppedArray1DRef;

class PanasonicV7Decompressor final : public AbstractDecompressor {
RawImage mRaw;

ByteStream input;

static constexpr int BytesPerBlock = 16;
static constexpr int BitsPerSample = 14;
static constexpr int PixelsPerBlock =
(CHAR_BIT * BytesPerBlock) / BitsPerSample;

static inline void __attribute__((always_inline))
decompressBlock(ByteStream block, CroppedArray1DRef<uint16_t> out) noexcept;

void decompressRow(int row) const noexcept;

public:
PanasonicV7Decompressor(const RawImage& img, ByteStream input_);

void decompress() const;
};

} 
