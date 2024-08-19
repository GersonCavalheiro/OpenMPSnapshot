

#pragma once

#include "decompressors/AbstractSamsungDecompressor.h" 
#include "io/BitPumpMSB32.h"                           
#include "io/ByteStream.h"                             
#include <array>                                       
#include <cstdint>                                     

namespace rawspeed {

class RawImage;

class SamsungV2Decompressor final : public AbstractSamsungDecompressor {
public:
enum struct OptFlags : uint32_t;

private:
uint32_t bitDepth;
int width;
int height;
OptFlags optflags;
uint16_t initVal;

ByteStream data;

int motion;
int scale;
std::array<std::array<int, 2>, 3> diffBitsMode;

static inline __attribute__((always_inline)) int16_t
getDiff(BitPumpMSB32& pump, uint32_t len);

inline __attribute__((always_inline)) std::array<uint16_t, 16>
prepareBaselineValues(BitPumpMSB32& pump, int row, int col);

inline __attribute__((always_inline)) std::array<uint32_t, 4>
decodeDiffLengths(BitPumpMSB32& pump, int row);

inline __attribute__((always_inline)) std::array<int, 16>
decodeDifferences(BitPumpMSB32& pump, int row);

inline __attribute__((always_inline)) void processBlock(BitPumpMSB32& pump,
int row, int col);

void decompressRow(int row);

public:
SamsungV2Decompressor(const RawImage& image, ByteStream bs, unsigned bit);

void decompress();
};

} 
