

#pragma once

#include "codes/PrefixCodeDecoder.h"
#include "common/RawImage.h"                    
#include "decompressors/AbstractDecompressor.h" 
#include "io/BitPumpMSB.h"                      
#include <array>                                
#include <cstdint>                              
#include <vector>                               

namespace rawspeed {
class ByteStream;
} 

namespace rawspeed {

class NikonDecompressor final : public AbstractDecompressor {
RawImage mRaw;
uint32_t bitsPS;

uint32_t huffSelect = 0;
uint32_t split = 0;

std::array<std::array<int, 2>, 2> pUp;

std::vector<uint16_t> curve;

uint32_t random;

public:
NikonDecompressor(const RawImage& raw, ByteStream metadata, uint32_t bitsPS);

void decompress(ByteStream data, bool uncorrectedRawValues);

private:
static const std::array<std::array<std::array<uint8_t, 16>, 2>, 6> nikon_tree;
static std::vector<uint16_t> createCurve(ByteStream& metadata,
uint32_t bitsPS, uint32_t v0,
uint32_t v1, uint32_t* split);

template <typename Huffman>
void decompress(BitPumpMSB& bits, int start_y, int end_y);

template <typename Huffman>
static Huffman createPrefixCodeDecoder(uint32_t huffSelect);
};

template <>
PrefixCodeDecoder<>
NikonDecompressor::createPrefixCodeDecoder<PrefixCodeDecoder<>>(
uint32_t huffSelect);

} 
