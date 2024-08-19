

#pragma once

#include "codes/PrefixCodeDecoder.h"            
#include "common/RawImage.h"                    
#include "decompressors/AbstractDecompressor.h" 
#include "io/BitPumpJPEG.h"                     
#include "io/ByteStream.h"                      
#include <array>                                
#include <cstdint>                              

namespace rawspeed {

class CrwDecompressor final : public AbstractDecompressor {
using crw_hts = std::array<PrefixCodeDecoder<>, 2>;

RawImage mRaw;
crw_hts mHuff;
const bool lowbits;

ByteStream lowbitInput;
ByteStream rawInput;

public:
CrwDecompressor(const RawImage& img, uint32_t dec_table_, bool lowbits_,
ByteStream rawData);

void decompress();

private:
static PrefixCodeDecoder<> makeDecoder(const uint8_t* ncpl,
const uint8_t* values);
static crw_hts initHuffTables(uint32_t table);

inline static void decodeBlock(std::array<int16_t, 64>* diffBuf,
const crw_hts& mHuff, BitPumpJPEG& bs);
};

} 
