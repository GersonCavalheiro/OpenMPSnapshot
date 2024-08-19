

#pragma once

#include "rawspeedconfig.h"                     
#include "adt/Array2DRef.h"                     
#include "adt/Point.h"                          
#include "common/RawImage.h"                    
#include "decompressors/AbstractDecompressor.h" 
#include "io/BitPumpMSB.h"                      
#include "io/ByteStream.h"                      
#include "metadata/ColorFilterArray.h"          
#include <array>                                
#include <cassert>                              
#include <cstdint>                              
#include <utility>                              
#include <vector>                               

namespace rawspeed {

class FujiDecompressor final : public AbstractDecompressor {
RawImage mRaw;

public:
FujiDecompressor(const RawImage& img, ByteStream input);

void decompress() const;

struct FujiHeader {
FujiHeader() = default;

explicit FujiHeader(ByteStream& input_);
explicit RAWSPEED_READONLY operator bool() const; 

uint16_t signature;
uint8_t version;
uint8_t raw_type;
uint8_t raw_bits;
uint16_t raw_height;
uint16_t raw_rounded_width;
uint16_t raw_width;
uint16_t block_size;
uint8_t blocks_in_row;
uint16_t total_lines;
iPoint2D MCU;
};

private:
FujiHeader header;

ByteStream input;

std::vector<ByteStream> strips;
};

} 
