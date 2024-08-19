

#pragma once

#include "common/Common.h"                      
#include "common/RawImage.h"                    
#include "common/SimpleLUT.h"                   
#include "decompressors/AbstractDecompressor.h" 
#include "io/BitPumpMSB.h"                      
#include <algorithm>                            
#include <array>                                
#include <cstdint>                              

namespace rawspeed {

class ByteStream;
template <class T> class Array2DRef;

class OlympusDecompressor final : public AbstractDecompressor {
RawImage mRaw;

const SimpleLUT<char, 12> bittable{
[](unsigned i, [[maybe_unused]] unsigned tableSize) {
int high;
for (high = 0; high < 12; high++)
if (extractHighBits(i, high, 11) & 1)
break;
return std::min(12, high);
}};

inline __attribute__((always_inline)) int
parseCarry(BitPumpMSB& bits, std::array<int, 3>* carry) const;

static inline int getPred(Array2DRef<uint16_t> out, int row, int col);

void decompressRow(BitPumpMSB& bits, int row) const;

public:
explicit OlympusDecompressor(const RawImage& img);
void decompress(ByteStream input) const;
};

} 
