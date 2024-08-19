

#pragma once

#include "adt/Point.h"                    
#include "adt/iterator_range.h"           
#include "codes/PrefixCodeDecoder.h"      
#include "common/RawImage.h"              
#include "common/RawspeedException.h"     
#include "decoders/RawDecoderException.h" 
#include "io/BitPumpMSB32.h"              
#include "io/Buffer.h"                    
#include "io/ByteStream.h"                
#include <array>                          
#include <cassert>                        
#include <cstddef>                        
#include <cstdint>                        
#include <functional>                     
#include <iterator>                       
#include <tuple>                          
#include <utility>                        
#include <vector>                         

namespace rawspeed {

class HasselbladDecompressor final {
public:
struct PerComponentRecipe {
const PrefixCodeDecoder<>& ht;
const uint16_t initPred;
};

private:
const RawImage mRaw;

const PerComponentRecipe& rec;

const ByteStream input;

static int getBits(BitPumpMSB32& bs, int len);

public:
HasselbladDecompressor(const RawImage& mRaw, const PerComponentRecipe& rec,
ByteStream input);

[[nodiscard]] ByteStream::size_type decompress();
};

} 
