

#pragma once

#include "codes/PrefixCodeDecoder.h"            
#include "common/RawImage.h"                    
#include "decompressors/AbstractDecompressor.h" 
#include <array>                                
#include <cstdint>                              
#include <optional>                             

namespace rawspeed {

class ByteStream;

class PentaxDecompressor final : public AbstractDecompressor {
RawImage mRaw;
const PrefixCodeDecoder<> ht;

public:
PentaxDecompressor(const RawImage& img, std::optional<ByteStream> metaData);

void decompress(ByteStream data) const;

private:
static HuffmanCode<BaselineCodeTag> SetupPrefixCodeDecoder_Legacy();
static HuffmanCode<BaselineCodeTag>
SetupPrefixCodeDecoder_Modern(ByteStream stream);
static PrefixCodeDecoder<>
SetupPrefixCodeDecoder(std::optional<ByteStream> metaData);

static const std::array<std::array<std::array<uint8_t, 16>, 2>, 1>
pentax_tree;
};

} 
