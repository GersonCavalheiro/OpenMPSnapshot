

#pragma once

#include "decompressors/AbstractLJpegDecoder.h" 
#include "decompressors/Cr2Decompressor.h"      
#include <cassert>                              
#include <cstdint>                              

namespace rawspeed {

class ByteStream;
class RawImage;

class Cr2LJpegDecoder final : public AbstractLJpegDecoder {
Cr2SliceWidths slicing;

void decodeScan() override;

public:
Cr2LJpegDecoder(ByteStream bs, const RawImage& img);
void decode(const Cr2SliceWidths& slicing);
};

} 
