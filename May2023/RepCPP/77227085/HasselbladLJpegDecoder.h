

#pragma once

#include "decompressors/AbstractLJpegDecoder.h" 
#include "io/BitPumpMSB32.h"                    

namespace rawspeed {

class ByteStream;
class RawImage;

class HasselbladLJpegDecoder final : public AbstractLJpegDecoder {
void decodeScan() override;

public:
HasselbladLJpegDecoder(ByteStream bs, const RawImage& img);

void decode();
};

} 
