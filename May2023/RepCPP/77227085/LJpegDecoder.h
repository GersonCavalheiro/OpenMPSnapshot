

#pragma once

#include "decompressors/AbstractLJpegDecoder.h" 
#include <cstdint>                              

namespace rawspeed {

class ByteStream;
class RawImage;


class LJpegDecoder final : public AbstractLJpegDecoder {
void decodeScan() override;

uint32_t offX = 0;
uint32_t offY = 0;
uint32_t w = 0;
uint32_t h = 0;

public:
LJpegDecoder(ByteStream bs, const RawImage& img);

void decode(uint32_t offsetX, uint32_t offsetY, uint32_t width,
uint32_t height, bool fixDng16Bug_);
};

} 
