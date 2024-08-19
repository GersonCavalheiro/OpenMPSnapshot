

#pragma once

#include "rawspeedconfig.h" 

#ifdef HAVE_JPEG

#include "common/RawImage.h"                    
#include "decompressors/AbstractDecompressor.h" 
#include "io/Buffer.h"                          
#include <cstdint>                              
#include <utility>                              

namespace rawspeed {

class JpegDecompressor final : public AbstractDecompressor {
struct JpegDecompressStruct;

Buffer input;
RawImage mRaw;

public:
JpegDecompressor(Buffer bs, const RawImage& img) : input(bs), mRaw(img) {}

void decode(uint32_t offsetX, uint32_t offsetY);
};

} 

#else

#pragma message                                                                \
"JPEG is not present! Lossy JPEG compression will not be supported!"

#endif
