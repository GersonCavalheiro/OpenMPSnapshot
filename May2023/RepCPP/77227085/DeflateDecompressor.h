

#pragma once

#include "rawspeedconfig.h" 

#ifdef HAVE_ZLIB

#include "common/RawImage.h"                    
#include "decompressors/AbstractDecompressor.h" 
#include "io/Buffer.h"                          
#include <memory>                               
#include <utility>                              

namespace rawspeed {

class iPoint2D;

class DeflateDecompressor final : public AbstractDecompressor {
Buffer input;
RawImage mRaw;
int predFactor;
int bps;

public:
DeflateDecompressor(Buffer bs, const RawImage& img, int predictor, int bps_);

void decode(std::unique_ptr<unsigned char[]>* uBuffer, 
iPoint2D maxDim, iPoint2D dim, iPoint2D off);
};

} 

#else

#pragma message                                                                \
"ZLIB is not present! Deflate compression will not be supported!"

#endif
