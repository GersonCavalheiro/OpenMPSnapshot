

#pragma once

#include "common/RawImage.h"                    
#include "decompressors/AbstractDecompressor.h" 

namespace rawspeed {

class AbstractSamsungDecompressor : public AbstractDecompressor {
protected:
RawImage mRaw;

public:
explicit AbstractSamsungDecompressor(const RawImage& raw) : mRaw(raw) {}
};

} 
