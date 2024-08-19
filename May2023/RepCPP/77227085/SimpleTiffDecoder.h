

#pragma once

#include "decoders/AbstractTiffDecoder.h" 
#include "io/Buffer.h"                    
#include "tiff/TiffIFD.h"                 
#include <cstdint>                        
#include <utility>                        

namespace rawspeed {

class Buffer;

class SimpleTiffDecoder : public AbstractTiffDecoder {
virtual void checkImageDimensions() = 0;

public:
SimpleTiffDecoder(TiffRootIFDOwner&& root, Buffer file)
: AbstractTiffDecoder(std::move(root), file) {}

void prepareForRawDecoding();

protected:
const TiffIFD* raw;
uint32_t width;
uint32_t height;
uint32_t off;
uint32_t c2;
};

} 
