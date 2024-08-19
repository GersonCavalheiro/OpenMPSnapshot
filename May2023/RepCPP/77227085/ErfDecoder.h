

#pragma once

#include "common/RawImage.h"            
#include "decoders/SimpleTiffDecoder.h" 
#include "io/Buffer.h"                  
#include "tiff/TiffIFD.h"               
#include <utility>                      

namespace rawspeed {

class Buffer;
class CameraMetaData;

class ErfDecoder final : public SimpleTiffDecoder {
void checkImageDimensions() override;

public:
static bool isAppropriateDecoder(const TiffRootIFD* rootIFD, Buffer file);
ErfDecoder(TiffRootIFDOwner&& root, Buffer file)
: SimpleTiffDecoder(std::move(root), file) {}

RawImage decodeRawInternal() override;
void decodeMetaDataInternal(const CameraMetaData* meta) override;

private:
[[nodiscard]] int getDecoderVersion() const override { return 0; }
};

} 
