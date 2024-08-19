

#pragma once

#include "common/RawImage.h"              
#include "decoders/AbstractTiffDecoder.h" 
#include "io/Buffer.h"                    
#include "tiff/TiffIFD.h"                 
#include <string>                         
#include <utility>                        

namespace rawspeed {

class Buffer;
class CameraMetaData;

class SrwDecoder final : public AbstractTiffDecoder {
public:
static bool isAppropriateDecoder(const TiffRootIFD* rootIFD, Buffer file);
SrwDecoder(TiffRootIFDOwner&& root, Buffer file)
: AbstractTiffDecoder(std::move(root), file) {}

RawImage decodeRawInternal() override;
void decodeMetaDataInternal(const CameraMetaData* meta) override;
void checkSupportInternal(const CameraMetaData* meta) override;

private:
[[nodiscard]] int getDecoderVersion() const override { return 3; }
[[nodiscard]] std::string getMode() const;
};

} 
