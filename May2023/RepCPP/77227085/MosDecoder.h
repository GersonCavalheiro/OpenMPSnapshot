

#pragma once

#include "common/RawImage.h"              
#include "decoders/AbstractTiffDecoder.h" 
#include "tiff/TiffIFD.h"                 
#include <string>                         
#include <string_view>                    

namespace rawspeed {

class Buffer;
class CameraMetaData;

class MosDecoder final : public AbstractTiffDecoder {
public:
static bool isAppropriateDecoder(const TiffRootIFD* rootIFD, Buffer file);
MosDecoder(TiffRootIFDOwner&& rootIFD, Buffer file);

RawImage decodeRawInternal() override;
void checkSupportInternal(const CameraMetaData* meta) override;
void decodeMetaDataInternal(const CameraMetaData* meta) override;

private:
[[nodiscard]] int getDecoderVersion() const override { return 0; }
std::string make;
std::string model;
static std::string getXMPTag(std::string_view xmp, std::string_view tag);
};

} 
