

#pragma once

#include "adt/Point.h"                    
#include "common/RawImage.h"              
#include "decoders/AbstractTiffDecoder.h" 
#include "io/Buffer.h"                    
#include "tiff/TiffIFD.h"                 
#include <utility>                        

namespace rawspeed {

class Buffer;
class CameraMetaData;

class Cr2Decoder final : public AbstractTiffDecoder {
public:
static bool isAppropriateDecoder(const TiffRootIFD* rootIFD, Buffer file);
Cr2Decoder(TiffRootIFDOwner&& root, Buffer file)
: AbstractTiffDecoder(std::move(root), file) {}

RawImage decodeRawInternal() override;
void checkSupportInternal(const CameraMetaData* meta) override;
void decodeMetaDataInternal(const CameraMetaData* meta) override;

private:
[[nodiscard]] int getDecoderVersion() const override { return 9; }
RawImage decodeOldFormat();
RawImage decodeNewFormat();
void sRawInterpolate();
[[nodiscard]] bool isSubSampled() const;
[[nodiscard]] iPoint2D getSubSampling() const;
[[nodiscard]] int getHue() const;
};

} 
