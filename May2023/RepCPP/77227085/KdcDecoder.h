

#pragma once

#include "common/RawImage.h"              
#include "decoders/AbstractTiffDecoder.h" 
#include "io/Buffer.h"                    
#include "tiff/TiffIFD.h"                 
#include <utility>                        

namespace rawspeed {

class CameraMetaData;

class KdcDecoder final : public AbstractTiffDecoder {
[[nodiscard]] Buffer getInputBuffer() const;

public:
static bool isAppropriateDecoder(const TiffRootIFD* rootIFD, Buffer file);
KdcDecoder(TiffRootIFDOwner&& root, Buffer file)
: AbstractTiffDecoder(std::move(root), file) {}

RawImage decodeRawInternal() override;
void decodeMetaDataInternal(const CameraMetaData* meta) override;

private:
[[nodiscard]] int getDecoderVersion() const override { return 0; }
};

} 
