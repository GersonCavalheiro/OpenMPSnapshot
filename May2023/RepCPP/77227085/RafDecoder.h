

#pragma once

#include "common/RawImage.h"              
#include "decoders/AbstractTiffDecoder.h" 
#include "io/Buffer.h"                    
#include "tiff/TiffIFD.h"                 
#include <utility>                        

namespace rawspeed {

class Buffer;
class Camera;
class CameraMetaData;

class RafDecoder final : public AbstractTiffDecoder {
bool alt_layout = false;

public:
static bool isAppropriateDecoder(const TiffRootIFD* rootIFD, Buffer file);
RafDecoder(TiffRootIFDOwner&& root, Buffer file)
: AbstractTiffDecoder(std::move(root), file) {}

RawImage decodeRawInternal() override;
void applyCorrections(const Camera* cam);
void decodeMetaDataInternal(const CameraMetaData* meta) override;
void checkSupportInternal(const CameraMetaData* meta) override;
static bool isRAF(Buffer input);

protected:
[[nodiscard]] int getDecoderVersion() const override { return 1; }

private:
[[nodiscard]] int isCompressed() const;
};

} 
