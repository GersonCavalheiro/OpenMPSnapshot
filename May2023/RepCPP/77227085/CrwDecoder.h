

#pragma once

#include "common/RawImage.h"     
#include "decoders/RawDecoder.h" 
#include "tiff/CiffIFD.h"        
#include <cstdint>               
#include <memory>                

namespace rawspeed {

class Buffer;
class CameraMetaData;

class CrwDecoder final : public RawDecoder {
std::unique_ptr<const CiffIFD> mRootIFD;

public:
CrwDecoder(std::unique_ptr<const CiffIFD> rootIFD, Buffer file);
RawImage decodeRawInternal() override;
void checkSupportInternal(const CameraMetaData* meta) override;
void decodeMetaDataInternal(const CameraMetaData* meta) override;
static bool isCRW(Buffer input);

private:
[[nodiscard]] int getDecoderVersion() const override { return 0; }
static float canonEv(int64_t in);
};

} 
