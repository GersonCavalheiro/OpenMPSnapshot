

#pragma once

#include "common/RawImage.h"     
#include "decoders/RawDecoder.h" 
#include "io/Buffer.h"           
#include "tiff/TiffIFD.h"        
#include <array>                 
#include <cmath>                 
#include <cstdint>               

namespace rawspeed {

class CameraMetaData;

class MrwDecoder final : public RawDecoder {
TiffRootIFDOwner rootIFD;

uint32_t raw_width = 0;
uint32_t raw_height = 0;
Buffer imageData;
uint32_t bpp = 0;
uint32_t packed = 0;
std::array<float, 4> wb_coeffs = {{NAN, NAN, NAN, NAN}};

public:
explicit MrwDecoder(Buffer file);
RawImage decodeRawInternal() override;
void checkSupportInternal(const CameraMetaData* meta) override;
void decodeMetaDataInternal(const CameraMetaData* meta) override;
static int isMRW(Buffer input);

private:
[[nodiscard]] int getDecoderVersion() const override { return 0; }
void parseHeader();
};

} 
