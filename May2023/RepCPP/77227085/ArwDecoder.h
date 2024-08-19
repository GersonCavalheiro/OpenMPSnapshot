

#pragma once

#include "common/RawImage.h"              
#include "decoders/AbstractTiffDecoder.h" 
#include "io/Buffer.h"                    
#include "io/ByteStream.h"                
#include "tiff/TiffIFD.h"                 
#include <cstdint>                        
#include <utility>                        

namespace rawspeed {

class Buffer;
class CameraMetaData;

class ArwDecoder final : public AbstractTiffDecoder {
public:
static bool isAppropriateDecoder(const TiffRootIFD* rootIFD, Buffer file);
ArwDecoder(TiffRootIFDOwner&& root, Buffer file)
: AbstractTiffDecoder(std::move(root), file) {}

RawImage decodeRawInternal() override;
void decodeMetaDataInternal(const CameraMetaData* meta) override;

private:
void ParseA100WB() const;

[[nodiscard]] int getDecoderVersion() const override { return 1; }
RawImage decodeSRF(const TiffIFD* raw);
void DecodeARW2(ByteStream input, uint32_t w, uint32_t h, uint32_t bpp);
void DecodeUncompressed(const TiffIFD* raw) const;
static void SonyDecrypt(const uint32_t* ibuf, uint32_t* obuf, uint32_t len,
uint32_t key);
void GetWB() const;
ByteStream in;
int mShiftDownScale = 0;
int mShiftDownScaleForExif = 0;
};

} 
