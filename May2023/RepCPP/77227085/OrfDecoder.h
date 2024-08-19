

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

class OrfDecoder final : public AbstractTiffDecoder {
[[nodiscard]] ByteStream handleSlices() const;

public:
static bool isAppropriateDecoder(const TiffRootIFD* rootIFD, Buffer file);
OrfDecoder(TiffRootIFDOwner&& root, Buffer file)
: AbstractTiffDecoder(std::move(root), file) {}

RawImage decodeRawInternal() override;
void decodeMetaDataInternal(const CameraMetaData* meta) override;

private:
void parseCFA() const;

[[nodiscard]] int getDecoderVersion() const override { return 3; }
void decodeUncompressedInterleaved(ByteStream s, uint32_t w, uint32_t h,
uint32_t size) const;
[[nodiscard]] bool decodeUncompressed(ByteStream s, uint32_t w, uint32_t h,
uint32_t size) const;
};

} 
