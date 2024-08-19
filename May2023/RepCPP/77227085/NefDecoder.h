

#pragma once

#include "common/RawImage.h"              
#include "decoders/AbstractTiffDecoder.h" 
#include "decoders/RawDecoder.h"          
#include "io/Buffer.h"                    
#include "tiff/TiffIFD.h"                 
#include <array>                          
#include <cstdint>                        
#include <string>                         
#include <utility>                        
#include <vector>                         

namespace rawspeed {

class Buffer;
class ByteStream;
class CameraMetaData;
class iPoint2D;

class NefDecoder final : public AbstractTiffDecoder {
public:
static bool isAppropriateDecoder(const TiffRootIFD* rootIFD, Buffer file);
NefDecoder(TiffRootIFDOwner&& root, Buffer file)
: AbstractTiffDecoder(std::move(root), file) {}

RawImage decodeRawInternal() override;
void decodeMetaDataInternal(const CameraMetaData* meta) override;
void checkSupportInternal(const CameraMetaData* meta) override;

private:
struct NefSlice final : RawSlice {};

[[nodiscard]] int getDecoderVersion() const override { return 5; }
[[nodiscard]] bool D100IsCompressed(uint32_t offset) const;
static bool NEFIsUncompressed(const TiffIFD* raw);
static bool NEFIsUncompressedRGB(const TiffIFD* raw);
void DecodeUncompressed() const;
void DecodeD100Uncompressed() const;
void DecodeSNefUncompressed() const;
void readCoolpixSplitRaw(ByteStream input, const iPoint2D& size,
const iPoint2D& offset, int inputPitch) const;
void DecodeNikonSNef(ByteStream input) const;
[[nodiscard]] int getBitPerSample() const;
[[nodiscard]] std::string getMode() const;
[[nodiscard]] std::string getExtendedMode(const std::string& mode) const;
static std::vector<uint16_t> gammaCurve(double pwr, double ts, int mode,
int imax);

static const std::array<uint8_t, 256> serialmap;
static const std::array<uint8_t, 256> keymap;
};

} 
