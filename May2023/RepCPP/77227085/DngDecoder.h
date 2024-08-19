

#pragma once

#include "common/RawImage.h"              
#include "decoders/AbstractTiffDecoder.h" 
#include "tiff/TiffIFD.h"                 
#include <cstdint>                        
#include <optional>                       
#include <vector>                         

namespace rawspeed {

class Buffer;
class CameraMetaData;
class iRectangle2D;
struct DngTilingDescription;

class DngDecoder final : public AbstractTiffDecoder {
public:
static bool isAppropriateDecoder(const TiffRootIFD* rootIFD, Buffer file);
DngDecoder(TiffRootIFDOwner&& rootIFD, Buffer file);

RawImage decodeRawInternal() override;
void decodeMetaDataInternal(const CameraMetaData* meta) override;
void checkSupportInternal(const CameraMetaData* meta) override;

private:
[[nodiscard]] int getDecoderVersion() const override { return 0; }
bool mFixLjpeg;
static void dropUnsuportedChunks(std::vector<const TiffIFD*>* data);
std::optional<iRectangle2D> parseACTIVEAREA(const TiffIFD* raw) const;
void parseCFA(const TiffIFD* raw) const;
void parseColorMatrix() const;
DngTilingDescription getTilingDescription(const TiffIFD* raw) const;
void decodeData(const TiffIFD* raw, uint32_t sample_format) const;
void handleMetadata(const TiffIFD* raw);
bool decodeMaskedAreas(const TiffIFD* raw) const;
bool decodeBlackLevels(const TiffIFD* raw) const;
void setBlack(const TiffIFD* raw) const;

int bps = -1;
int compression = -1;
};

} 
