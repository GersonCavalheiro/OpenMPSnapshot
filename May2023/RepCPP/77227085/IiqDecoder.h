

#pragma once

#include "common/RawImage.h"              
#include "decoders/AbstractTiffDecoder.h" 
#include "io/Buffer.h"                    
#include "tiff/TiffIFD.h"                 
#include <cstdint>                        
#include <utility>                        
#include <vector>                         

namespace rawspeed {

class Buffer;
class ByteStream;
class CameraMetaData;
struct PhaseOneStrip;

class IiqDecoder final : public AbstractTiffDecoder {
struct IiqOffset {
uint32_t n;
uint32_t offset;

IiqOffset() = default;
IiqOffset(uint32_t block, uint32_t offset_) : n(block), offset(offset_) {}
};

static std::vector<PhaseOneStrip>
computeSripes(Buffer raw_data, std::vector<IiqOffset> offsets,
uint32_t height);

public:
static bool isAppropriateDecoder(Buffer file);
static bool isAppropriateDecoder(const TiffRootIFD* rootIFD, Buffer file);

IiqDecoder(TiffRootIFDOwner&& rootIFD, Buffer file)
: AbstractTiffDecoder(std::move(rootIFD), file) {}

RawImage decodeRawInternal() override;
void checkSupportInternal(const CameraMetaData* meta) override;
void decodeMetaDataInternal(const CameraMetaData* meta) override;

private:
[[nodiscard]] int getDecoderVersion() const override { return 0; }
uint32_t black_level = 0;
void CorrectPhaseOneC(ByteStream meta_data, uint32_t split_row,
uint32_t split_col);
void CorrectQuadrantMultipliersCombined(ByteStream data, uint32_t split_row,
uint32_t split_col) const;
enum class IiqCorr;
void PhaseOneFlatField(ByteStream data, IiqCorr corr) const;
void correctSensorDefects(ByteStream data) const;
void correctBadColumn(uint16_t col) const;
void handleBadPixel(uint16_t col, uint16_t row) const;
};

} 
