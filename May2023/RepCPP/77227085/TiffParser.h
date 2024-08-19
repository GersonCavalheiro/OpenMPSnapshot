

#pragma once

#include "decoders/RawDecoder.h" 
#include "io/Buffer.h"           
#include "parsers/RawParser.h"   
#include "tiff/TiffIFD.h"        
#include <array>                 
#include <memory>                
#include <utility>               

namespace rawspeed {

class Buffer;
class CameraMetaData;
class RawDecoder;

class TiffParser final : public RawParser {
public:
explicit TiffParser(Buffer file);

std::unique_ptr<RawDecoder>
getDecoder(const CameraMetaData* meta = nullptr) override;

static TiffRootIFDOwner parse(TiffIFD* parent, Buffer data);

static std::unique_ptr<RawDecoder> makeDecoder(TiffRootIFDOwner root,
Buffer data);

template <class Decoder>
static std::unique_ptr<RawDecoder> constructor(TiffRootIFDOwner&& root,
Buffer data);
using checker_t = bool (*)(const TiffRootIFD* root, Buffer data);
using constructor_t = std::unique_ptr<RawDecoder> (*)(TiffRootIFDOwner&& root,
Buffer data);
static const std::array<std::pair<checker_t, constructor_t>, 16> Map;
};

} 
