

#pragma once

#include "parsers/RawParser.h" 
#include "tiff/TiffIFD.h"      
#include <memory>              

namespace rawspeed {

class Buffer;
class CameraMetaData;
class RawDecoder;

class FiffParser final : public RawParser {
TiffRootIFDOwner rootIFD;

public:
explicit FiffParser(Buffer input);

void parseData();
std::unique_ptr<RawDecoder>
getDecoder(const CameraMetaData* meta = nullptr) override;
};

} 
