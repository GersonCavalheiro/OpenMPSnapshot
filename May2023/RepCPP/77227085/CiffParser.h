

#pragma once

#include "parsers/RawParser.h" 
#include "tiff/CiffIFD.h"      
#include <memory>              

namespace rawspeed {

class Buffer;
class CameraMetaData;
class RawDecoder;

class CiffParser final : public RawParser {
std::unique_ptr<const CiffIFD> mRootIFD;

public:
explicit CiffParser(Buffer input);

void parseData();

std::unique_ptr<RawDecoder>
getDecoder(const CameraMetaData* meta = nullptr) override;
};

} 
