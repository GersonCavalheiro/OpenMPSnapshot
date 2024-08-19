

#pragma once

#include "io/Buffer.h" 
#include <memory>      

namespace rawspeed {

class CameraMetaData;
class RawDecoder;

class RawParser {
public:
explicit RawParser(Buffer inputData) : mInput(inputData) {}
virtual ~RawParser() = default;

virtual std::unique_ptr<RawDecoder>
getDecoder(const CameraMetaData* meta = nullptr);

protected:
Buffer mInput;
};

} 
