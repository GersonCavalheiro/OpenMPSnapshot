

#pragma once

#include "common/Common.h"       
#include "common/RawImage.h"     
#include "decoders/RawDecoder.h" 
#include <cstdint>               
#include <functional>            
#include <map>                   
#include <string>                

namespace rawspeed {

class Buffer;
class Camera;
class CameraMetaData;

class NakedDecoder final : public RawDecoder {
const Camera* cam;

uint32_t width{0};
uint32_t height{0};
uint32_t filesize{0};
uint32_t bits{0};
uint32_t offset{0};
BitOrder bo{BitOrder::MSB16};

void parseHints();

public:
NakedDecoder(Buffer file, const Camera* c);
RawImage decodeRawInternal() override;
void checkSupportInternal(const CameraMetaData* meta) override;
void decodeMetaDataInternal(const CameraMetaData* meta) override;

private:
[[nodiscard]] int getDecoderVersion() const override { return 0; }
};

} 
