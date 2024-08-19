

#pragma once

#include "rawspeedconfig.h" 

#include "common/RawspeedException.h" 

namespace rawspeed {

class CameraMetadataException final : public RawspeedException {
public:
using RawspeedException::RawspeedException;
};

#define ThrowCME(...)                                                          \
ThrowExceptionHelper(rawspeed::CameraMetadataException, __VA_ARGS__)

} 
