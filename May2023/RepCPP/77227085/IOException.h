

#pragma once

#include "rawspeedconfig.h" 

#include "common/RawspeedException.h" 

namespace rawspeed {

class IOException final : public RawspeedException {
public:
using RawspeedException::RawspeedException;
};

#define ThrowIOE(...) ThrowExceptionHelper(rawspeed::IOException, __VA_ARGS__)

} 
