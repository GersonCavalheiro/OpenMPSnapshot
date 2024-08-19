

#pragma once

#include "rawspeedconfig.h" 

#include "common/RawspeedException.h" 

namespace rawspeed {

class RawDecoderException : public RawspeedException {
public:
using RawspeedException::RawspeedException;
};

#define ThrowRDE(...)                                                          \
ThrowExceptionHelper(rawspeed::RawDecoderException, __VA_ARGS__)

} 
