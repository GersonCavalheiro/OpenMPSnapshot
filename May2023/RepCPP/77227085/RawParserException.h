

#pragma once

#include "rawspeedconfig.h" 

#include "common/RawspeedException.h" 

namespace rawspeed {

class RawParserException : public RawspeedException {
public:
using RawspeedException::RawspeedException;
};

#define ThrowRPE(...)                                                          \
ThrowExceptionHelper(rawspeed::RawParserException, __VA_ARGS__)

} 
