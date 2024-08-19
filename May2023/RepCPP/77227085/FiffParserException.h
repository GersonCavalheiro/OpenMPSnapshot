

#pragma once

#include "rawspeedconfig.h" 

#include "common/RawspeedException.h"   
#include "parsers/RawParserException.h" 

namespace rawspeed {

class FiffParserException final : public RawParserException {
public:
using RawParserException::RawParserException;
};

#define ThrowFPE(...)                                                          \
ThrowExceptionHelper(rawspeed::FiffParserException, __VA_ARGS__)

} 
