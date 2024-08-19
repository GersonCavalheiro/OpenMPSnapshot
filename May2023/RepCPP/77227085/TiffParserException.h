

#pragma once

#include "rawspeedconfig.h" 

#include "common/RawspeedException.h"   
#include "parsers/RawParserException.h" 

namespace rawspeed {

class TiffParserException final : public RawParserException {
public:
using RawParserException::RawParserException;
};

#define ThrowTPE(...)                                                          \
ThrowExceptionHelper(rawspeed::TiffParserException, __VA_ARGS__)

} 
