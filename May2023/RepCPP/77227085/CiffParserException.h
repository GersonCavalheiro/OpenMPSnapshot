

#pragma once

#include "rawspeedconfig.h" 

#include "common/RawspeedException.h"   
#include "parsers/RawParserException.h" 

namespace rawspeed {

class CiffParserException final : public RawParserException {
public:
using RawParserException::RawParserException;
};

#define ThrowCPE(...)                                                          \
ThrowExceptionHelper(rawspeed::CiffParserException, __VA_ARGS__)

} 
