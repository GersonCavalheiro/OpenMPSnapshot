

#pragma once

#include "rawspeedconfig.h" 

#include "common/RawspeedException.h"     
#include "decoders/RawDecoderException.h" 

namespace rawspeed {

class FileIOException final : public RawDecoderException {
public:
using RawDecoderException::RawDecoderException;
};

#define ThrowFIE(...)                                                          \
ThrowExceptionHelper(rawspeed::FileIOException, __VA_ARGS__)

} 
