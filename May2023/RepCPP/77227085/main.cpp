

#ifndef PARSER
#error PARSER must be defined
#endif

#ifndef GETDECODER
#error GETDECODER must be defined as bool
#endif

#ifndef DECODE
#error DECODE must be defined as bool
#endif

#include "io/Buffer.h"                  
#include "io/IOException.h"             
#include "parsers/CiffParser.h"         
#include "parsers/FiffParser.h"         
#include "parsers/RawParser.h"          
#include "parsers/RawParserException.h" 
#include "parsers/TiffParser.h"         
#include <cassert>                      
#include <cstdint>                      
#include <cstdio>                       

#if GETDECODER
#include "decoders/RawDecoder.h"          
#include "decoders/RawDecoderException.h" 
#if DECODE
#include "common/RawspeedException.h" 
#include "metadata/CameraMetaData.h"  
#include <memory>                     
#endif
#endif

#if GETDECODER && DECODE
static const rawspeed::CameraMetaData metadata{};
#endif

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* Data, size_t Size);

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* Data, size_t Size) {
assert(Data);

const rawspeed::Buffer buffer(Data, Size);

try {
rawspeed::PARSER parser(buffer);

#if GETDECODER
#if DECODE
auto decoder =
#endif
parser.getDecoder();
#endif

#if DECODE
decoder->applyCrop = false;
decoder->interpolateBadPixels = false;
decoder->failOnUnknown = false;

decoder->decodeRaw();
decoder->decodeMetaData(&metadata);
#endif
} catch (const rawspeed::RawParserException&) {
return 0;
#if GETDECODER
} catch (const rawspeed::RawDecoderException&) {
return 0;
#endif
} catch (const rawspeed::IOException&) {
return 0;
#if DECODE
} catch (const rawspeed::RawspeedException&) {
return 0;
#endif
}

return 0;
}
