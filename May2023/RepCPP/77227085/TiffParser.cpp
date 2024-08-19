

#include "parsers/TiffParser.h"
#include "adt/NORangesSet.h"             
#include "decoders/ArwDecoder.h"         
#include "decoders/Cr2Decoder.h"         
#include "decoders/DcrDecoder.h"         
#include "decoders/DcsDecoder.h"         
#include "decoders/DngDecoder.h"         
#include "decoders/ErfDecoder.h"         
#include "decoders/IiqDecoder.h"         
#include "decoders/KdcDecoder.h"         
#include "decoders/MefDecoder.h"         
#include "decoders/MosDecoder.h"         
#include "decoders/NefDecoder.h"         
#include "decoders/OrfDecoder.h"         
#include "decoders/PefDecoder.h"         
#include "decoders/Rw2Decoder.h"         
#include "decoders/SrwDecoder.h"         
#include "decoders/ThreefrDecoder.h"     
#include "io/Buffer.h"                   
#include "io/ByteStream.h"               
#include "io/Endianness.h"               
#include "parsers/TiffParserException.h" 
#include <cassert>                       
#include <cstdint>                       
#include <memory>                        
#include <tuple>                         
#include <vector>                        

namespace rawspeed {
class RawDecoder;

TiffParser::TiffParser(Buffer file) : RawParser(file) {}

std::unique_ptr<RawDecoder> TiffParser::getDecoder(const CameraMetaData* meta) {
return TiffParser::makeDecoder(TiffParser::parse(nullptr, mInput), mInput);
}

TiffRootIFDOwner TiffParser::parse(TiffIFD* parent, Buffer data) {
ByteStream bs(DataBuffer(data, Endianness::unknown));
bs.setByteOrder(getTiffByteOrder(bs, 0, "TIFF header"));
bs.skipBytes(2);

if (uint16_t magic = bs.getU16();
magic != 42 && magic != 0x4f52 && magic != 0x5352 &&
magic != 0x55) 
ThrowTPE("Not a TIFF file (magic 42)");

auto root = std::make_unique<TiffRootIFD>(
parent, nullptr, bs,
UINT32_MAX); 

NORangesSet<Buffer> ifds;

for (uint32_t IFDOffset = bs.getU32(); IFDOffset;
IFDOffset = root->getSubIFDs().back()->getNextIFD()) {
root->add(std::make_unique<TiffIFD>(root.get(), &ifds, bs, IFDOffset));
}

return root;
}

std::unique_ptr<RawDecoder> TiffParser::makeDecoder(TiffRootIFDOwner root,
Buffer data) {
if (!root)
ThrowTPE("TiffIFD is null.");

for (const auto& decoder : Map) {
checker_t dChecker = nullptr;
constructor_t dConstructor = nullptr;

std::tie(dChecker, dConstructor) = decoder;

assert(dChecker);
assert(dConstructor);

if (!dChecker(root.get(), data))
continue;

return dConstructor(std::move(root), data);
}

ThrowTPE("No decoder found. Sorry.");
}

template <class Decoder>
std::unique_ptr<RawDecoder> TiffParser::constructor(TiffRootIFDOwner&& root,
Buffer data) {
return std::make_unique<Decoder>(std::move(root), data);
}

#define DECODER(name)                                                          \
{                                                                            \
std::make_pair(                                                            \
static_cast<TiffParser::checker_t>(&name::isAppropriateDecoder),       \
&constructor<name>)                                                    \
}

const std::array<std::pair<TiffParser::checker_t, TiffParser::constructor_t>,
16>
TiffParser::Map = {{
DECODER(DngDecoder),
DECODER(MosDecoder),
DECODER(IiqDecoder),
DECODER(Cr2Decoder),
DECODER(NefDecoder),
DECODER(OrfDecoder),
DECODER(ArwDecoder),
DECODER(PefDecoder),
DECODER(Rw2Decoder),
DECODER(SrwDecoder),
DECODER(MefDecoder),
DECODER(DcrDecoder),
DECODER(DcsDecoder),
DECODER(KdcDecoder),
DECODER(ErfDecoder),
DECODER(ThreefrDecoder),

}};

} 
