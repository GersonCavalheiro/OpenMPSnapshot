

#pragma once

#include "rawspeedconfig.h"              
#include "adt/NORangesSet.h"             
#include "common/RawspeedException.h"    
#include "io/Buffer.h"                   
#include "io/ByteStream.h"               
#include "io/Endianness.h"               
#include "parsers/TiffParserException.h" 
#include "tiff/TiffEntry.h"              
#include "tiff/TiffTag.h"                
#include <cstdint>                       
#include <map>                           
#include <memory>                        
#include <string>                        
#include <vector>                        

namespace rawspeed {

class TiffIFD;
class TiffRootIFD;
template <typename T> class NORangesSet;

using TiffIFDOwner = std::unique_ptr<TiffIFD>;
using TiffRootIFDOwner = std::unique_ptr<TiffRootIFD>;
using TiffEntryOwner = std::unique_ptr<TiffEntry>;

class TiffIFD {
uint32_t nextIFD = 0;

TiffIFD* const parent;

std::vector<TiffIFDOwner> subIFDs;

int subIFDCount = 0;
int subIFDCountRecursive = 0;

std::map<TiffTag, TiffEntryOwner> entries;

friend class TiffEntry;
friend class FiffParser;
friend class TiffParser;

void recursivelyIncrementSubIFDCount();
void checkSubIFDs(int headroom) const;
void recursivelyCheckSubIFDs(int headroom) const;

void add(TiffIFDOwner subIFD);
void add(TiffEntryOwner entry);
TiffRootIFDOwner parseMakerNote(NORangesSet<Buffer>* ifds,
const TiffEntry* t);
void parseIFDEntry(NORangesSet<Buffer>* ifds, ByteStream& bs);

struct Limits final {
static constexpr int Depth = 4 + 1;

static constexpr int SubIFDCount = 5 * 2;

static constexpr int RecursiveSubIFDCount = 14 * 2;
};

public:
explicit TiffIFD(TiffIFD* parent);

TiffIFD(TiffIFD* parent, NORangesSet<Buffer>* ifds, DataBuffer data,
uint32_t offset);

virtual ~TiffIFD() = default;

TiffIFD(const TiffIFD&) = delete;
TiffIFD& operator=(const TiffIFD&) = delete;

[[nodiscard]] uint32_t getNextIFD() const { return nextIFD; }
[[nodiscard]] std::vector<const TiffIFD*> getIFDsWithTag(TiffTag tag) const;
[[nodiscard]] const TiffIFD* getIFDWithTag(TiffTag tag,
uint32_t index = 0) const;
[[nodiscard]] TiffEntry* getEntry(TiffTag tag) const;
[[nodiscard]] TiffEntry* RAWSPEED_READONLY
getEntryRecursive(TiffTag tag) const;
[[nodiscard]] bool RAWSPEED_READONLY hasEntry(TiffTag tag) const {
return entries.find(tag) != entries.end();
}
[[nodiscard]] bool hasEntryRecursive(TiffTag tag) const {
return getEntryRecursive(tag) != nullptr;
}

[[nodiscard]] const std::vector<TiffIFDOwner>& getSubIFDs() const {
return subIFDs;
}
};

struct TiffID {
std::string make;
std::string model;
};

class TiffRootIFD final : public TiffIFD {
public:
const DataBuffer rootBuffer;

TiffRootIFD(TiffIFD* parent_, NORangesSet<Buffer>* ifds, DataBuffer data,
uint32_t offset)
: TiffIFD(parent_, ifds, data, offset), rootBuffer(data) {}

[[nodiscard]] TiffID getID() const;
};

inline Endianness getTiffByteOrder(ByteStream bs, uint32_t pos,
const char* context = "") {
if (bs.hasPatternAt("II", 2, pos))
return Endianness::little;
if (bs.hasPatternAt("MM", 2, pos))
return Endianness::big;

ThrowTPE("Failed to parse TIFF endianness information in %s.", context);
}

} 
