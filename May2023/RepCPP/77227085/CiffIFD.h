

#pragma once

#include "rawspeedconfig.h"  
#include "adt/NORangesSet.h" 
#include "tiff/CiffEntry.h"  
#include "tiff/CiffTag.h"    
#include <cstdint>           
#include <map>               
#include <memory>            
#include <string>            
#include <vector>            

namespace rawspeed {

class Buffer;
class ByteStream;
class CiffEntry;
template <typename T> class NORangesSet;

class CiffIFD final {
CiffIFD* const parent;

std::vector<std::unique_ptr<const CiffIFD>> mSubIFD;
std::map<CiffTag, std::unique_ptr<const CiffEntry>> mEntry;

int subIFDCount = 0;
int subIFDCountRecursive = 0;

void recursivelyIncrementSubIFDCount();
void checkSubIFDs(int headroom) const;
void recursivelyCheckSubIFDs(int headroom) const;

struct Limits final {
static constexpr int Depth = 3 + 1;

static constexpr int SubIFDCount = 4 * 2;

static constexpr int RecursiveSubIFDCount = 6 * 2;
};

void add(std::unique_ptr<CiffIFD> subIFD);
void add(std::unique_ptr<CiffEntry> entry);

void parseIFDEntry(NORangesSet<Buffer>* valueDatas, ByteStream valueData,
ByteStream& dirEntries);

template <typename Lambda>
std::vector<const CiffIFD*>
RAWSPEED_READONLY getIFDsWithTagIf(CiffTag tag, const Lambda& f) const;

template <typename Lambda>
const CiffEntry* RAWSPEED_READONLY getEntryRecursiveIf(CiffTag tag,
const Lambda& f) const;

public:
explicit CiffIFD(CiffIFD* parent);
CiffIFD(CiffIFD* parent, ByteStream directory);

[[nodiscard]] std::vector<const CiffIFD*>
RAWSPEED_READONLY getIFDsWithTag(CiffTag tag) const;
[[nodiscard]] std::vector<const CiffIFD*> RAWSPEED_READONLY
getIFDsWithTagWhere(CiffTag tag, uint32_t isValue) const;
[[nodiscard]] std::vector<const CiffIFD*> RAWSPEED_READONLY
getIFDsWithTagWhere(CiffTag tag, const std::string& isValue) const;

[[nodiscard]] bool RAWSPEED_READONLY hasEntry(CiffTag tag) const;
[[nodiscard]] bool RAWSPEED_READONLY hasEntryRecursive(CiffTag tag) const;

[[nodiscard]] const CiffEntry* RAWSPEED_READONLY getEntry(CiffTag tag) const;
[[nodiscard]] const CiffEntry* RAWSPEED_READONLY
getEntryRecursive(CiffTag tag) const;
[[nodiscard]] const CiffEntry* RAWSPEED_READONLY
getEntryRecursiveWhere(CiffTag tag, uint32_t isValue) const;
[[nodiscard]] const CiffEntry* RAWSPEED_READONLY
getEntryRecursiveWhere(CiffTag tag, const std::string& isValue) const;
};

} 
