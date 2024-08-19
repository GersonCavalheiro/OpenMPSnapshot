

#pragma once

#include "rawspeedconfig.h"  
#include "adt/NORangesSet.h" 
#include "io/ByteStream.h"   
#include "tiff/CiffTag.h"    
#include <cstdint>           
#include <string>            
#include <string_view>       
#include <vector>            

namespace rawspeed {

class Buffer;
class CiffIFD; 
template <typename T> class NORangesSet;


enum class CiffDataType {
BYTE = 0x0000,  
ASCII = 0x0800, 
SHORT = 0x1000, 
LONG = 0x1800,  
MIX = 0x2000,   
SUB1 = 0x2800,  
SUB2 = 0x3000,  

};

class CiffEntry {
friend class CiffIFD;

ByteStream data;

public:
explicit CiffEntry(NORangesSet<Buffer>* valueDatas, ByteStream valueData,
ByteStream dirEntry);

[[nodiscard]] ByteStream getData() const { return data; }

[[nodiscard]] uint8_t getByte(uint32_t num = 0) const;
[[nodiscard]] uint32_t getU32(uint32_t num = 0) const;
[[nodiscard]] uint16_t getU16(uint32_t num = 0) const;

[[nodiscard]] std::string_view getString() const;
[[nodiscard]] std::vector<std::string> getStrings() const;

[[nodiscard]] uint32_t RAWSPEED_READONLY getElementSize() const;
[[nodiscard]] uint32_t RAWSPEED_READONLY getElementShift() const;

CiffTag tag;
CiffDataType type;
uint32_t count;

[[nodiscard]] bool RAWSPEED_READONLY isInt() const;
[[nodiscard]] bool RAWSPEED_READONLY isString() const;
};

} 
