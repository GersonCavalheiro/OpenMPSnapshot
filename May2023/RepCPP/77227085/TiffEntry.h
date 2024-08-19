

#pragma once

#include "rawspeedconfig.h"   
#include "adt/NotARational.h" 
#include "io/ByteStream.h"    
#include "tiff/TiffTag.h"     
#include <algorithm>          
#include <array>              
#include <cstdint>            
#include <string>             
#include <vector>             

namespace rawspeed {

class DataBuffer;
class TiffIFD;
class Buffer;


enum class TiffDataType {
NOTYPE = 0,     
BYTE = 1,       
ASCII = 2,      
SHORT = 3,      
LONG = 4,       
RATIONAL = 5,   
SBYTE = 6,      
UNDEFINED = 7,  
SSHORT = 8,     
SLONG = 9,      
SRATIONAL = 10, 
FLOAT = 11,     
DOUBLE = 12,    
OFFSET = 13,    
};

class TiffEntry {
TiffIFD* parent;
ByteStream data;

friend class TiffIFD;

template <typename T, T (TiffEntry::*getter)(uint32_t index) const>
[[nodiscard]] std::vector<T> getArray(uint32_t count_) const {
std::vector<T> res(count_);
for (uint32_t i = 0; i < count_; ++i)
res[i] = (this->*getter)(i);
return res;
}

protected:
void setData(ByteStream data_);

public:
TiffTag tag;
TiffDataType type;
uint32_t count;

TiffEntry(TiffIFD* parent, TiffTag tag, TiffDataType type, uint32_t count,
ByteStream data);
TiffEntry(TiffIFD* parent, ByteStream& bs);

virtual ~TiffEntry() = default;

[[nodiscard]] bool RAWSPEED_READONLY isFloat() const;
[[nodiscard]] bool RAWSPEED_READONLY isRational() const;
[[nodiscard]] bool RAWSPEED_READONLY isSRational() const;
[[nodiscard]] bool RAWSPEED_READONLY isInt() const;
[[nodiscard]] bool RAWSPEED_READONLY isString() const;
[[nodiscard]] uint8_t getByte(uint32_t index = 0) const;
[[nodiscard]] uint32_t getU32(uint32_t index = 0) const;
[[nodiscard]] int32_t getI32(uint32_t index = 0) const;
[[nodiscard]] uint16_t getU16(uint32_t index = 0) const;
[[nodiscard]] int16_t getI16(uint32_t index = 0) const;
[[nodiscard]] NotARational<uint32_t> getRational(uint32_t index = 0) const;
[[nodiscard]] NotARational<int32_t> getSRational(uint32_t index = 0) const;
[[nodiscard]] float getFloat(uint32_t index = 0) const;
[[nodiscard]] std::string getString() const;

[[nodiscard]] inline std::vector<uint16_t>
getU16Array(uint32_t count_) const {
return getArray<uint16_t, &TiffEntry::getU16>(count_);
}

[[nodiscard]] inline std::vector<uint32_t>
getU32Array(uint32_t count_) const {
return getArray<uint32_t, &TiffEntry::getU32>(count_);
}

[[nodiscard]] inline std::vector<float> getFloatArray(uint32_t count_) const {
return getArray<float, &TiffEntry::getFloat>(count_);
}

[[nodiscard]] inline std::vector<NotARational<uint32_t>>
getRationalArray(uint32_t count_) const {
return getArray<NotARational<uint32_t>, &TiffEntry::getRational>(count_);
}

[[nodiscard]] inline std::vector<NotARational<int32_t>>
getSRationalArray(uint32_t count_) const {
return getArray<NotARational<int>, &TiffEntry::getSRational>(count_);
}

[[nodiscard]] ByteStream getData() const { return data; }

[[nodiscard]] DataBuffer getRootIfdData() const;

protected:
static const std::array<uint32_t, 14> datashifts;
};

class TiffEntryWithData : public TiffEntry {
const std::vector<uint8_t> data;

public:
TiffEntryWithData(TiffIFD* parent, TiffTag tag, TiffDataType type,
uint32_t count, Buffer mirror);
};

} 
