

#pragma once

#include "AddressSanitizer.h" 
#include "adt/Invariant.h"    
#include "common/Common.h"    
#include "io/Endianness.h"    
#include "io/IOException.h"   
#include <cassert>            
#include <cstdint>            
#include <memory>             
#include <utility>            

namespace rawspeed {


class Buffer {
public:
using size_type = uint32_t;

protected:
const uint8_t* data = nullptr;

private:
size_type size = 0;

public:
Buffer() = default;

explicit Buffer(const uint8_t* data_, size_type size_)
: data(data_), size(size_) {
assert(!ASan::RegionIsPoisoned(data, size));
}

[[nodiscard]] Buffer getSubView(size_type offset, size_type size_) const {
if (!isValid(0, offset))
ThrowIOE("Buffer overflow: image file may be truncated");

return Buffer(getData(offset, size_), size_);
}

[[nodiscard]] Buffer getSubView(size_type offset) const {
if (!isValid(0, offset))
ThrowIOE("Buffer overflow: image file may be truncated");

size_type newSize = size - offset;
return getSubView(offset, newSize);
}

[[nodiscard]] const uint8_t* getData(size_type offset,
size_type count) const {
if (!isValid(offset, count))
ThrowIOE("Buffer overflow: image file may be truncated");

invariant(data);

return data + offset;
}

uint8_t operator[](size_type offset) const { return *getData(offset, 1); }

[[nodiscard]] const uint8_t* begin() const {
invariant(data);
return data;
}
[[nodiscard]] const uint8_t* end() const {
invariant(data);
return data + size;
}

template <typename T>
[[nodiscard]] inline T get(bool inNativeByteOrder, size_type offset,
size_type index = 0) const {
return getByteSwapped<T>(
getData(offset + index * static_cast<size_type>(sizeof(T)),
static_cast<size_type>(sizeof(T))),
!inNativeByteOrder);
}

[[nodiscard]] inline size_type RAWSPEED_READONLY getSize() const {
return size;
}

[[nodiscard]] inline bool isValid(size_type offset,
size_type count = 1) const {
return static_cast<uint64_t>(offset) + count <= static_cast<uint64_t>(size);
}
};

inline bool operator<(Buffer lhs, Buffer rhs) {
return std::pair(lhs.begin(), lhs.end()) < std::pair(rhs.begin(), rhs.end());
}


class DataBuffer : public Buffer {

Endianness endianness = Endianness::little;

public:
DataBuffer() = default;

explicit DataBuffer(Buffer data_, Endianness endianness_)
: Buffer(data_), endianness(endianness_) {}

template <typename T>
[[nodiscard]] inline T get(size_type offset, size_type index = 0) const {
assert(Endianness::unknown != endianness);
assert(Endianness::little == endianness || Endianness::big == endianness);

return Buffer::get<T>(getHostEndianness() == endianness, offset, index);
}

[[nodiscard]] inline Endianness getByteOrder() const { return endianness; }

inline Endianness setByteOrder(Endianness endianness_) {
std::swap(endianness, endianness_);
return endianness_;
}
};

} 
