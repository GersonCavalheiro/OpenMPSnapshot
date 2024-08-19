

#pragma once

#include "adt/CroppedArray1DRef.h" 
#include "adt/CroppedArray2DRef.h" 
#include <cstddef>                 

#ifndef __has_feature      
#define __has_feature(x) 0 
#endif
#ifndef __has_extension
#define __has_extension __has_feature 
#endif

#if __has_feature(memory_sanitizer) || defined(__SANITIZE_MEMORY__)
#include <sanitizer/msan_interface.h>
#endif

namespace rawspeed {

struct MSan final {
MSan() = delete;
MSan(const MSan&) = delete;
MSan(MSan&&) = delete;
MSan& operator=(const MSan&) = delete;
MSan& operator=(MSan&&) = delete;
~MSan() = delete;

private:
static void Allocated(const void* addr, size_t size);
static void Allocated(CroppedArray1DRef<std::byte> row);

public:
template <typename T> static void Allocated(const T& elt);
static void Allocated(CroppedArray2DRef<std::byte> frame);

private:
static void CheckMemIsInitialized(const void* addr, size_t size);
static void CheckMemIsInitialized(CroppedArray1DRef<std::byte> row);

public:
static void CheckMemIsInitialized(CroppedArray2DRef<std::byte> frame);
};

#if __has_feature(memory_sanitizer) || defined(__SANITIZE_MEMORY__)
inline void MSan::Allocated(const void* addr, size_t size) {
__msan_allocated_memory(addr, size);
}
#else
inline void MSan::Allocated([[maybe_unused]] const void* addr,
[[maybe_unused]] size_t size) {
}
#endif

template <typename T> inline void MSan::Allocated(const T& elt) {
Allocated(&elt, sizeof(T));
}
inline void MSan::Allocated(CroppedArray1DRef<std::byte> row) {
MSan::Allocated(row.begin(), row.size());
}
inline void MSan::Allocated(CroppedArray2DRef<std::byte> frame) {
for (int row = 0; row < frame.croppedHeight; row++)
Allocated(frame[row]);
}

#if __has_feature(memory_sanitizer) || defined(__SANITIZE_MEMORY__)
inline void MSan::CheckMemIsInitialized(const void* addr, size_t size) {
__msan_check_mem_is_initialized(addr, size);
}
#else
inline void MSan::CheckMemIsInitialized([[maybe_unused]] const void* addr,
[[maybe_unused]] size_t size) {
}
#endif

inline void MSan::CheckMemIsInitialized(CroppedArray1DRef<std::byte> row) {
MSan::CheckMemIsInitialized(row.begin(), row.size());
}
inline void MSan::CheckMemIsInitialized(CroppedArray2DRef<std::byte> frame) {
for (int row = 0; row < frame.croppedHeight; row++)
CheckMemIsInitialized(frame[row]);
}

} 
