

#pragma once

#include <cstddef> 

#ifndef __has_feature      
#define __has_feature(x) 0 
#endif
#ifndef __has_extension
#define __has_extension __has_feature 
#endif

#if __has_feature(address_sanitizer) || defined(__SANITIZE_ADDRESS__)
#include <sanitizer/asan_interface.h>
#endif

namespace rawspeed {

struct ASan final {
ASan() = delete;
ASan(const ASan&) = delete;
ASan(ASan&&) = delete;
ASan& operator=(const ASan&) = delete;
ASan& operator=(ASan&&) = delete;
~ASan() = delete;

static void PoisonMemoryRegion(const volatile void* addr, size_t size);
static void UnPoisonMemoryRegion(const volatile void* addr, size_t size);

static bool RegionIsPoisoned(const volatile void* addr, size_t size);
};

#if __has_feature(address_sanitizer) || defined(__SANITIZE_ADDRESS__)
inline void ASan::PoisonMemoryRegion(const volatile void* addr, size_t size) {
__asan_poison_memory_region(addr, size);
}
inline void ASan::UnPoisonMemoryRegion(const volatile void* addr, size_t size) {
__asan_unpoison_memory_region(addr, size);
}
inline bool ASan::RegionIsPoisoned(const volatile void* addr, size_t size) {
auto* beg = const_cast<void*>(addr); 
return nullptr != __asan_region_is_poisoned(beg, size);
}
#else
inline void ASan::PoisonMemoryRegion([[maybe_unused]] const volatile void* addr,
[[maybe_unused]] size_t size) {
}
inline void
ASan::UnPoisonMemoryRegion([[maybe_unused]] const volatile void* addr,
[[maybe_unused]] size_t size) {
}
inline bool ASan::RegionIsPoisoned([[maybe_unused]] const volatile void* addr,
[[maybe_unused]] size_t size) {
return false;
}
#endif

} 
