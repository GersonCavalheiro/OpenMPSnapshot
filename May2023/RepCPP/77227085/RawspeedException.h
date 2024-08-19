

#pragma once

#include "rawspeedconfig.h" 
#include "common/Common.h"  
#include <array>            
#include <cstdarg>          
#include <cstdio>           
#include <stdexcept>        

namespace rawspeed {

template <typename T>
[[noreturn]] void RAWSPEED_UNLIKELY_FUNCTION RAWSPEED_NOINLINE
__attribute__((noreturn, format(printf, 1, 2)))
ThrowException(const char* fmt, ...) {
static constexpr size_t bufSize = 8192;
#if defined(HAVE_CXX_THREAD_LOCAL)
static thread_local std::array<char, bufSize> buf;
#elif defined(HAVE_GCC_THREAD_LOCAL)
static __thread char buf[bufSize];
#else
#pragma message                                                                \
"Don't have thread-local-storage! Exception text may be garbled if used multithreaded"
static char buf[bufSize];
#endif

va_list val;
va_start(val, fmt);
vsnprintf(buf.data(), sizeof(buf), fmt, val);
va_end(val);
writeLog(DEBUG_PRIO::EXTRA, "EXCEPTION: %s", buf.data());
throw T(buf.data());
}

class RawspeedException : public std::runtime_error {
private:
static void RAWSPEED_UNLIKELY_FUNCTION RAWSPEED_NOINLINE
log(const char* msg) {
writeLog(DEBUG_PRIO::EXTRA, "EXCEPTION: %s", msg);
}

public:
explicit RAWSPEED_UNLIKELY_FUNCTION RAWSPEED_NOINLINE
RawspeedException(const char* msg)
: std::runtime_error(msg) {
log(msg);
}
};

#ifdef XSTR
#undef XSTR
#endif
#define XSTR(a) #a

#ifdef STR
#undef STR
#endif
#define STR(a) XSTR(a)

#ifndef DEBUG
#define ThrowExceptionHelper(CLASS, fmt, ...)                                  \
rawspeed::ThrowException<CLASS>("%s, line " STR(__LINE__) ": " fmt,          \
__PRETTY_FUNCTION__, ##__VA_ARGS__)
#else
#define ThrowExceptionHelper(CLASS, fmt, ...)                                  \
rawspeed::ThrowException<CLASS>(__FILE__ ":" STR(__LINE__) ": %s: " fmt,     \
__PRETTY_FUNCTION__, ##__VA_ARGS__)
#endif

#define ThrowRSE(...)                                                          \
ThrowExceptionHelper(rawspeed::RawspeedException, __VA_ARGS__)

} 
