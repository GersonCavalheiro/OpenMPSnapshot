
#ifndef CCGL_BASIC_H
#define CCGL_BASIC_H


#ifndef NDEBUG
#ifndef _DEBUG
#define _DEBUG
#endif 
#endif 


#if defined(_WIN64) || defined(__x86_64) || defined(__LP64__)
#define CPP_64
#endif


#if defined _MSC_VER
#define CPP_MSVC
#endif 


#if defined(__INTEL_COMPILER) || defined(__ICL) || defined(__ICC)
#define CPP_ICC
#endif 


#if defined(__GNUC__)
#define CPP_GCC

#if defined(__APPLE__)
#define CPP_APPLE
#endif 
#endif 

#include <stdint.h>
#include <memory>
#include <stdexcept>
#include <cfloat>
#include <map>
#include <string>
#include <cstring> 
#if defined WINDOWS
#include <winsock2.h>
#include <windows.h>
#endif 

#if defined CPP_GCC
#include <dirent.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <fcntl.h>
#include <cerrno>
#endif 

using std::string;

#ifdef CPP_MSVC
#define stringcat strcat_s
#define stringcpy strcpy_s
#define strprintf sprintf_s
#define stringtoken strtok_s
#define stringscanf sscanf_s
#else
#define stringcat strcat
#define stringcpy strcpy
#define strprintf snprintf
#define stringtoken strtok_r
#define stringscanf sscanf
#endif 

#if defined(__MINGW32_MAJOR_VERSION) || defined(__MINGW64_VERSION_MAJOR)
#define MINGW
#endif

#if defined(MINGW) || defined(_MSC_VER)
#define strcasecmp _stricmp
#endif 

#if defined(__clang__) && defined(__apple_build_version__)
#if ((__clang_major__ * 100) + __clang_minor__) >= 400
#if __has_feature(cxx_noexcept)
#define HAS_NOEXCEPT
#endif 
#if __has_feature(cxx_override_control)
#define HAS_OVERRIDE
#endif 
#endif 
#elif defined(__clang__)
#if ((__clang_major__ * 100) + __clang_minor__) >= 304
#if __has_feature(cxx_noexcept)
#define HAS_NOEXCEPT
#endif 
#if __has_feature(cxx_override_control)
#define HAS_OVERRIDE
#endif 
#if __has_feature(cxx_variadic_templates)
#define HAS_VARIADIC_TEMPLATES
#endif 
#endif 
#elif defined(CPP_ICC)
#if ((__INTEL_COMPILER >= 1400) && (__INTEL_COMPILER != 9999)) || (__ICL >= 1400)
#define HAS_NOEXCEPT
#define HAS_OVERRIDE
#define HAS_VARIADIC_TEMPLATES
#endif 
#elif defined(CPP_GCC)
#if (__GNUC__ * 100 + __GNUC_MINOR__) >= 406 && (__cplusplus >= 201103L || (defined(__GXX_EXPERIMENTAL_CXX0X__) && __GXX_EXPERIMENTAL_CXX0X__))
#define HAS_NOEXCEPT
#define HAS_OVERRIDE
#define HAS_VARIADIC_TEMPLATES
#endif 
#elif defined(_MSC_VER)
#if _MSC_VER >= 1900
#define HAS_NOEXCEPT
#endif 
#if _MSC_VER >= 1800
#define HAS_VARIADIC_TEMPLATES
#endif 
#if _MSC_VER>= 1600
#define HAS_OVERRIDE
#endif 
#endif 


#ifdef HAS_NOEXCEPT
#define NOEXCEPT noexcept
#else
#define NOEXCEPT throw()
#endif 


#ifdef HAS_OVERRIDE
#define OVERRIDE override
#else
#define OVERRIDE
#endif 


#ifdef MSVC
#define DLL_STL_LIST(STL_API, STL_TYPE) \
template class STL_API std::allocator< STL_TYPE >; \
template class STL_API std::vector<STL_TYPE, std::allocator< STL_TYPE > >;
#endif 

#ifdef USE_GDAL

#if defined(_MSC_VER) && (_MSC_VER >= 1400)
#pragma warning(disable: 4100 4190 4251 4275 4305 4309 4819 4996)
#endif 
#endif 


namespace ccgl {
#if defined CPP_MSVC
typedef signed __int8 vint8_t;
typedef unsigned __int8 vuint8_t;
typedef signed __int16 vint16_t;
typedef unsigned __int16 vuint16_t;
typedef signed __int32 vint32_t;
typedef unsigned __int32 vuint32_t;
typedef signed __int64 vint64_t;
typedef unsigned __int64 vuint64_t;
#else
typedef          int8_t            vint8_t;
typedef          uint8_t           vuint8_t;
typedef          int16_t           vint16_t;
typedef          uint16_t          vuint16_t;
typedef          int32_t           vint32_t;
typedef          uint32_t          vuint32_t;
typedef          int64_t           vint64_t;
typedef          uint64_t          vuint64_t;
#endif

#ifdef _WIN32

#define LLD "%I64d"
#define LLU "%I64u"
#else
#define LLD "%lld"
#define LLU "%llu"
#endif

#ifdef CPP_64
typedef vint64_t vint;
typedef vint64_t vsint;
typedef vuint64_t vuint;
#else
typedef vint32_t vint;
typedef vint32_t vsint;
typedef vuint32_t vuint;
#endif
typedef vint64_t pos_t;



#ifndef NODATA_VALUE
#define NODATA_VALUE    (-9999.)
#endif 


#ifndef MISSINGFLOAT
#define MISSINGFLOAT    (-1 * FLT_MAX)
#endif 


#ifndef MAXIMUMFLOAT
#define MAXIMUMFLOAT    FLT_MAX
#endif 


#ifndef PATH_MAX
#define PATH_MAX        1024
#endif 


#ifndef UTIL_ZERO
#define UTIL_ZERO       1.0e-6
#endif 


#ifndef PI
#define PI              3.14159265358979323846
#endif 


#ifndef MINI_SLOPE
#define MINI_SLOPE      0.0001
#endif 

#ifdef MSVC
#if _MSC_VER <= 1600
#define isnan(x) ((x) != (x))
#define isinf(x) (!_finite(x) && !_isnan(x))
#endif
#endif

#ifdef WINDOWS
#define SEP             '\\'
#define SEPSTR          "\\"
#ifndef MSVC
#define LIBPREFIX       "lib"
#endif
#define LIBSUFFIX       ".dll"
#else
#define SEP             '/'
#define SEPSTR          "/"
#define LIBPREFIX       "lib"
#endif 
#ifdef LINUX
#define LIBSUFFIX       ".so"
#elif defined(MACOS) || defined(MACOSX)
#define LIBSUFFIX       ".dylib"
#endif 


#ifdef _DEBUG
#define POSTFIX         "d"
#endif

#ifdef RELWITHDEBINFO
#define POSTFIX         "rd"
#endif

#ifdef MINSIZEREL
#define POSTFIX         "s"
#endif

#ifndef POSTFIX
#define POSTFIX         ""
#endif



#define CVT_INT(param)   static_cast<int>((param))

#define CVT_SIZET(param) static_cast<size_t>((param))

#define CVT_FLT(param)   static_cast<float>((param))

#define CVT_DBL(param)   static_cast<double>((param))

#define CVT_TIMET(param) static_cast<time_t>((param))

#define CVT_CHAR(param)  static_cast<char>((param))

#define CVT_STR(param)   static_cast<string>((param))


#define CVT_VINT(param)  static_cast<vint>((param))

#define CVT_VSINT(param) static_cast<vsint>((param))

#define CVT_VUINT(param) static_cast<vuint>((param))

#define CVT_VUINT64(param) static_cast<vuint64_t>((param))


typedef std::map<string, string> STRING_MAP;


typedef std::map<string, double> STRDBL_MAP;

#ifdef CPP_64
#define ITOA_S		_i64toa_s
#define ITOW_S		_i64tow_s
#define I64TOA_S	_i64toa_s
#define I64TOW_S	_i64tow_s
#define UITOA_S		_ui64toa_s
#define UITOW_S		_ui64tow_s
#define UI64TOA_S	_ui64toa_s
#define UI64TOW_S	_ui64tow_s
#else
#define ITOA_S		_itoa_s
#define ITOW_S		_itow_s
#define I64TOA_S	_i64toa_s
#define I64TOW_S	_i64tow_s
#define UITOA_S		_ui64toa_s
#define UITOW_S		_ui64tow_s
#define UI64TOA_S	_ui64toa_s
#define UI64TOW_S	_ui64tow_s
#endif


class NotCopyable {
private:
NotCopyable(const NotCopyable&);

NotCopyable& operator=(const NotCopyable&);
public:
NotCopyable();
};


class Object {
public:
virtual ~Object();
};


class Interface: NotCopyable {
public:
virtual ~Interface();
};


class ModelException: public std::exception {
public:

ModelException(const string& class_name, const string& function_name, const string& msg);


string ToString();


const char* what() const NOEXCEPT OVERRIDE;

private:
std::runtime_error runtime_error_;
};


bool IsIpAddress(const char* ip);


void Log(const string& msg, const string& logpath = "debugInfo.log");


int GetAvailableThreadNum();


void SetDefaultOpenMPThread();


void SetOpenMPThread(int n);


void StatusMessage(const char* msg);


void StatusMessage(const string& msg);


inline void SleepMs(const int millisecs) {
#ifdef WINDOWS
Sleep(millisecs);
#else
usleep(millisecs * 1000);   
#endif
}

} 
#endif 
