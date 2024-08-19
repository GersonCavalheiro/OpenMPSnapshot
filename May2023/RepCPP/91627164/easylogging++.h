#ifndef EASYLOGGINGPP_H
#define EASYLOGGINGPP_H
#if __cplusplus >= 201103L
#  define ELPP_CXX11 1
#endif  
#if (defined(__GNUC__))
#  define ELPP_COMPILER_GCC 1
#else
#  define ELPP_COMPILER_GCC 0
#endif
#if ELPP_COMPILER_GCC
#    define ELPP_GCC_VERSION (__GNUC__ * 10000 \
+ __GNUC_MINOR__ * 100 \
+ __GNUC_PATCHLEVEL__)
#  if defined(__GXX_EXPERIMENTAL_CXX0X__)
#    define ELPP_CXX0X 1
#  endif
#endif
#if defined(_MSC_VER)
#  define ELPP_COMPILER_MSVC 1
#else
#  define ELPP_COMPILER_MSVC 0
#endif
#define ELPP_CRT_DBG_WARNINGS ELPP_COMPILER_MSVC
#if ELPP_COMPILER_MSVC
#  if (_MSC_VER == 1600)
#    define ELPP_CXX0X 1
#  elif(_MSC_VER >= 1700)
#    define ELPP_CXX11 1
#  endif
#endif
#if (defined(__clang__) && (__clang__ == 1))
#  define ELPP_COMPILER_CLANG 1
#else
#  define ELPP_COMPILER_CLANG 0
#endif
#if ELPP_COMPILER_CLANG
#  if __has_include(<thread>)
#    include <cstddef> 
#    if !defined(__GLIBCXX__) || __GLIBCXX__ >= 20150426
#      define ELPP_CLANG_SUPPORTS_THREAD
#    endif 
#  endif 
#endif
#if (defined(__MINGW32__) || defined(__MINGW64__))
#  define ELPP_MINGW 1
#else
#  define ELPP_MINGW 0
#endif
#if (defined(__CYGWIN__) && (__CYGWIN__ == 1))
#  define ELPP_CYGWIN 1
#else
#  define ELPP_CYGWIN 0
#endif
#if (defined(__INTEL_COMPILER))
#  define ELPP_COMPILER_INTEL 1
#else
#  define ELPP_COMPILER_INTEL 0
#endif
#if (defined(_WIN32) || defined(_WIN64))
#  define ELPP_OS_WINDOWS 1
#else
#  define ELPP_OS_WINDOWS 0
#endif
#if (defined(__linux) || defined(__linux__))
#  define ELPP_OS_LINUX 1
#else
#  define ELPP_OS_LINUX 0
#endif
#if (defined(__APPLE__))
#  define ELPP_OS_MAC 1
#else
#  define ELPP_OS_MAC 0
#endif
#if (defined(__FreeBSD__))
#  define ELPP_OS_FREEBSD 1
#else
#  define ELPP_OS_FREEBSD 0
#endif
#if (defined(__sun))
#  define ELPP_OS_SOLARIS 1
#else
#  define ELPP_OS_SOLARIS 0
#endif
#if ((ELPP_OS_LINUX || ELPP_OS_MAC || ELPP_OS_FREEBSD || ELPP_OS_SOLARIS) && (!ELPP_OS_WINDOWS))
#  define ELPP_OS_UNIX 1
#else
#  define ELPP_OS_UNIX 0
#endif
#if (defined(__ANDROID__))
#  define ELPP_OS_ANDROID 1
#else
#  define ELPP_OS_ANDROID 0
#endif
#if !ELPP_OS_UNIX && !ELPP_OS_WINDOWS && ELPP_CYGWIN
#  undef ELPP_OS_UNIX
#  undef ELPP_OS_LINUX
#  define ELPP_OS_UNIX 1
#  define ELPP_OS_LINUX 1
#endif 
#if !defined(ELPP_INTERNAL_DEBUGGING_OUT_INFO)
#  define ELPP_INTERNAL_DEBUGGING_OUT_INFO std::cout
#endif 
#if !defined(ELPP_INTERNAL_DEBUGGING_OUT_ERROR)
#  define ELPP_INTERNAL_DEBUGGING_OUT_ERROR std::cerr
#endif 
#if !defined(ELPP_INTERNAL_DEBUGGING_ENDL)
#  define ELPP_INTERNAL_DEBUGGING_ENDL std::endl
#endif 
#if !defined(ELPP_INTERNAL_DEBUGGING_MSG)
#  define ELPP_INTERNAL_DEBUGGING_MSG(msg) msg
#endif 
#if !defined(ELPP_DISABLE_ASSERT)
#  if (defined(ELPP_DEBUG_ASSERT_FAILURE))
#    define ELPP_ASSERT(expr, msg) if (!(expr)) { \
std::stringstream internalInfoStream; internalInfoStream << msg; \
ELPP_INTERNAL_DEBUGGING_OUT_ERROR \
<< "EASYLOGGING++ ASSERTION FAILED (LINE: " << __LINE__ << ") [" #expr << "] WITH MESSAGE \"" \
<< ELPP_INTERNAL_DEBUGGING_MSG(internalInfoStream.str()) << "\"" << ELPP_INTERNAL_DEBUGGING_ENDL; base::utils::abort(1, \
"ELPP Assertion failure, please define ELPP_DEBUG_ASSERT_FAILURE"); }
#  else
#    define ELPP_ASSERT(expr, msg) if (!(expr)) { \
std::stringstream internalInfoStream; internalInfoStream << msg; \
ELPP_INTERNAL_DEBUGGING_OUT_ERROR\
<< "ASSERTION FAILURE FROM EASYLOGGING++ (LINE: " \
<< __LINE__ << ") [" #expr << "] WITH MESSAGE \"" << ELPP_INTERNAL_DEBUGGING_MSG(internalInfoStream.str()) << "\"" \
<< ELPP_INTERNAL_DEBUGGING_ENDL; }
#  endif  
#else
#  define ELPP_ASSERT(x, y)
#endif  
#if ELPP_COMPILER_MSVC
#  define ELPP_INTERNAL_DEBUGGING_WRITE_PERROR \
{ char buff[256]; strerror_s(buff, 256, errno); \
ELPP_INTERNAL_DEBUGGING_OUT_ERROR << ": " << buff << " [" << errno << "]";} (void)0
#else
#  define ELPP_INTERNAL_DEBUGGING_WRITE_PERROR \
ELPP_INTERNAL_DEBUGGING_OUT_ERROR << ": " << strerror(errno) << " [" << errno << "]"; (void)0
#endif  
#if defined(ELPP_DEBUG_ERRORS)
#  if !defined(ELPP_INTERNAL_ERROR)
#    define ELPP_INTERNAL_ERROR(msg, pe) { \
std::stringstream internalInfoStream; internalInfoStream << "<ERROR> " << msg; \
ELPP_INTERNAL_DEBUGGING_OUT_ERROR \
<< "ERROR FROM EASYLOGGING++ (LINE: " << __LINE__ << ") " \
<< ELPP_INTERNAL_DEBUGGING_MSG(internalInfoStream.str()) << ELPP_INTERNAL_DEBUGGING_ENDL; \
if (pe) { ELPP_INTERNAL_DEBUGGING_OUT_ERROR << "    "; ELPP_INTERNAL_DEBUGGING_WRITE_PERROR; }} (void)0
#  endif
#else
#  undef ELPP_INTERNAL_INFO
#  define ELPP_INTERNAL_ERROR(msg, pe)
#endif  
#if (defined(ELPP_DEBUG_INFO))
#  if !(defined(ELPP_INTERNAL_INFO_LEVEL))
#    define ELPP_INTERNAL_INFO_LEVEL 9
#  endif  
#  if !defined(ELPP_INTERNAL_INFO)
#    define ELPP_INTERNAL_INFO(lvl, msg) { if (lvl <= ELPP_INTERNAL_INFO_LEVEL) { \
std::stringstream internalInfoStream; internalInfoStream << "<INFO> " << msg; \
ELPP_INTERNAL_DEBUGGING_OUT_INFO << ELPP_INTERNAL_DEBUGGING_MSG(internalInfoStream.str()) \
<< ELPP_INTERNAL_DEBUGGING_ENDL; }}
#  endif
#else
#  undef ELPP_INTERNAL_INFO
#  define ELPP_INTERNAL_INFO(lvl, msg)
#endif  
#if (defined(ELPP_FEATURE_ALL)) || (defined(ELPP_FEATURE_CRASH_LOG))
#  if (ELPP_COMPILER_GCC && !ELPP_MINGW && !ELPP_OS_ANDROID)
#    define ELPP_STACKTRACE 1
#  else
#      if ELPP_COMPILER_MSVC
#         pragma message("Stack trace not available for this compiler")
#      else
#         warning "Stack trace not available for this compiler";
#      endif  
#    define ELPP_STACKTRACE 0
#  endif  
#else
#    define ELPP_STACKTRACE 0
#endif  
#define ELPP_UNUSED(x) (void)x
#if ELPP_OS_UNIX
#  define ELPP_LOG_PERMS S_IRUSR | S_IWUSR | S_IXUSR | S_IWGRP | S_IRGRP | S_IXGRP | S_IWOTH | S_IXOTH
#endif  
#if defined(ELPP_AS_DLL) && ELPP_COMPILER_MSVC
#  if defined(ELPP_EXPORT_SYMBOLS)
#    define ELPP_EXPORT __declspec(dllexport)
#  else
#    define ELPP_EXPORT __declspec(dllimport)
#  endif  
#else
#  define ELPP_EXPORT
#endif  
#undef STRTOK
#undef STRERROR
#undef STRCAT
#undef STRCPY
#if ELPP_CRT_DBG_WARNINGS
#  define STRTOK(a, b, c) strtok_s(a, b, c)
#  define STRERROR(a, b, c) strerror_s(a, b, c)
#  define STRCAT(a, b, len) strcat_s(a, len, b)
#  define STRCPY(a, b, len) strcpy_s(a, len, b)
#else
#  define STRTOK(a, b, c) strtok(a, b)
#  define STRERROR(a, b, c) strerror(c)
#  define STRCAT(a, b, len) strcat(a, b)
#  define STRCPY(a, b, len) strcpy(a, b)
#endif
#if (ELPP_MINGW && !defined(ELPP_FORCE_USE_STD_THREAD))
#  define ELPP_USE_STD_THREADING 0
#else
#  if ((ELPP_COMPILER_CLANG && defined(ELPP_CLANG_SUPPORTS_THREAD)) || \
(!ELPP_COMPILER_CLANG && defined(ELPP_CXX11)) || \
defined(ELPP_FORCE_USE_STD_THREAD))
#    define ELPP_USE_STD_THREADING 1
#  else
#    define ELPP_USE_STD_THREADING 0
#  endif
#endif
#undef ELPP_FINAL
#if ELPP_COMPILER_INTEL || (ELPP_GCC_VERSION < 40702)
#  define ELPP_FINAL
#else
#  define ELPP_FINAL final
#endif  
#if defined(ELPP_EXPERIMENTAL_ASYNC)
#  define ELPP_ASYNC_LOGGING 1
#else
#  define ELPP_ASYNC_LOGGING 0
#endif 
#if defined(ELPP_THREAD_SAFE) || ELPP_ASYNC_LOGGING
#  define ELPP_THREADING_ENABLED 1
#else
#  define ELPP_THREADING_ENABLED 0
#endif  
#undef ELPP_FUNC
#if ELPP_COMPILER_MSVC  
#  define ELPP_FUNC __FUNCSIG__
#elif ELPP_COMPILER_GCC  
#  define ELPP_FUNC __PRETTY_FUNCTION__
#elif ELPP_COMPILER_INTEL  
#  define ELPP_FUNC __PRETTY_FUNCTION__
#elif ELPP_COMPILER_CLANG  
#  define ELPP_FUNC __PRETTY_FUNCTION__
#else
#  if defined(__func__)
#    define ELPP_FUNC __func__
#  else
#    define ELPP_FUNC ""
#  endif  
#endif  
#undef ELPP_VARIADIC_TEMPLATES_SUPPORTED
#define ELPP_VARIADIC_TEMPLATES_SUPPORTED \
(ELPP_COMPILER_GCC || ELPP_COMPILER_CLANG || ELPP_COMPILER_INTEL || (ELPP_COMPILER_MSVC && _MSC_VER >= 1800))
#if defined(ELPP_DISABLE_LOGS)
#define ELPP_LOGGING_ENABLED 0
#else
#define ELPP_LOGGING_ENABLED 1
#endif
#if (!defined(ELPP_DISABLE_DEBUG_LOGS) && (ELPP_LOGGING_ENABLED) && ((defined(_DEBUG)) || (!defined(NDEBUG))))
#  define ELPP_DEBUG_LOG 1
#else
#  define ELPP_DEBUG_LOG 0
#endif  
#if (!defined(ELPP_DISABLE_INFO_LOGS) && (ELPP_LOGGING_ENABLED))
#  define ELPP_INFO_LOG 1
#else
#  define ELPP_INFO_LOG 0
#endif  
#if (!defined(ELPP_DISABLE_WARNING_LOGS) && (ELPP_LOGGING_ENABLED))
#  define ELPP_WARNING_LOG 1
#else
#  define ELPP_WARNING_LOG 0
#endif  
#if (!defined(ELPP_DISABLE_ERROR_LOGS) && (ELPP_LOGGING_ENABLED))
#  define ELPP_ERROR_LOG 1
#else
#  define ELPP_ERROR_LOG 0
#endif  
#if (!defined(ELPP_DISABLE_FATAL_LOGS) && (ELPP_LOGGING_ENABLED))
#  define ELPP_FATAL_LOG 1
#else
#  define ELPP_FATAL_LOG 0
#endif  
#if (!defined(ELPP_DISABLE_TRACE_LOGS) && (ELPP_LOGGING_ENABLED))
#  define ELPP_TRACE_LOG 1
#else
#  define ELPP_TRACE_LOG 0
#endif  
#if (!defined(ELPP_DISABLE_VERBOSE_LOGS) && (ELPP_LOGGING_ENABLED))
#  define ELPP_VERBOSE_LOG 1
#else
#  define ELPP_VERBOSE_LOG 0
#endif  
#if (!(ELPP_CXX0X || ELPP_CXX11))
#   error "C++0x (or higher) support not detected! (Is `-std=c++11' missing?)"
#endif  
#if defined(ELPP_SYSLOG)
#   include <syslog.h>
#endif  
#include <ctime>
#include <cstring>
#include <cstdlib>
#include <cctype>
#include <cwchar>
#include <csignal>
#include <cerrno>
#include <cstdarg>
#if defined(ELPP_UNICODE)
#   include <locale>
#  if ELPP_OS_WINDOWS
#      include <codecvt>
#  endif 
#endif  
#if ELPP_STACKTRACE
#   include <cxxabi.h>
#   include <execinfo.h>
#endif  
#if ELPP_OS_ANDROID
#   include <sys/system_properties.h>
#endif  
#if ELPP_OS_UNIX
#   include <sys/stat.h>
#   include <sys/time.h>
#elif ELPP_OS_WINDOWS
#   include <direct.h>
#   include <windows.h>
#  if defined(WIN32_LEAN_AND_MEAN)
#      if defined(ELPP_WINSOCK2)
#         include <winsock2.h>
#      else
#         include <winsock.h>
#      endif 
#  endif 
#endif  
#include <string>
#include <vector>
#include <map>
#include <utility>
#include <functional>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <memory>
#include <type_traits>
#if ELPP_THREADING_ENABLED
#  if ELPP_USE_STD_THREADING
#      include <mutex>
#      include <thread>
#  else
#      if ELPP_OS_UNIX
#         include <pthread.h>
#      endif  
#  endif  
#endif  
#if ELPP_ASYNC_LOGGING
#  if defined(ELPP_NO_SLEEP_FOR)
#      include <unistd.h>
#  endif  
#   include <thread>
#   include <queue>
#   include <condition_variable>
#endif  
#if defined(ELPP_STL_LOGGING)
#   include <list>
#   include <queue>
#   include <deque>
#   include <set>
#   include <bitset>
#   include <stack>
#  if defined(ELPP_LOG_STD_ARRAY)
#      include <array>
#  endif  
#  if defined(ELPP_LOG_UNORDERED_MAP)
#      include <unordered_map>
#  endif  
#  if defined(ELPP_LOG_UNORDERED_SET)
#      include <unordered_set>
#  endif  
#endif  
#if defined(ELPP_QT_LOGGING)
#   include <QString>
#   include <QByteArray>
#   include <QVector>
#   include <QList>
#   include <QPair>
#   include <QMap>
#   include <QQueue>
#   include <QSet>
#   include <QLinkedList>
#   include <QHash>
#   include <QMultiHash>
#   include <QStack>
#endif  
#if defined(ELPP_BOOST_LOGGING)
#   include <boost/container/vector.hpp>
#   include <boost/container/stable_vector.hpp>
#   include <boost/container/list.hpp>
#   include <boost/container/deque.hpp>
#   include <boost/container/map.hpp>
#   include <boost/container/flat_map.hpp>
#   include <boost/container/set.hpp>
#   include <boost/container/flat_set.hpp>
#endif  
#if defined(ELPP_WXWIDGETS_LOGGING)
#   include <wx/vector.h>
#endif  
#if defined(ELPP_UTC_DATETIME)
#   define elpptime_r gmtime_r
#   define elpptime_s gmtime_s
#   define elpptime   gmtime
#else
#   define elpptime_r localtime_r
#   define elpptime_s localtime_s
#   define elpptime   localtime
#endif  
namespace el {
class Logger;
class LogMessage;
class PerformanceTrackingData;
class Loggers;
class Helpers;
template <typename T> class Callback;
class LogDispatchCallback;
class PerformanceTrackingCallback;
class LoggerRegistrationCallback;
class LogDispatchData;
namespace base {
class Storage;
class RegisteredLoggers;
class PerformanceTracker;
class MessageBuilder;
class Writer;
class PErrorWriter;
class LogDispatcher;
class DefaultLogBuilder;
class DefaultLogDispatchCallback;
#if ELPP_ASYNC_LOGGING
class AsyncLogDispatchCallback;
class AsyncDispatchWorker;
#endif 
class DefaultPerformanceTrackingCallback;
}  
}  
namespace el {
namespace base {
namespace type {
#undef ELPP_LITERAL
#undef ELPP_STRLEN
#undef ELPP_COUT
#if defined(ELPP_UNICODE)
#  define ELPP_LITERAL(txt) L##txt
#  define ELPP_STRLEN wcslen
#  if defined ELPP_CUSTOM_COUT
#    define ELPP_COUT ELPP_CUSTOM_COUT
#  else
#    define ELPP_COUT std::wcout
#  endif  
typedef wchar_t char_t;
typedef std::wstring string_t;
typedef std::wstringstream stringstream_t;
typedef std::wfstream fstream_t;
typedef std::wostream ostream_t;
#else
#  define ELPP_LITERAL(txt) txt
#  define ELPP_STRLEN strlen
#  if defined ELPP_CUSTOM_COUT
#    define ELPP_COUT ELPP_CUSTOM_COUT
#  else
#    define ELPP_COUT std::cout
#  endif  
typedef char char_t;
typedef std::string string_t;
typedef std::stringstream stringstream_t;
typedef std::fstream fstream_t;
typedef std::ostream ostream_t;
#endif  
#if defined(ELPP_CUSTOM_COUT_LINE)
#  define ELPP_COUT_LINE(logLine) ELPP_CUSTOM_COUT_LINE(logLine)
#else
#  define ELPP_COUT_LINE(logLine) logLine << std::flush
#endif 
typedef unsigned int EnumType;
typedef unsigned short VerboseLevel;
typedef unsigned long int LineNumber;
typedef std::shared_ptr<base::Storage> StoragePointer;
typedef std::shared_ptr<LogDispatchCallback> LogDispatchCallbackPtr;
typedef std::shared_ptr<PerformanceTrackingCallback> PerformanceTrackingCallbackPtr;
typedef std::shared_ptr<LoggerRegistrationCallback> LoggerRegistrationCallbackPtr;
typedef std::unique_ptr<el::base::PerformanceTracker> PerformanceTrackerPtr;
}  
class NoCopy {
protected:
NoCopy(void) {}
private:
NoCopy(const NoCopy&);
NoCopy& operator=(const NoCopy&);
};
class StaticClass {
private:
StaticClass(void);
StaticClass(const StaticClass&);
StaticClass& operator=(const StaticClass&);
};
}  
enum class Level : base::type::EnumType {
Global = 1,
Trace = 2,
Debug = 4,
Fatal = 8,
Error = 16,
Warning = 32,
Verbose = 64,
Info = 128,
Unknown = 1010
};
class LevelHelper : base::StaticClass {
public:
static const base::type::EnumType kMinValid = static_cast<base::type::EnumType>(Level::Trace);
static const base::type::EnumType kMaxValid = static_cast<base::type::EnumType>(Level::Info);
static base::type::EnumType castToInt(Level level) {
return static_cast<base::type::EnumType>(level);
}
static Level castFromInt(base::type::EnumType l) {
return static_cast<Level>(l);
}
static const char* convertToString(Level level);
static Level convertFromString(const char* levelStr);
static void forEachLevel(base::type::EnumType* startIndex, const std::function<bool(void)>& fn);
};
enum class ConfigurationType : base::type::EnumType {
Enabled = 1,
ToFile = 2,
ToStandardOutput = 4,
Format = 8,
Filename = 16,
SubsecondPrecision = 32,
MillisecondsWidth = SubsecondPrecision,
PerformanceTracking = 64,
MaxLogFileSize = 128,
LogFlushThreshold = 256,
Unknown = 1010
};
class ConfigurationTypeHelper : base::StaticClass {
public:
static const base::type::EnumType kMinValid = static_cast<base::type::EnumType>(ConfigurationType::Enabled);
static const base::type::EnumType kMaxValid = static_cast<base::type::EnumType>(ConfigurationType::MaxLogFileSize);
static base::type::EnumType castToInt(ConfigurationType configurationType) {
return static_cast<base::type::EnumType>(configurationType);
}
static ConfigurationType castFromInt(base::type::EnumType c) {
return static_cast<ConfigurationType>(c);
}
static const char* convertToString(ConfigurationType configurationType);
static ConfigurationType convertFromString(const char* configStr);
static inline void forEachConfigType(base::type::EnumType* startIndex, const std::function<bool(void)>& fn);
};
enum class LoggingFlag : base::type::EnumType {
NewLineForContainer = 1,
AllowVerboseIfModuleNotSpecified = 2,
LogDetailedCrashReason = 4,
DisableApplicationAbortOnFatalLog = 8,
ImmediateFlush = 16,
StrictLogFileSizeCheck = 32,
ColoredTerminalOutput = 64,
MultiLoggerSupport = 128,
DisablePerformanceTrackingCheckpointComparison = 256,
DisableVModules = 512,
DisableVModulesExtensions = 1024,
HierarchicalLogging = 2048,
CreateLoggerAutomatically = 4096,
AutoSpacing = 8192,
FixedTimeFormat = 16384
};
namespace base {
namespace consts {
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
#endif
static const base::type::char_t* kInfoLevelLogValue     =   ELPP_LITERAL("INFO");
static const base::type::char_t* kDebugLevelLogValue    =   ELPP_LITERAL("DEBUG");
static const base::type::char_t* kWarningLevelLogValue  =   ELPP_LITERAL("WARNING");
static const base::type::char_t* kErrorLevelLogValue    =   ELPP_LITERAL("ERROR");
static const base::type::char_t* kFatalLevelLogValue    =   ELPP_LITERAL("FATAL");
static const base::type::char_t* kVerboseLevelLogValue  =
ELPP_LITERAL("VERBOSE"); 
static const base::type::char_t* kTraceLevelLogValue    =   ELPP_LITERAL("TRACE");
static const base::type::char_t* kInfoLevelShortLogValue     =   ELPP_LITERAL("I");
static const base::type::char_t* kDebugLevelShortLogValue    =   ELPP_LITERAL("D");
static const base::type::char_t* kWarningLevelShortLogValue  =   ELPP_LITERAL("W");
static const base::type::char_t* kErrorLevelShortLogValue    =   ELPP_LITERAL("E");
static const base::type::char_t* kFatalLevelShortLogValue    =   ELPP_LITERAL("F");
static const base::type::char_t* kVerboseLevelShortLogValue  =   ELPP_LITERAL("V");
static const base::type::char_t* kTraceLevelShortLogValue    =   ELPP_LITERAL("T");
static const base::type::char_t* kAppNameFormatSpecifier          =      ELPP_LITERAL("%app");
static const base::type::char_t* kLoggerIdFormatSpecifier         =      ELPP_LITERAL("%logger");
static const base::type::char_t* kThreadIdFormatSpecifier         =      ELPP_LITERAL("%thread");
static const base::type::char_t* kSeverityLevelFormatSpecifier    =      ELPP_LITERAL("%level");
static const base::type::char_t* kSeverityLevelShortFormatSpecifier    =      ELPP_LITERAL("%levshort");
static const base::type::char_t* kDateTimeFormatSpecifier         =      ELPP_LITERAL("%datetime");
static const base::type::char_t* kLogFileFormatSpecifier          =      ELPP_LITERAL("%file");
static const base::type::char_t* kLogFileBaseFormatSpecifier      =      ELPP_LITERAL("%fbase");
static const base::type::char_t* kLogLineFormatSpecifier          =      ELPP_LITERAL("%line");
static const base::type::char_t* kLogLocationFormatSpecifier      =      ELPP_LITERAL("%loc");
static const base::type::char_t* kLogFunctionFormatSpecifier      =      ELPP_LITERAL("%func");
static const base::type::char_t* kCurrentUserFormatSpecifier      =      ELPP_LITERAL("%user");
static const base::type::char_t* kCurrentHostFormatSpecifier      =      ELPP_LITERAL("%host");
static const base::type::char_t* kMessageFormatSpecifier          =      ELPP_LITERAL("%msg");
static const base::type::char_t* kVerboseLevelFormatSpecifier     =      ELPP_LITERAL("%vlevel");
static const char* kDateTimeFormatSpecifierForFilename            =      "%datetime";
static const char* kDays[7]                         =      { "Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday" };
static const char* kDaysAbbrev[7]                   =      { "Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat" };
static const char* kMonths[12]                      =      { "January", "February", "March", "Apri", "May", "June", "July", "August",
"September", "October", "November", "December"
};
static const char* kMonthsAbbrev[12]                =      { "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec" };
static const char* kDefaultDateTimeFormat           =      "%Y-%M-%d %H:%m:%s,%g";
static const char* kDefaultDateTimeFormatInFilename =      "%Y-%M-%d_%H-%m";
static const int kYearBase                          =      1900;
static const char* kAm                              =      "AM";
static const char* kPm                              =      "PM";
#ifdef ELPP_DEFAULT_LOGGER
static const char* kDefaultLoggerId                        =      ELPP_DEFAULT_LOGGER;
#else
static const char* kDefaultLoggerId                        =      "default";
#endif
#ifdef ELPP_DEFAULT_PERFORMANCE_LOGGER
static const char* kPerformanceLoggerId                    =      ELPP_DEFAULT_PERFORMANCE_LOGGER;
#else
static const char* kPerformanceLoggerId                    =      "performance";
#endif
#if defined(ELPP_SYSLOG)
static const char* kSysLogLoggerId                         =      "syslog";
#endif  
static const char* kNullPointer                            =      "nullptr";
static const char  kFormatSpecifierChar                    =      '%';
#if ELPP_VARIADIC_TEMPLATES_SUPPORTED
static const char  kFormatSpecifierCharValue               =      'v';
#endif  
static const unsigned int kMaxLogPerContainer              =      100;
static const unsigned int kMaxLogPerCounter                =      100000;
static const unsigned int kDefaultSubsecondPrecision       =      3;
static const base::type::VerboseLevel kMaxVerboseLevel     =      9;
static const char* kUnknownUser                            =      "user";
static const char* kUnknownHost                            =      "unknown-host";
#if defined(ELPP_DEFAULT_LOG_FILE)
static const char* kDefaultLogFile                         =      ELPP_DEFAULT_LOG_FILE;
#else
#  if ELPP_OS_UNIX
#      if ELPP_OS_ANDROID
static const char* kDefaultLogFile                         =      "logs/myeasylog.log";
#      else
static const char* kDefaultLogFile                         =      "logs/myeasylog.log";
#      endif  
#  elif ELPP_OS_WINDOWS
static const char* kDefaultLogFile                         =      "logs\\myeasylog.log";
#  endif  
#endif  
#if !defined(ELPP_DISABLE_LOG_FILE_FROM_ARG)
static const char* kDefaultLogFileParam                    =      "--default-log-file";
#endif  
#if defined(ELPP_LOGGING_FLAGS_FROM_ARG)
static const char* kLoggingFlagsParam                      =      "--logging-flags";
#endif  
#if ELPP_OS_WINDOWS
static const char* kFilePathSeperator                      =      "\\";
#else
static const char* kFilePathSeperator                      =      "/";
#endif  
static const char* kValidLoggerIdSymbols                   =
"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-._";
static const char* kConfigurationComment                   =      "##";
static const char* kConfigurationLevel                     =      "*";
static const char* kConfigurationLoggerId                  =      "--";
static const std::size_t kSourceFilenameMaxLength          =      100;
static const std::size_t kSourceLineMaxLength              =      10;
static const Level kPerformanceTrackerDefaultLevel         =      Level::Info;
const struct {
double value;
const base::type::char_t* unit;
} kTimeFormats[] = {
{ 1000.0f, ELPP_LITERAL("us") },
{ 1000.0f, ELPP_LITERAL("ms") },
{ 60.0f, ELPP_LITERAL("seconds") },
{ 60.0f, ELPP_LITERAL("minutes") },
{ 24.0f, ELPP_LITERAL("hours") },
{ 7.0f, ELPP_LITERAL("days") }
};
static const int kTimeFormatsCount                           =      sizeof(kTimeFormats) / sizeof(kTimeFormats[0]);
const struct {
int numb;
const char* name;
const char* brief;
const char* detail;
} kCrashSignals[] = {
{
SIGABRT, "SIGABRT", "Abnormal termination",
"Program was abnormally terminated."
},
{
SIGFPE, "SIGFPE", "Erroneous arithmetic operation",
"Arithemetic operation issue such as division by zero or operation resulting in overflow."
},
{
SIGILL, "SIGILL", "Illegal instruction",
"Generally due to a corruption in the code or to an attempt to execute data."
},
{
SIGSEGV, "SIGSEGV", "Invalid access to memory",
"Program is trying to read an invalid (unallocated, deleted or corrupted) or inaccessible memory."
},
{
SIGINT, "SIGINT", "Interactive attention signal",
"Interruption generated (generally) by user or operating system."
},
};
static const int kCrashSignalsCount                          =      sizeof(kCrashSignals) / sizeof(kCrashSignals[0]);
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif
}  
}  
typedef std::function<void(const char*, std::size_t)> PreRollOutCallback;
namespace base {
static inline void defaultPreRollOutCallback(const char*, std::size_t) {}
enum class TimestampUnit : base::type::EnumType {
Microsecond = 0, Millisecond = 1, Second = 2, Minute = 3, Hour = 4, Day = 5
};
enum class FormatFlags : base::type::EnumType {
DateTime = 1 << 1,
LoggerId = 1 << 2,
File = 1 << 3,
Line = 1 << 4,
Location = 1 << 5,
Function = 1 << 6,
User = 1 << 7,
Host = 1 << 8,
LogMessage = 1 << 9,
VerboseLevel = 1 << 10,
AppName = 1 << 11,
ThreadId = 1 << 12,
Level = 1 << 13,
FileBase = 1 << 14,
LevelShort = 1 << 15
};
class SubsecondPrecision {
public:
SubsecondPrecision(void) {
init(base::consts::kDefaultSubsecondPrecision);
}
explicit SubsecondPrecision(int width) {
init(width);
}
bool operator==(const SubsecondPrecision& ssPrec) {
return m_width == ssPrec.m_width && m_offset == ssPrec.m_offset;
}
int m_width;
unsigned int m_offset;
private:
void init(int width);
};
typedef SubsecondPrecision MillisecondsWidth;
namespace utils {
template <typename T>
static
typename std::enable_if<std::is_pointer<T*>::value, void>::type
safeDelete(T*& pointer) {
if (pointer == nullptr)
return;
delete pointer;
pointer = nullptr;
}
namespace bitwise {
template <typename Enum>
static inline base::type::EnumType And(Enum e, base::type::EnumType flag) {
return static_cast<base::type::EnumType>(flag) & static_cast<base::type::EnumType>(e);
}
template <typename Enum>
static inline base::type::EnumType Not(Enum e, base::type::EnumType flag) {
return static_cast<base::type::EnumType>(flag) & ~(static_cast<base::type::EnumType>(e));
}
template <typename Enum>
static inline base::type::EnumType Or(Enum e, base::type::EnumType flag) {
return static_cast<base::type::EnumType>(flag) | static_cast<base::type::EnumType>(e);
}
}  
template <typename Enum>
static inline void addFlag(Enum e, base::type::EnumType* flag) {
*flag = base::utils::bitwise::Or<Enum>(e, *flag);
}
template <typename Enum>
static inline void removeFlag(Enum e, base::type::EnumType* flag) {
*flag = base::utils::bitwise::Not<Enum>(e, *flag);
}
template <typename Enum>
static inline bool hasFlag(Enum e, base::type::EnumType flag) {
return base::utils::bitwise::And<Enum>(e, flag) > 0x0;
}
}  
namespace threading {
#if ELPP_THREADING_ENABLED
#  if !ELPP_USE_STD_THREADING
namespace internal {
class Mutex : base::NoCopy {
public:
Mutex(void) {
#  if ELPP_OS_UNIX
pthread_mutexattr_t attr;
pthread_mutexattr_init(&attr);
pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_RECURSIVE);
pthread_mutex_init(&m_underlyingMutex, &attr);
pthread_mutexattr_destroy(&attr);
#  elif ELPP_OS_WINDOWS
InitializeCriticalSection(&m_underlyingMutex);
#  endif  
}

virtual ~Mutex(void) {
#  if ELPP_OS_UNIX
pthread_mutex_destroy(&m_underlyingMutex);
#  elif ELPP_OS_WINDOWS
DeleteCriticalSection(&m_underlyingMutex);
#  endif  
}

inline void lock(void) {
#  if ELPP_OS_UNIX
pthread_mutex_lock(&m_underlyingMutex);
#  elif ELPP_OS_WINDOWS
EnterCriticalSection(&m_underlyingMutex);
#  endif  
}

inline bool try_lock(void) {
#  if ELPP_OS_UNIX
return (pthread_mutex_trylock(&m_underlyingMutex) == 0);
#  elif ELPP_OS_WINDOWS
return TryEnterCriticalSection(&m_underlyingMutex);
#  endif  
}

inline void unlock(void) {
#  if ELPP_OS_UNIX
pthread_mutex_unlock(&m_underlyingMutex);
#  elif ELPP_OS_WINDOWS
LeaveCriticalSection(&m_underlyingMutex);
#  endif  
}

private:
#  if ELPP_OS_UNIX
pthread_mutex_t m_underlyingMutex;
#  elif ELPP_OS_WINDOWS
CRITICAL_SECTION m_underlyingMutex;
#  endif  
};
template <typename M>
class ScopedLock : base::NoCopy {
public:
explicit ScopedLock(M& mutex) {
m_mutex = &mutex;
m_mutex->lock();
}

virtual ~ScopedLock(void) {
m_mutex->unlock();
}
private:
M* m_mutex;
ScopedLock(void);
};
} 
typedef base::threading::internal::Mutex Mutex;
typedef base::threading::internal::ScopedLock<base::threading::Mutex> ScopedLock;
#  else
typedef std::recursive_mutex Mutex;
typedef std::lock_guard<base::threading::Mutex> ScopedLock;
#  endif  
#else
namespace internal {
class NoMutex : base::NoCopy {
public:
NoMutex(void) {}
inline void lock(void) {}
inline bool try_lock(void) {
return true;
}
inline void unlock(void) {}
};
template <typename Mutex>
class NoScopedLock : base::NoCopy {
public:
explicit NoScopedLock(Mutex&) {
}
virtual ~NoScopedLock(void) {
}
private:
NoScopedLock(void);
};
}  
typedef base::threading::internal::NoMutex Mutex;
typedef base::threading::internal::NoScopedLock<base::threading::Mutex> ScopedLock;
#endif  
class ThreadSafe {
public:
virtual inline void acquireLock(void) ELPP_FINAL { m_mutex.lock(); }
virtual inline void releaseLock(void) ELPP_FINAL { m_mutex.unlock(); }
virtual inline base::threading::Mutex& lock(void) ELPP_FINAL { return m_mutex; }
protected:
ThreadSafe(void) {}
virtual ~ThreadSafe(void) {}
private:
base::threading::Mutex m_mutex;
};

#if ELPP_THREADING_ENABLED
#  if !ELPP_USE_STD_THREADING
static std::string getCurrentThreadId(void) {
std::stringstream ss;
#      if (ELPP_OS_WINDOWS)
ss << GetCurrentThreadId();
#      endif  
return ss.str();
}
#  else
static std::string getCurrentThreadId(void) {
std::stringstream ss;
ss << std::this_thread::get_id();
return ss.str();
}
#  endif  
#else
static inline std::string getCurrentThreadId(void) {
return std::string();
}
#endif  
}  
namespace utils {
class File : base::StaticClass {
public:
static base::type::fstream_t* newFileStream(const std::string& filename);

static std::size_t getSizeOfFile(base::type::fstream_t* fs);

static bool pathExists(const char* path, bool considerFile = false);

static bool createPath(const std::string& path);
static std::string extractPathFromFilename(const std::string& fullPath,
const char* seperator = base::consts::kFilePathSeperator);
static void buildStrippedFilename(const char* filename, char buff[],
std::size_t limit = base::consts::kSourceFilenameMaxLength);
static void buildBaseFilename(const std::string& fullPath, char buff[],
std::size_t limit = base::consts::kSourceFilenameMaxLength,
const char* seperator = base::consts::kFilePathSeperator);
};
class Str : base::StaticClass {
public:
static inline bool isDigit(char c) {
return c >= '0' && c <= '9';
}

static bool wildCardMatch(const char* str, const char* pattern);

static std::string& ltrim(std::string& str);
static std::string& rtrim(std::string& str);
static std::string& trim(std::string& str);

static bool startsWith(const std::string& str, const std::string& start);

static bool endsWith(const std::string& str, const std::string& end);

static std::string& replaceAll(std::string& str, char replaceWhat, char replaceWith);

static std::string& replaceAll(std::string& str, const std::string& replaceWhat,
const std::string& replaceWith);

static void replaceFirstWithEscape(base::type::string_t& str, const base::type::string_t& replaceWhat,
const base::type::string_t& replaceWith);
#if defined(ELPP_UNICODE)
static void replaceFirstWithEscape(base::type::string_t& str, const base::type::string_t& replaceWhat,
const std::string& replaceWith);
#endif  
static std::string& toUpper(std::string& str);

static bool cStringEq(const char* s1, const char* s2);

static bool cStringCaseEq(const char* s1, const char* s2);

static bool contains(const char* str, char c);

static char* convertAndAddToBuff(std::size_t n, int len, char* buf, const char* bufLim, bool zeroPadded = true);
static char* addToBuff(const char* str, char* buf, const char* bufLim);
static char* clearBuff(char buff[], std::size_t lim);

static char* wcharPtrToCharPtr(const wchar_t* line);
};
class OS : base::StaticClass {
public:
#if ELPP_OS_WINDOWS
static const char* getWindowsEnvironmentVariable(const char* varname);
#endif  
#if ELPP_OS_ANDROID
static std::string getProperty(const char* prop);

static std::string getDeviceName(void);
#endif  

static const std::string getBashOutput(const char* command);

static std::string getEnvironmentVariable(const char* variableName, const char* defaultVal,
const char* alternativeBashCommand = nullptr);
static std::string currentUser(void);

static std::string currentHost(void);
static bool termSupportsColor(void);
};
class DateTime : base::StaticClass {
public:
static void gettimeofday(struct timeval* tv);

static std::string getDateTime(const char* format, const base::SubsecondPrecision* ssPrec);

static std::string timevalToString(struct timeval tval, const char* format,
const el::base::SubsecondPrecision* ssPrec);

static base::type::string_t formatTime(unsigned long long time, base::TimestampUnit timestampUnit);

static unsigned long long getTimeDifference(const struct timeval& endTime, const struct timeval& startTime,
base::TimestampUnit timestampUnit);


private:
static struct ::tm* buildTimeInfo(struct timeval* currTime, struct ::tm* timeInfo);
static char* parseFormat(char* buf, std::size_t bufSz, const char* format, const struct tm* tInfo,
std::size_t msec, const base::SubsecondPrecision* ssPrec);
};
class CommandLineArgs {
public:
CommandLineArgs(void) {
setArgs(0, static_cast<char**>(nullptr));
}
CommandLineArgs(int argc, const char** argv) {
setArgs(argc, argv);
}
CommandLineArgs(int argc, char** argv) {
setArgs(argc, argv);
}
virtual ~CommandLineArgs(void) {}
inline void setArgs(int argc, const char** argv) {
setArgs(argc, const_cast<char**>(argv));
}
void setArgs(int argc, char** argv);
bool hasParamWithValue(const char* paramKey) const;
const char* getParamValue(const char* paramKey) const;
bool hasParam(const char* paramKey) const;
bool empty(void) const;
std::size_t size(void) const;
friend base::type::ostream_t& operator<<(base::type::ostream_t& os, const CommandLineArgs& c);

private:
int m_argc;
char** m_argv;
std::map<std::string, std::string> m_paramsWithValue;
std::vector<std::string> m_params;
};
template <typename T_Ptr, typename Container>
class AbstractRegistry : public base::threading::ThreadSafe {
public:
typedef typename Container::iterator iterator;
typedef typename Container::const_iterator const_iterator;

AbstractRegistry(void) {}

AbstractRegistry(AbstractRegistry&& sr) {
if (this == &sr) {
return;
}
unregisterAll();
m_list = std::move(sr.m_list);
}

bool operator==(const AbstractRegistry<T_Ptr, Container>& other) {
if (size() != other.size()) {
return false;
}
for (std::size_t i = 0; i < m_list.size(); ++i) {
if (m_list.at(i) != other.m_list.at(i)) {
return false;
}
}
return true;
}

bool operator!=(const AbstractRegistry<T_Ptr, Container>& other) {
if (size() != other.size()) {
return true;
}
for (std::size_t i = 0; i < m_list.size(); ++i) {
if (m_list.at(i) != other.m_list.at(i)) {
return true;
}
}
return false;
}

AbstractRegistry& operator=(AbstractRegistry&& sr) {
if (this == &sr) {
return *this;
}
unregisterAll();
m_list = std::move(sr.m_list);
return *this;
}

virtual ~AbstractRegistry(void) {
}

virtual inline iterator begin(void) ELPP_FINAL {
return m_list.begin();
}

virtual inline iterator end(void) ELPP_FINAL {
return m_list.end();
}


virtual inline const_iterator cbegin(void) const ELPP_FINAL {
return m_list.cbegin();
}

virtual inline const_iterator cend(void) const ELPP_FINAL {
return m_list.cend();
}

virtual inline bool empty(void) const ELPP_FINAL {
return m_list.empty();
}

virtual inline std::size_t size(void) const ELPP_FINAL {
return m_list.size();
}

virtual inline Container& list(void) ELPP_FINAL {
return m_list;
}

virtual inline const Container& list(void) const ELPP_FINAL {
return m_list;
}

virtual void unregisterAll(void) = 0;

protected:
virtual void deepCopy(const AbstractRegistry<T_Ptr, Container>&) = 0;
void reinitDeepCopy(const AbstractRegistry<T_Ptr, Container>& sr) {
unregisterAll();
deepCopy(sr);
}

private:
Container m_list;
};

template <typename T_Ptr, typename T_Key = const char*>
class Registry : public AbstractRegistry<T_Ptr, std::map<T_Key, T_Ptr*>> {
public:
typedef typename Registry<T_Ptr, T_Key>::iterator iterator;
typedef typename Registry<T_Ptr, T_Key>::const_iterator const_iterator;

Registry(void) {}

Registry(const Registry& sr) : AbstractRegistry<T_Ptr, std::vector<T_Ptr*>>() {
if (this == &sr) {
return;
}
this->reinitDeepCopy(sr);
}

Registry& operator=(const Registry& sr) {
if (this == &sr) {
return *this;
}
this->reinitDeepCopy(sr);
return *this;
}

virtual ~Registry(void) {
unregisterAll();
}

protected:
virtual void unregisterAll(void) ELPP_FINAL {
if (!this->empty()) {
for (auto&& curr : this->list()) {
base::utils::safeDelete(curr.second);
}
this->list().clear();
}
}

virtual void registerNew(const T_Key& uniqKey, T_Ptr* ptr) ELPP_FINAL {
unregister(uniqKey);
this->list().insert(std::make_pair(uniqKey, ptr));
}

void unregister(const T_Key& uniqKey) {
T_Ptr* existing = get(uniqKey);
if (existing != nullptr) {
base::utils::safeDelete(existing);
this->list().erase(uniqKey);
}
}

T_Ptr* get(const T_Key& uniqKey) {
iterator it = this->list().find(uniqKey);
return it == this->list().end()
? nullptr
: it->second;
}

private:
virtual void deepCopy(const AbstractRegistry<T_Ptr, std::map<T_Key, T_Ptr*>>& sr) ELPP_FINAL {
for (const_iterator it = sr.cbegin(); it != sr.cend(); ++it) {
registerNew(it->first, new T_Ptr(*it->second));
}
}
};

template <typename T_Ptr, typename Pred>
class RegistryWithPred : public AbstractRegistry<T_Ptr, std::vector<T_Ptr*>> {
public:
typedef typename RegistryWithPred<T_Ptr, Pred>::iterator iterator;
typedef typename RegistryWithPred<T_Ptr, Pred>::const_iterator const_iterator;

RegistryWithPred(void) {
}

virtual ~RegistryWithPred(void) {
unregisterAll();
}

RegistryWithPred(const RegistryWithPred& sr) : AbstractRegistry<T_Ptr, std::vector<T_Ptr*>>() {
if (this == &sr) {
return;
}
this->reinitDeepCopy(sr);
}

RegistryWithPred& operator=(const RegistryWithPred& sr) {
if (this == &sr) {
return *this;
}
this->reinitDeepCopy(sr);
return *this;
}

friend base::type::ostream_t& operator<<(base::type::ostream_t& os, const RegistryWithPred& sr) {
for (const_iterator it = sr.list().begin(); it != sr.list().end(); ++it) {
os << ELPP_LITERAL("    ") << **it << ELPP_LITERAL("\n");
}
return os;
}

protected:
virtual void unregisterAll(void) ELPP_FINAL {
if (!this->empty()) {
for (auto&& curr : this->list()) {
base::utils::safeDelete(curr);
}
this->list().clear();
}
}

virtual void unregister(T_Ptr*& ptr) ELPP_FINAL {
if (ptr) {
iterator iter = this->begin();
for (; iter != this->end(); ++iter) {
if (ptr == *iter) {
break;
}
}
if (iter != this->end() && *iter != nullptr) {
this->list().erase(iter);
base::utils::safeDelete(*iter);
}
}
}

virtual inline void registerNew(T_Ptr* ptr) ELPP_FINAL {
this->list().push_back(ptr);
}

template <typename T, typename T2>
T_Ptr* get(const T& arg1, const T2 arg2) {
iterator iter = std::find_if(this->list().begin(), this->list().end(), Pred(arg1, arg2));
if (iter != this->list().end() && *iter != nullptr) {
return *iter;
}
return nullptr;
}

private:
virtual void deepCopy(const AbstractRegistry<T_Ptr, std::vector<T_Ptr*>>& sr) {
for (const_iterator it = sr.list().begin(); it != sr.list().end(); ++it) {
registerNew(new T_Ptr(**it));
}
}
};
class Utils {
public:
template <typename T, typename TPtr>
static bool installCallback(const std::string& id, std::map<std::string, TPtr>* mapT) {
if (mapT->find(id) == mapT->end()) {
mapT->insert(std::make_pair(id, TPtr(new T())));
return true;
}
return false;
}

template <typename T, typename TPtr>
static void uninstallCallback(const std::string& id, std::map<std::string, TPtr>* mapT) {
if (mapT->find(id) != mapT->end()) {
mapT->erase(id);
}
}

template <typename T, typename TPtr>
static T* callback(const std::string& id, std::map<std::string, TPtr>* mapT) {
typename std::map<std::string, TPtr>::iterator iter = mapT->find(id);
if (iter != mapT->end()) {
return static_cast<T*>(iter->second.get());
}
return nullptr;
}
};
}  
} 
class Loggable {
public:
virtual ~Loggable(void) {}
virtual void log(el::base::type::ostream_t&) const = 0;
private:
friend inline el::base::type::ostream_t& operator<<(el::base::type::ostream_t& os, const Loggable& loggable) {
loggable.log(os);
return os;
}
};
namespace base {
class LogFormat : public Loggable {
public:
LogFormat(void);
LogFormat(Level level, const base::type::string_t& format);
LogFormat(const LogFormat& logFormat);
LogFormat(LogFormat&& logFormat);
LogFormat& operator=(const LogFormat& logFormat);
virtual ~LogFormat(void) {}
bool operator==(const LogFormat& other);

void parseFromFormat(const base::type::string_t& userFormat);

inline Level level(void) const {
return m_level;
}

inline const base::type::string_t& userFormat(void) const {
return m_userFormat;
}

inline const base::type::string_t& format(void) const {
return m_format;
}

inline const std::string& dateTimeFormat(void) const {
return m_dateTimeFormat;
}

inline base::type::EnumType flags(void) const {
return m_flags;
}

inline bool hasFlag(base::FormatFlags flag) const {
return base::utils::hasFlag(flag, m_flags);
}

virtual void log(el::base::type::ostream_t& os) const {
os << m_format;
}

protected:
virtual void updateDateFormat(std::size_t index, base::type::string_t& currFormat) ELPP_FINAL;

virtual void updateFormatSpec(void) ELPP_FINAL;

inline void addFlag(base::FormatFlags flag) {
base::utils::addFlag(flag, &m_flags);
}

private:
Level m_level;
base::type::string_t m_userFormat;
base::type::string_t m_format;
std::string m_dateTimeFormat;
base::type::EnumType m_flags;
std::string m_currentUser;
std::string m_currentHost;
friend class el::Logger;  
};
}  
typedef std::function<std::string(const LogMessage*)> FormatSpecifierValueResolver;
class CustomFormatSpecifier {
public:
CustomFormatSpecifier(const char* formatSpecifier, const FormatSpecifierValueResolver& resolver) :
m_formatSpecifier(formatSpecifier), m_resolver(resolver) {}
inline const char* formatSpecifier(void) const {
return m_formatSpecifier;
}
inline const FormatSpecifierValueResolver& resolver(void) const {
return m_resolver;
}
inline bool operator==(const char* formatSpecifier) {
return strcmp(m_formatSpecifier, formatSpecifier) == 0;
}

private:
const char* m_formatSpecifier;
FormatSpecifierValueResolver m_resolver;
};
class Configuration : public Loggable {
public:
Configuration(const Configuration& c);
Configuration& operator=(const Configuration& c);

virtual ~Configuration(void) {
}

Configuration(Level level, ConfigurationType configurationType, const std::string& value);

inline Level level(void) const {
return m_level;
}

inline ConfigurationType configurationType(void) const {
return m_configurationType;
}

inline const std::string& value(void) const {
return m_value;
}

inline void setValue(const std::string& value) {
m_value = value;
}

virtual void log(el::base::type::ostream_t& os) const;

class Predicate {
public:
Predicate(Level level, ConfigurationType configurationType);

bool operator()(const Configuration* conf) const;

private:
Level m_level;
ConfigurationType m_configurationType;
};

private:
Level m_level;
ConfigurationType m_configurationType;
std::string m_value;
};

class Configurations : public base::utils::RegistryWithPred<Configuration, Configuration::Predicate> {
public:
Configurations(void);

Configurations(const std::string& configurationFile, bool useDefaultsForRemaining = true,
Configurations* base = nullptr);

virtual ~Configurations(void) {
}

bool parseFromFile(const std::string& configurationFile, Configurations* base = nullptr);

bool parseFromText(const std::string& configurationsString, Configurations* base = nullptr);

void setFromBase(Configurations* base);

bool hasConfiguration(ConfigurationType configurationType);

bool hasConfiguration(Level level, ConfigurationType configurationType);

void set(Level level, ConfigurationType configurationType, const std::string& value);

void set(Configuration* conf);

inline Configuration* get(Level level, ConfigurationType configurationType) {
base::threading::ScopedLock scopedLock(lock());
return RegistryWithPred<Configuration, Configuration::Predicate>::get(level, configurationType);
}

inline void setGlobally(ConfigurationType configurationType, const std::string& value) {
setGlobally(configurationType, value, false);
}

inline void clear(void) {
base::threading::ScopedLock scopedLock(lock());
unregisterAll();
}

inline const std::string& configurationFile(void) const {
return m_configurationFile;
}

void setToDefault(void);

void setRemainingToDefault(void);

class Parser : base::StaticClass {
public:
static bool parseFromFile(const std::string& configurationFile, Configurations* sender,
Configurations* base = nullptr);

static bool parseFromText(const std::string& configurationsString, Configurations* sender,
Configurations* base = nullptr);

private:
friend class el::Loggers;
static void ignoreComments(std::string* line);
static bool isLevel(const std::string& line);
static bool isComment(const std::string& line);
static inline bool isConfig(const std::string& line);
static bool parseLine(std::string* line, std::string* currConfigStr, std::string* currLevelStr, Level* currLevel,
Configurations* conf);
};

private:
std::string m_configurationFile;
bool m_isFromFile;
friend class el::Loggers;

void unsafeSetIfNotExist(Level level, ConfigurationType configurationType, const std::string& value);

void unsafeSet(Level level, ConfigurationType configurationType, const std::string& value);

void setGlobally(ConfigurationType configurationType, const std::string& value, bool includeGlobalLevel);

void unsafeSetGlobally(ConfigurationType configurationType, const std::string& value, bool includeGlobalLevel);
};

namespace base {
typedef std::shared_ptr<base::type::fstream_t> FileStreamPtr;
typedef std::map<std::string, FileStreamPtr> LogStreamsReferenceMap;
class TypedConfigurations : public base::threading::ThreadSafe {
public:
TypedConfigurations(Configurations* configurations, base::LogStreamsReferenceMap* logStreamsReference);

TypedConfigurations(const TypedConfigurations& other);

virtual ~TypedConfigurations(void) {
}

const Configurations* configurations(void) const {
return m_configurations;
}

bool enabled(Level level);
bool toFile(Level level);
const std::string& filename(Level level);
bool toStandardOutput(Level level);
const base::LogFormat& logFormat(Level level);
const base::SubsecondPrecision& subsecondPrecision(Level level = Level::Global);
const base::MillisecondsWidth& millisecondsWidth(Level level = Level::Global);
bool performanceTracking(Level level = Level::Global);
base::type::fstream_t* fileStream(Level level);
std::size_t maxLogFileSize(Level level);
std::size_t logFlushThreshold(Level level);

private:
Configurations* m_configurations;
std::map<Level, bool> m_enabledMap;
std::map<Level, bool> m_toFileMap;
std::map<Level, std::string> m_filenameMap;
std::map<Level, bool> m_toStandardOutputMap;
std::map<Level, base::LogFormat> m_logFormatMap;
std::map<Level, base::SubsecondPrecision> m_subsecondPrecisionMap;
std::map<Level, bool> m_performanceTrackingMap;
std::map<Level, base::FileStreamPtr> m_fileStreamMap;
std::map<Level, std::size_t> m_maxLogFileSizeMap;
std::map<Level, std::size_t> m_logFlushThresholdMap;
base::LogStreamsReferenceMap* m_logStreamsReference;

friend class el::Helpers;
friend class el::base::MessageBuilder;
friend class el::base::Writer;
friend class el::base::DefaultLogDispatchCallback;
friend class el::base::LogDispatcher;

template <typename Conf_T>
inline Conf_T getConfigByVal(Level level, const std::map<Level, Conf_T>* confMap, const char* confName) {
base::threading::ScopedLock scopedLock(lock());
return unsafeGetConfigByVal(level, confMap, confName);  
}

template <typename Conf_T>
inline Conf_T& getConfigByRef(Level level, std::map<Level, Conf_T>* confMap, const char* confName) {
base::threading::ScopedLock scopedLock(lock());
return unsafeGetConfigByRef(level, confMap, confName);  
}

template <typename Conf_T>
Conf_T unsafeGetConfigByVal(Level level, const std::map<Level, Conf_T>* confMap, const char* confName) {
ELPP_UNUSED(confName);
typename std::map<Level, Conf_T>::const_iterator it = confMap->find(level);
if (it == confMap->end()) {
try {
return confMap->at(Level::Global);
} catch (...) {
ELPP_INTERNAL_ERROR("Unable to get configuration [" << confName << "] for level ["
<< LevelHelper::convertToString(level) << "]"
<< std::endl << "Please ensure you have properly configured logger.", false);
return Conf_T();
}
}
return it->second;
}

template <typename Conf_T>
Conf_T& unsafeGetConfigByRef(Level level, std::map<Level, Conf_T>* confMap, const char* confName) {
ELPP_UNUSED(confName);
typename std::map<Level, Conf_T>::iterator it = confMap->find(level);
if (it == confMap->end()) {
try {
return confMap->at(Level::Global);
} catch (...) {
ELPP_INTERNAL_ERROR("Unable to get configuration [" << confName << "] for level ["
<< LevelHelper::convertToString(level) << "]"
<< std::endl << "Please ensure you have properly configured logger.", false);
}
}
return it->second;
}

template <typename Conf_T>
void setValue(Level level, const Conf_T& value, std::map<Level, Conf_T>* confMap, bool includeGlobalLevel = true) {
if (confMap->empty() && includeGlobalLevel) {
confMap->insert(std::make_pair(Level::Global, value));
return;
}
typename std::map<Level, Conf_T>::iterator it = confMap->find(Level::Global);
if (it != confMap->end() && it->second == value) {
return;
}
it = confMap->find(level);
if (it == confMap->end()) {
confMap->insert(std::make_pair(level, value));
} else {
confMap->at(level) = value;
}
}

void build(Configurations* configurations);
unsigned long getULong(std::string confVal);
std::string resolveFilename(const std::string& filename);
void insertFile(Level level, const std::string& fullFilename);
bool unsafeValidateFileRolling(Level level, const PreRollOutCallback& preRollOutCallback);

inline bool validateFileRolling(Level level, const PreRollOutCallback& preRollOutCallback) {
base::threading::ScopedLock scopedLock(lock());
return unsafeValidateFileRolling(level, preRollOutCallback);
}
};
class HitCounter {
public:
HitCounter(void) :
m_filename(""),
m_lineNumber(0),
m_hitCounts(0) {
}

HitCounter(const char* filename, base::type::LineNumber lineNumber) :
m_filename(filename),
m_lineNumber(lineNumber),
m_hitCounts(0) {
}

HitCounter(const HitCounter& hitCounter) :
m_filename(hitCounter.m_filename),
m_lineNumber(hitCounter.m_lineNumber),
m_hitCounts(hitCounter.m_hitCounts) {
}

HitCounter& operator=(const HitCounter& hitCounter) {
if (&hitCounter != this) {
m_filename = hitCounter.m_filename;
m_lineNumber = hitCounter.m_lineNumber;
m_hitCounts = hitCounter.m_hitCounts;
}
return *this;
}

virtual ~HitCounter(void) {
}

inline void resetLocation(const char* filename, base::type::LineNumber lineNumber) {
m_filename = filename;
m_lineNumber = lineNumber;
}

inline void validateHitCounts(std::size_t n) {
if (m_hitCounts >= base::consts::kMaxLogPerCounter) {
m_hitCounts = (n >= 1 ? base::consts::kMaxLogPerCounter % n : 0);
}
++m_hitCounts;
}

inline const char* filename(void) const {
return m_filename;
}

inline base::type::LineNumber lineNumber(void) const {
return m_lineNumber;
}

inline std::size_t hitCounts(void) const {
return m_hitCounts;
}

inline void increment(void) {
++m_hitCounts;
}

class Predicate {
public:
Predicate(const char* filename, base::type::LineNumber lineNumber)
: m_filename(filename),
m_lineNumber(lineNumber) {
}
inline bool operator()(const HitCounter* counter) {
return ((counter != nullptr) &&
(strcmp(counter->m_filename, m_filename) == 0) &&
(counter->m_lineNumber == m_lineNumber));
}

private:
const char* m_filename;
base::type::LineNumber m_lineNumber;
};

private:
const char* m_filename;
base::type::LineNumber m_lineNumber;
std::size_t m_hitCounts;
};
class RegisteredHitCounters : public base::utils::RegistryWithPred<base::HitCounter, base::HitCounter::Predicate> {
public:
bool validateEveryN(const char* filename, base::type::LineNumber lineNumber, std::size_t n);

bool validateAfterN(const char* filename, base::type::LineNumber lineNumber, std::size_t n);

bool validateNTimes(const char* filename, base::type::LineNumber lineNumber, std::size_t n);

inline const base::HitCounter* getCounter(const char* filename, base::type::LineNumber lineNumber) {
base::threading::ScopedLock scopedLock(lock());
return get(filename, lineNumber);
}
};
enum class DispatchAction : base::type::EnumType {
None = 1, NormalLog = 2, SysLog = 4
};
}  
template <typename T>
class Callback : protected base::threading::ThreadSafe {
public:
Callback(void) : m_enabled(true) {}
inline bool enabled(void) const {
return m_enabled;
}
inline void setEnabled(bool enabled) {
base::threading::ScopedLock scopedLock(lock());
m_enabled = enabled;
}
protected:
virtual void handle(const T* handlePtr) = 0;
private:
bool m_enabled;
};
class LogDispatchData {
public:
LogDispatchData() : m_logMessage(nullptr), m_dispatchAction(base::DispatchAction::None) {}
inline const LogMessage* logMessage(void) const {
return m_logMessage;
}
inline base::DispatchAction dispatchAction(void) const {
return m_dispatchAction;
}
private:
LogMessage* m_logMessage;
base::DispatchAction m_dispatchAction;
friend class base::LogDispatcher;

inline void setLogMessage(LogMessage* logMessage) {
m_logMessage = logMessage;
}
inline void setDispatchAction(base::DispatchAction dispatchAction) {
m_dispatchAction = dispatchAction;
}
};
class LogDispatchCallback : public Callback<LogDispatchData> {
private:
friend class base::LogDispatcher;
};
class PerformanceTrackingCallback : public Callback<PerformanceTrackingData> {
private:
friend class base::PerformanceTracker;
};
class LoggerRegistrationCallback : public Callback<Logger> {
private:
friend class base::RegisteredLoggers;
};
class LogBuilder : base::NoCopy {
public:
LogBuilder() : m_termSupportsColor(base::utils::OS::termSupportsColor()) {}
virtual ~LogBuilder(void) {
ELPP_INTERNAL_INFO(3, "Destroying log builder...")
}
virtual base::type::string_t build(const LogMessage* logMessage, bool appendNewLine) const = 0;
void convertToColoredOutput(base::type::string_t* logLine, Level level);
private:
bool m_termSupportsColor;
friend class el::base::DefaultLogDispatchCallback;
};
typedef std::shared_ptr<LogBuilder> LogBuilderPtr;
class Logger : public base::threading::ThreadSafe, public Loggable {
public:
Logger(const std::string& id, base::LogStreamsReferenceMap* logStreamsReference);
Logger(const std::string& id, const Configurations& configurations, base::LogStreamsReferenceMap* logStreamsReference);
Logger(const Logger& logger);
Logger& operator=(const Logger& logger);

virtual ~Logger(void) {
base::utils::safeDelete(m_typedConfigurations);
}

virtual inline void log(el::base::type::ostream_t& os) const {
os << m_id.c_str();
}

void configure(const Configurations& configurations);

void reconfigure(void);

inline const std::string& id(void) const {
return m_id;
}

inline const std::string& parentApplicationName(void) const {
return m_parentApplicationName;
}

inline void setParentApplicationName(const std::string& parentApplicationName) {
m_parentApplicationName = parentApplicationName;
}

inline Configurations* configurations(void) {
return &m_configurations;
}

inline base::TypedConfigurations* typedConfigurations(void) {
return m_typedConfigurations;
}

static bool isValidId(const std::string& id);

void flush(void);

void flush(Level level, base::type::fstream_t* fs);

inline bool isFlushNeeded(Level level) {
return ++m_unflushedCount.find(level)->second >= m_typedConfigurations->logFlushThreshold(level);
}

inline LogBuilder* logBuilder(void) const {
return m_logBuilder.get();
}

inline void setLogBuilder(const LogBuilderPtr& logBuilder) {
m_logBuilder = logBuilder;
}

inline bool enabled(Level level) const {
return m_typedConfigurations->enabled(level);
}

#if ELPP_VARIADIC_TEMPLATES_SUPPORTED
#  define LOGGER_LEVEL_WRITERS_SIGNATURES(FUNCTION_NAME)\
template <typename T, typename... Args>\
inline void FUNCTION_NAME(const char*, const T&, const Args&...);\
template <typename T>\
inline void FUNCTION_NAME(const T&);

template <typename T, typename... Args>
inline void verbose(int, const char*, const T&, const Args&...);

template <typename T>
inline void verbose(int, const T&);

LOGGER_LEVEL_WRITERS_SIGNATURES(info)
LOGGER_LEVEL_WRITERS_SIGNATURES(debug)
LOGGER_LEVEL_WRITERS_SIGNATURES(warn)
LOGGER_LEVEL_WRITERS_SIGNATURES(error)
LOGGER_LEVEL_WRITERS_SIGNATURES(fatal)
LOGGER_LEVEL_WRITERS_SIGNATURES(trace)
#  undef LOGGER_LEVEL_WRITERS_SIGNATURES
#endif 
private:
std::string m_id;
base::TypedConfigurations* m_typedConfigurations;
base::type::stringstream_t m_stream;
std::string m_parentApplicationName;
bool m_isConfigured;
Configurations m_configurations;
std::map<Level, unsigned int> m_unflushedCount;
base::LogStreamsReferenceMap* m_logStreamsReference;
LogBuilderPtr m_logBuilder;

friend class el::LogMessage;
friend class el::Loggers;
friend class el::Helpers;
friend class el::base::RegisteredLoggers;
friend class el::base::DefaultLogDispatchCallback;
friend class el::base::MessageBuilder;
friend class el::base::Writer;
friend class el::base::PErrorWriter;
friend class el::base::Storage;
friend class el::base::PerformanceTracker;
friend class el::base::LogDispatcher;

Logger(void);

#if ELPP_VARIADIC_TEMPLATES_SUPPORTED
template <typename T, typename... Args>
void log_(Level, int, const char*, const T&, const Args&...);

template <typename T>
inline void log_(Level, int, const T&);

template <typename T, typename... Args>
void log(Level, const char*, const T&, const Args&...);

template <typename T>
inline void log(Level, const T&);
#endif 

void initUnflushedCount(void);

inline base::type::stringstream_t& stream(void) {
return m_stream;
}

void resolveLoggerFormatSpec(void) const;
};
namespace base {
class RegisteredLoggers : public base::utils::Registry<Logger, std::string> {
public:
explicit RegisteredLoggers(const LogBuilderPtr& defaultLogBuilder);

virtual ~RegisteredLoggers(void) {
unsafeFlushAll();
}

inline void setDefaultConfigurations(const Configurations& configurations) {
base::threading::ScopedLock scopedLock(lock());
m_defaultConfigurations.setFromBase(const_cast<Configurations*>(&configurations));
}

inline Configurations* defaultConfigurations(void) {
return &m_defaultConfigurations;
}

Logger* get(const std::string& id, bool forceCreation = true);

template <typename T>
inline bool installLoggerRegistrationCallback(const std::string& id) {
return base::utils::Utils::installCallback<T, base::type::LoggerRegistrationCallbackPtr>(id,
&m_loggerRegistrationCallbacks);
}

template <typename T>
inline void uninstallLoggerRegistrationCallback(const std::string& id) {
base::utils::Utils::uninstallCallback<T, base::type::LoggerRegistrationCallbackPtr>(id, &m_loggerRegistrationCallbacks);
}

template <typename T>
inline T* loggerRegistrationCallback(const std::string& id) {
return base::utils::Utils::callback<T, base::type::LoggerRegistrationCallbackPtr>(id, &m_loggerRegistrationCallbacks);
}

bool remove(const std::string& id);

inline bool has(const std::string& id) {
return get(id, false) != nullptr;
}

inline void unregister(Logger*& logger) {
base::threading::ScopedLock scopedLock(lock());
base::utils::Registry<Logger, std::string>::unregister(logger->id());
}

inline base::LogStreamsReferenceMap* logStreamsReference(void) {
return &m_logStreamsReference;
}

inline void flushAll(void) {
base::threading::ScopedLock scopedLock(lock());
unsafeFlushAll();
}

inline void setDefaultLogBuilder(LogBuilderPtr& logBuilderPtr) {
base::threading::ScopedLock scopedLock(lock());
m_defaultLogBuilder = logBuilderPtr;
}

private:
LogBuilderPtr m_defaultLogBuilder;
Configurations m_defaultConfigurations;
base::LogStreamsReferenceMap m_logStreamsReference;
std::map<std::string, base::type::LoggerRegistrationCallbackPtr> m_loggerRegistrationCallbacks;
friend class el::base::Storage;

void unsafeFlushAll(void);
};
class VRegistry : base::NoCopy, public base::threading::ThreadSafe {
public:
explicit VRegistry(base::type::VerboseLevel level, base::type::EnumType* pFlags);

void setLevel(base::type::VerboseLevel level);

inline base::type::VerboseLevel level(void) const {
return m_level;
}

inline void clearModules(void) {
base::threading::ScopedLock scopedLock(lock());
m_modules.clear();
}

void setModules(const char* modules);

bool allowed(base::type::VerboseLevel vlevel, const char* file);

inline const std::map<std::string, base::type::VerboseLevel>& modules(void) const {
return m_modules;
}

void setFromArgs(const base::utils::CommandLineArgs* commandLineArgs);

inline bool vModulesEnabled(void) {
return !base::utils::hasFlag(LoggingFlag::DisableVModules, *m_pFlags);
}

private:
base::type::VerboseLevel m_level;
base::type::EnumType* m_pFlags;
std::map<std::string, base::type::VerboseLevel> m_modules;
};
}  
class LogMessage {
public:
LogMessage(Level level, const std::string& file, base::type::LineNumber line, const std::string& func,
base::type::VerboseLevel verboseLevel, Logger* logger) :
m_level(level), m_file(file), m_line(line), m_func(func),
m_verboseLevel(verboseLevel), m_logger(logger), m_message(logger->stream().str()) {
}
inline Level level(void) const {
return m_level;
}
inline const std::string& file(void) const {
return m_file;
}
inline base::type::LineNumber line(void) const {
return m_line;
}
inline const std::string& func(void) const {
return m_func;
}
inline base::type::VerboseLevel verboseLevel(void) const {
return m_verboseLevel;
}
inline Logger* logger(void) const {
return m_logger;
}
inline const base::type::string_t& message(void) const {
return m_message;
}
private:
Level m_level;
std::string m_file;
base::type::LineNumber m_line;
std::string m_func;
base::type::VerboseLevel m_verboseLevel;
Logger* m_logger;
base::type::string_t m_message;
};
namespace base {
#if ELPP_ASYNC_LOGGING
class AsyncLogItem {
public:
explicit AsyncLogItem(const LogMessage& logMessage, const LogDispatchData& data, const base::type::string_t& logLine)
: m_logMessage(logMessage), m_dispatchData(data), m_logLine(logLine) {}
virtual ~AsyncLogItem() {}
inline LogMessage* logMessage(void) {
return &m_logMessage;
}
inline LogDispatchData* data(void) {
return &m_dispatchData;
}
inline base::type::string_t logLine(void) {
return m_logLine;
}
private:
LogMessage m_logMessage;
LogDispatchData m_dispatchData;
base::type::string_t m_logLine;
};
class AsyncLogQueue : public base::threading::ThreadSafe {
public:
virtual ~AsyncLogQueue() {
ELPP_INTERNAL_INFO(6, "~AsyncLogQueue");
}

inline AsyncLogItem next(void) {
base::threading::ScopedLock scopedLock(lock());
AsyncLogItem result = m_queue.front();
m_queue.pop();
return result;
}

inline void push(const AsyncLogItem& item) {
base::threading::ScopedLock scopedLock(lock());
m_queue.push(item);
}
inline void pop(void) {
base::threading::ScopedLock scopedLock(lock());
m_queue.pop();
}
inline AsyncLogItem front(void) {
base::threading::ScopedLock scopedLock(lock());
return m_queue.front();
}
inline bool empty(void) {
base::threading::ScopedLock scopedLock(lock());
return m_queue.empty();
}
private:
std::queue<AsyncLogItem> m_queue;
};
class IWorker {
public:
virtual ~IWorker() {}
virtual void start() = 0;
};
#endif 
class Storage : base::NoCopy, public base::threading::ThreadSafe {
public:
#if ELPP_ASYNC_LOGGING
Storage(const LogBuilderPtr& defaultLogBuilder, base::IWorker* asyncDispatchWorker);
#else
explicit Storage(const LogBuilderPtr& defaultLogBuilder);
#endif  

virtual ~Storage(void);

inline bool validateEveryNCounter(const char* filename, base::type::LineNumber lineNumber, std::size_t occasion) {
return hitCounters()->validateEveryN(filename, lineNumber, occasion);
}

inline bool validateAfterNCounter(const char* filename, base::type::LineNumber lineNumber, std::size_t n) {
return hitCounters()->validateAfterN(filename, lineNumber, n);
}

inline bool validateNTimesCounter(const char* filename, base::type::LineNumber lineNumber, std::size_t n) {
return hitCounters()->validateNTimes(filename, lineNumber, n);
}

inline base::RegisteredHitCounters* hitCounters(void) const {
return m_registeredHitCounters;
}

inline base::RegisteredLoggers* registeredLoggers(void) const {
return m_registeredLoggers;
}

inline base::VRegistry* vRegistry(void) const {
return m_vRegistry;
}

#if ELPP_ASYNC_LOGGING
inline base::AsyncLogQueue* asyncLogQueue(void) const {
return m_asyncLogQueue;
}
#endif  

inline const base::utils::CommandLineArgs* commandLineArgs(void) const {
return &m_commandLineArgs;
}

inline void addFlag(LoggingFlag flag) {
base::utils::addFlag(flag, &m_flags);
}

inline void removeFlag(LoggingFlag flag) {
base::utils::removeFlag(flag, &m_flags);
}

inline bool hasFlag(LoggingFlag flag) const {
return base::utils::hasFlag(flag, m_flags);
}

inline base::type::EnumType flags(void) const {
return m_flags;
}

inline void setFlags(base::type::EnumType flags) {
m_flags = flags;
}

inline void setPreRollOutCallback(const PreRollOutCallback& callback) {
m_preRollOutCallback = callback;
}

inline void unsetPreRollOutCallback(void) {
m_preRollOutCallback = base::defaultPreRollOutCallback;
}

inline PreRollOutCallback& preRollOutCallback(void) {
return m_preRollOutCallback;
}

bool hasCustomFormatSpecifier(const char* formatSpecifier);
void installCustomFormatSpecifier(const CustomFormatSpecifier& customFormatSpecifier);
bool uninstallCustomFormatSpecifier(const char* formatSpecifier);

const std::vector<CustomFormatSpecifier>* customFormatSpecifiers(void) const {
return &m_customFormatSpecifiers;
}

inline void setLoggingLevel(Level level) {
m_loggingLevel = level;
}

template <typename T>
inline bool installLogDispatchCallback(const std::string& id) {
return base::utils::Utils::installCallback<T, base::type::LogDispatchCallbackPtr>(id, &m_logDispatchCallbacks);
}

template <typename T>
inline void uninstallLogDispatchCallback(const std::string& id) {
base::utils::Utils::uninstallCallback<T, base::type::LogDispatchCallbackPtr>(id, &m_logDispatchCallbacks);
}
template <typename T>
inline T* logDispatchCallback(const std::string& id) {
return base::utils::Utils::callback<T, base::type::LogDispatchCallbackPtr>(id, &m_logDispatchCallbacks);
}

#if defined(ELPP_FEATURE_ALL) || defined(ELPP_FEATURE_PERFORMANCE_TRACKING)
template <typename T>
inline bool installPerformanceTrackingCallback(const std::string& id) {
return base::utils::Utils::installCallback<T, base::type::PerformanceTrackingCallbackPtr>(id,
&m_performanceTrackingCallbacks);
}

template <typename T>
inline void uninstallPerformanceTrackingCallback(const std::string& id) {
base::utils::Utils::uninstallCallback<T, base::type::PerformanceTrackingCallbackPtr>(id,
&m_performanceTrackingCallbacks);
}

template <typename T>
inline T* performanceTrackingCallback(const std::string& id) {
return base::utils::Utils::callback<T, base::type::PerformanceTrackingCallbackPtr>(id, &m_performanceTrackingCallbacks);
}
#endif 

inline void setThreadName(const std::string& name) {
if (name.empty()) return;
base::threading::ScopedLock scopedLock(lock());
m_threadNames[base::threading::getCurrentThreadId()] = name;
}

inline std::string getThreadName(const std::string& threadId) {
std::map<std::string, std::string>::const_iterator it = m_threadNames.find(threadId);
if (it == m_threadNames.end()) {
return threadId;
}
return it->second;
}
private:
base::RegisteredHitCounters* m_registeredHitCounters;
base::RegisteredLoggers* m_registeredLoggers;
base::type::EnumType m_flags;
base::VRegistry* m_vRegistry;
#if ELPP_ASYNC_LOGGING
base::AsyncLogQueue* m_asyncLogQueue;
base::IWorker* m_asyncDispatchWorker;
#endif  
base::utils::CommandLineArgs m_commandLineArgs;
PreRollOutCallback m_preRollOutCallback;
std::map<std::string, base::type::LogDispatchCallbackPtr> m_logDispatchCallbacks;
std::map<std::string, base::type::PerformanceTrackingCallbackPtr> m_performanceTrackingCallbacks;
std::map<std::string, std::string> m_threadNames;
std::vector<CustomFormatSpecifier> m_customFormatSpecifiers;
Level m_loggingLevel;

friend class el::Helpers;
friend class el::base::DefaultLogDispatchCallback;
friend class el::LogBuilder;
friend class el::base::MessageBuilder;
friend class el::base::Writer;
friend class el::base::PerformanceTracker;
friend class el::base::LogDispatcher;

void setApplicationArguments(int argc, char** argv);

inline void setApplicationArguments(int argc, const char** argv) {
setApplicationArguments(argc, const_cast<char**>(argv));
}
};
extern ELPP_EXPORT base::type::StoragePointer elStorage;
#define ELPP el::base::elStorage
class DefaultLogDispatchCallback : public LogDispatchCallback {
protected:
void handle(const LogDispatchData* data);
private:
const LogDispatchData* m_data;
void dispatch(base::type::string_t&& logLine);
};
#if ELPP_ASYNC_LOGGING
class AsyncLogDispatchCallback : public LogDispatchCallback {
protected:
void handle(const LogDispatchData* data);
};
class AsyncDispatchWorker : public base::IWorker, public base::threading::ThreadSafe {
public:
AsyncDispatchWorker();
virtual ~AsyncDispatchWorker();

bool clean(void);
void emptyQueue(void);
virtual void start(void);
void handle(AsyncLogItem* logItem);
void run(void);

void setContinueRunning(bool value) {
base::threading::ScopedLock scopedLock(m_continueRunningMutex);
m_continueRunning = value;
}

bool continueRunning(void) const {
return m_continueRunning;
}
private:
std::condition_variable cv;
bool m_continueRunning;
base::threading::Mutex m_continueRunningMutex;
};
#endif  
}  
namespace base {
class DefaultLogBuilder : public LogBuilder {
public:
base::type::string_t build(const LogMessage* logMessage, bool appendNewLine) const;
};
class LogDispatcher : base::NoCopy {
public:
LogDispatcher(bool proceed, LogMessage&& logMessage, base::DispatchAction dispatchAction) :
m_proceed(proceed),
m_logMessage(std::move(logMessage)),
m_dispatchAction(std::move(dispatchAction)) {
}

void dispatch(void);

private:
bool m_proceed;
LogMessage m_logMessage;
base::DispatchAction m_dispatchAction;
};
#if defined(ELPP_STL_LOGGING)
namespace workarounds {
template <typename T, typename Container>
class IterableContainer {
public:
typedef typename Container::iterator iterator;
typedef typename Container::const_iterator const_iterator;
IterableContainer(void) {}
virtual ~IterableContainer(void) {}
iterator begin(void) {
return getContainer().begin();
}
iterator end(void) {
return getContainer().end();
}
private:
virtual Container& getContainer(void) = 0;
};
template<typename T, typename Container = std::vector<T>, typename Comparator = std::less<typename Container::value_type>>
class IterablePriorityQueue : public IterableContainer<T, Container>,
public std::priority_queue<T, Container, Comparator> {
public:
IterablePriorityQueue(std::priority_queue<T, Container, Comparator> queue_) {
std::size_t count_ = 0;
while (++count_ < base::consts::kMaxLogPerContainer && !queue_.empty()) {
this->push(queue_.top());
queue_.pop();
}
}
private:
inline Container& getContainer(void) {
return this->c;
}
};
template<typename T, typename Container = std::deque<T>>
class IterableQueue : public IterableContainer<T, Container>, public std::queue<T, Container> {
public:
IterableQueue(std::queue<T, Container> queue_) {
std::size_t count_ = 0;
while (++count_ < base::consts::kMaxLogPerContainer && !queue_.empty()) {
this->push(queue_.front());
queue_.pop();
}
}
private:
inline Container& getContainer(void) {
return this->c;
}
};
template<typename T, typename Container = std::deque<T>>
class IterableStack : public IterableContainer<T, Container>, public std::stack<T, Container> {
public:
IterableStack(std::stack<T, Container> stack_) {
std::size_t count_ = 0;
while (++count_ < base::consts::kMaxLogPerContainer && !stack_.empty()) {
this->push(stack_.top());
stack_.pop();
}
}
private:
inline Container& getContainer(void) {
return this->c;
}
};
}  
#endif  
class MessageBuilder {
public:
MessageBuilder(void) : m_logger(nullptr), m_containerLogSeperator(ELPP_LITERAL("")) {}
void initialize(Logger* logger);

#  define ELPP_SIMPLE_LOG(LOG_TYPE)\
MessageBuilder& operator<<(LOG_TYPE msg) {\
m_logger->stream() << msg;\
if (ELPP->hasFlag(LoggingFlag::AutoSpacing)) {\
m_logger->stream() << " ";\
}\
return *this;\
}

inline MessageBuilder& operator<<(const std::string& msg) {
return operator<<(msg.c_str());
}
ELPP_SIMPLE_LOG(char)
ELPP_SIMPLE_LOG(bool)
ELPP_SIMPLE_LOG(signed short)
ELPP_SIMPLE_LOG(unsigned short)
ELPP_SIMPLE_LOG(signed int)
ELPP_SIMPLE_LOG(unsigned int)
ELPP_SIMPLE_LOG(signed long)
ELPP_SIMPLE_LOG(unsigned long)
ELPP_SIMPLE_LOG(float)
ELPP_SIMPLE_LOG(double)
ELPP_SIMPLE_LOG(char*)
ELPP_SIMPLE_LOG(const char*)
ELPP_SIMPLE_LOG(const void*)
ELPP_SIMPLE_LOG(long double)
inline MessageBuilder& operator<<(const std::wstring& msg) {
return operator<<(msg.c_str());
}
MessageBuilder& operator<<(const wchar_t* msg);
inline MessageBuilder& operator<<(std::ostream& (*OStreamMani)(std::ostream&)) {
m_logger->stream() << OStreamMani;
return *this;
}
#define ELPP_ITERATOR_CONTAINER_LOG_ONE_ARG(temp)                                                    \
template <typename T>                                                                            \
inline MessageBuilder& operator<<(const temp<T>& template_inst) {                                \
return writeIterator(template_inst.begin(), template_inst.end(), template_inst.size());      \
}
#define ELPP_ITERATOR_CONTAINER_LOG_TWO_ARG(temp)                                                    \
template <typename T1, typename T2>                                                              \
inline MessageBuilder& operator<<(const temp<T1, T2>& template_inst) {                           \
return writeIterator(template_inst.begin(), template_inst.end(), template_inst.size());      \
}
#define ELPP_ITERATOR_CONTAINER_LOG_THREE_ARG(temp)                                                  \
template <typename T1, typename T2, typename T3>                                                 \
inline MessageBuilder& operator<<(const temp<T1, T2, T3>& template_inst) {                       \
return writeIterator(template_inst.begin(), template_inst.end(), template_inst.size());      \
}
#define ELPP_ITERATOR_CONTAINER_LOG_FOUR_ARG(temp)                                                   \
template <typename T1, typename T2, typename T3, typename T4>                                    \
inline MessageBuilder& operator<<(const temp<T1, T2, T3, T4>& template_inst) {                   \
return writeIterator(template_inst.begin(), template_inst.end(), template_inst.size());      \
}
#define ELPP_ITERATOR_CONTAINER_LOG_FIVE_ARG(temp)                                                   \
template <typename T1, typename T2, typename T3, typename T4, typename T5>                       \
inline MessageBuilder& operator<<(const temp<T1, T2, T3, T4, T5>& template_inst) {               \
return writeIterator(template_inst.begin(), template_inst.end(), template_inst.size());      \
}

#if defined(ELPP_STL_LOGGING)
ELPP_ITERATOR_CONTAINER_LOG_TWO_ARG(std::vector)
ELPP_ITERATOR_CONTAINER_LOG_TWO_ARG(std::list)
ELPP_ITERATOR_CONTAINER_LOG_TWO_ARG(std::deque)
ELPP_ITERATOR_CONTAINER_LOG_THREE_ARG(std::set)
ELPP_ITERATOR_CONTAINER_LOG_THREE_ARG(std::multiset)
ELPP_ITERATOR_CONTAINER_LOG_FOUR_ARG(std::map)
ELPP_ITERATOR_CONTAINER_LOG_FOUR_ARG(std::multimap)
template <class T, class Container>
inline MessageBuilder& operator<<(const std::queue<T, Container>& queue_) {
base::workarounds::IterableQueue<T, Container> iterableQueue_ =
static_cast<base::workarounds::IterableQueue<T, Container> >(queue_);
return writeIterator(iterableQueue_.begin(), iterableQueue_.end(), iterableQueue_.size());
}
template <class T, class Container>
inline MessageBuilder& operator<<(const std::stack<T, Container>& stack_) {
base::workarounds::IterableStack<T, Container> iterableStack_ =
static_cast<base::workarounds::IterableStack<T, Container> >(stack_);
return writeIterator(iterableStack_.begin(), iterableStack_.end(), iterableStack_.size());
}
template <class T, class Container, class Comparator>
inline MessageBuilder& operator<<(const std::priority_queue<T, Container, Comparator>& priorityQueue_) {
base::workarounds::IterablePriorityQueue<T, Container, Comparator> iterablePriorityQueue_ =
static_cast<base::workarounds::IterablePriorityQueue<T, Container, Comparator> >(priorityQueue_);
return writeIterator(iterablePriorityQueue_.begin(), iterablePriorityQueue_.end(), iterablePriorityQueue_.size());
}
template <class First, class Second>
MessageBuilder& operator<<(const std::pair<First, Second>& pair_) {
m_logger->stream() << ELPP_LITERAL("(");
operator << (static_cast<First>(pair_.first));
m_logger->stream() << ELPP_LITERAL(", ");
operator << (static_cast<Second>(pair_.second));
m_logger->stream() << ELPP_LITERAL(")");
return *this;
}
template <std::size_t Size>
MessageBuilder& operator<<(const std::bitset<Size>& bitset_) {
m_logger->stream() << ELPP_LITERAL("[");
operator << (bitset_.to_string());
m_logger->stream() << ELPP_LITERAL("]");
return *this;
}
#  if defined(ELPP_LOG_STD_ARRAY)
template <class T, std::size_t Size>
inline MessageBuilder& operator<<(const std::array<T, Size>& array) {
return writeIterator(array.begin(), array.end(), array.size());
}
#  endif  
#  if defined(ELPP_LOG_UNORDERED_MAP)
ELPP_ITERATOR_CONTAINER_LOG_FIVE_ARG(std::unordered_map)
ELPP_ITERATOR_CONTAINER_LOG_FIVE_ARG(std::unordered_multimap)
#  endif  
#  if defined(ELPP_LOG_UNORDERED_SET)
ELPP_ITERATOR_CONTAINER_LOG_FOUR_ARG(std::unordered_set)
ELPP_ITERATOR_CONTAINER_LOG_FOUR_ARG(std::unordered_multiset)
#  endif  
#endif  
#if defined(ELPP_QT_LOGGING)
inline MessageBuilder& operator<<(const QString& msg) {
#  if defined(ELPP_UNICODE)
m_logger->stream() << msg.toStdWString();
#  else
m_logger->stream() << msg.toStdString();
#  endif  
return *this;
}
inline MessageBuilder& operator<<(const QByteArray& msg) {
return operator << (QString(msg));
}
inline MessageBuilder& operator<<(const QStringRef& msg) {
return operator<<(msg.toString());
}
inline MessageBuilder& operator<<(qint64 msg) {
#  if defined(ELPP_UNICODE)
m_logger->stream() << QString::number(msg).toStdWString();
#  else
m_logger->stream() << QString::number(msg).toStdString();
#  endif  
return *this;
}
inline MessageBuilder& operator<<(quint64 msg) {
#  if defined(ELPP_UNICODE)
m_logger->stream() << QString::number(msg).toStdWString();
#  else
m_logger->stream() << QString::number(msg).toStdString();
#  endif  
return *this;
}
inline MessageBuilder& operator<<(QChar msg) {
m_logger->stream() << msg.toLatin1();
return *this;
}
inline MessageBuilder& operator<<(const QLatin1String& msg) {
m_logger->stream() << msg.latin1();
return *this;
}
ELPP_ITERATOR_CONTAINER_LOG_ONE_ARG(QList)
ELPP_ITERATOR_CONTAINER_LOG_ONE_ARG(QVector)
ELPP_ITERATOR_CONTAINER_LOG_ONE_ARG(QQueue)
ELPP_ITERATOR_CONTAINER_LOG_ONE_ARG(QSet)
ELPP_ITERATOR_CONTAINER_LOG_ONE_ARG(QLinkedList)
ELPP_ITERATOR_CONTAINER_LOG_ONE_ARG(QStack)
template <typename First, typename Second>
MessageBuilder& operator<<(const QPair<First, Second>& pair_) {
m_logger->stream() << ELPP_LITERAL("(");
operator << (static_cast<First>(pair_.first));
m_logger->stream() << ELPP_LITERAL(", ");
operator << (static_cast<Second>(pair_.second));
m_logger->stream() << ELPP_LITERAL(")");
return *this;
}
template <typename K, typename V>
MessageBuilder& operator<<(const QMap<K, V>& map_) {
m_logger->stream() << ELPP_LITERAL("[");
QList<K> keys = map_.keys();
typename QList<K>::const_iterator begin = keys.begin();
typename QList<K>::const_iterator end = keys.end();
int max_ = static_cast<int>(base::consts::kMaxLogPerContainer);  
for (int index_ = 0; begin != end && index_ < max_; ++index_, ++begin) {
m_logger->stream() << ELPP_LITERAL("(");
operator << (static_cast<K>(*begin));
m_logger->stream() << ELPP_LITERAL(", ");
operator << (static_cast<V>(map_.value(*begin)));
m_logger->stream() << ELPP_LITERAL(")");
m_logger->stream() << ((index_ < keys.size() -1) ? m_containerLogSeperator : ELPP_LITERAL(""));
}
if (begin != end) {
m_logger->stream() << ELPP_LITERAL("...");
}
m_logger->stream() << ELPP_LITERAL("]");
return *this;
}
template <typename K, typename V>
inline MessageBuilder& operator<<(const QMultiMap<K, V>& map_) {
operator << (static_cast<QMap<K, V>>(map_));
return *this;
}
template <typename K, typename V>
MessageBuilder& operator<<(const QHash<K, V>& hash_) {
m_logger->stream() << ELPP_LITERAL("[");
QList<K> keys = hash_.keys();
typename QList<K>::const_iterator begin = keys.begin();
typename QList<K>::const_iterator end = keys.end();
int max_ = static_cast<int>(base::consts::kMaxLogPerContainer);  
for (int index_ = 0; begin != end && index_ < max_; ++index_, ++begin) {
m_logger->stream() << ELPP_LITERAL("(");
operator << (static_cast<K>(*begin));
m_logger->stream() << ELPP_LITERAL(", ");
operator << (static_cast<V>(hash_.value(*begin)));
m_logger->stream() << ELPP_LITERAL(")");
m_logger->stream() << ((index_ < keys.size() -1) ? m_containerLogSeperator : ELPP_LITERAL(""));
}
if (begin != end) {
m_logger->stream() << ELPP_LITERAL("...");
}
m_logger->stream() << ELPP_LITERAL("]");
return *this;
}
template <typename K, typename V>
inline MessageBuilder& operator<<(const QMultiHash<K, V>& multiHash_) {
operator << (static_cast<QHash<K, V>>(multiHash_));
return *this;
}
#endif  
#if defined(ELPP_BOOST_LOGGING)
ELPP_ITERATOR_CONTAINER_LOG_TWO_ARG(boost::container::vector)
ELPP_ITERATOR_CONTAINER_LOG_TWO_ARG(boost::container::stable_vector)
ELPP_ITERATOR_CONTAINER_LOG_TWO_ARG(boost::container::list)
ELPP_ITERATOR_CONTAINER_LOG_TWO_ARG(boost::container::deque)
ELPP_ITERATOR_CONTAINER_LOG_FOUR_ARG(boost::container::map)
ELPP_ITERATOR_CONTAINER_LOG_FOUR_ARG(boost::container::flat_map)
ELPP_ITERATOR_CONTAINER_LOG_THREE_ARG(boost::container::set)
ELPP_ITERATOR_CONTAINER_LOG_THREE_ARG(boost::container::flat_set)
#endif  

#define MAKE_CONTAINERELPP_FRIENDLY(ContainerType, SizeMethod, ElementInstance) \
el::base::type::ostream_t& operator<<(el::base::type::ostream_t& ss, const ContainerType& container) {\
const el::base::type::char_t* sep = ELPP->hasFlag(el::LoggingFlag::NewLineForContainer) ? \
ELPP_LITERAL("\n    ") : ELPP_LITERAL(", ");\
ContainerType::const_iterator elem = container.begin();\
ContainerType::const_iterator endElem = container.end();\
std::size_t size_ = container.SizeMethod; \
ss << ELPP_LITERAL("[");\
for (std::size_t i = 0; elem != endElem && i < el::base::consts::kMaxLogPerContainer; ++i, ++elem) { \
ss << ElementInstance;\
ss << ((i < size_ - 1) ? sep : ELPP_LITERAL(""));\
}\
if (elem != endElem) {\
ss << ELPP_LITERAL("...");\
}\
ss << ELPP_LITERAL("]");\
return ss;\
}
#if defined(ELPP_WXWIDGETS_LOGGING)
ELPP_ITERATOR_CONTAINER_LOG_ONE_ARG(wxVector)
#  define ELPP_WX_PTR_ENABLED(ContainerType) MAKE_CONTAINERELPP_FRIENDLY(ContainerType, size(), *(*elem))
#  define ELPP_WX_ENABLED(ContainerType) MAKE_CONTAINERELPP_FRIENDLY(ContainerType, size(), (*elem))
#  define ELPP_WX_HASH_MAP_ENABLED(ContainerType) MAKE_CONTAINERELPP_FRIENDLY(ContainerType, size(), \
ELPP_LITERAL("(") << elem->first << ELPP_LITERAL(", ") << elem->second << ELPP_LITERAL(")")
#else
#  define ELPP_WX_PTR_ENABLED(ContainerType)
#  define ELPP_WX_ENABLED(ContainerType)
#  define ELPP_WX_HASH_MAP_ENABLED(ContainerType)
#endif  
template <class Class>
ELPP_SIMPLE_LOG(const Class&)
#undef ELPP_SIMPLE_LOG
#undef ELPP_ITERATOR_CONTAINER_LOG_ONE_ARG
#undef ELPP_ITERATOR_CONTAINER_LOG_TWO_ARG
#undef ELPP_ITERATOR_CONTAINER_LOG_THREE_ARG
#undef ELPP_ITERATOR_CONTAINER_LOG_FOUR_ARG
#undef ELPP_ITERATOR_CONTAINER_LOG_FIVE_ARG
private:
Logger* m_logger;
const base::type::char_t* m_containerLogSeperator;

template<class Iterator>
MessageBuilder& writeIterator(Iterator begin_, Iterator end_, std::size_t size_) {
m_logger->stream() << ELPP_LITERAL("[");
for (std::size_t i = 0; begin_ != end_ && i < base::consts::kMaxLogPerContainer; ++i, ++begin_) {
operator << (*begin_);
m_logger->stream() << ((i < size_ - 1) ? m_containerLogSeperator : ELPP_LITERAL(""));
}
if (begin_ != end_) {
m_logger->stream() << ELPP_LITERAL("...");
}
m_logger->stream() << ELPP_LITERAL("]");
if (ELPP->hasFlag(LoggingFlag::AutoSpacing)) {
m_logger->stream() << " ";
}
return *this;
}
};
class NullWriter : base::NoCopy {
public:
NullWriter(void) {}

inline NullWriter& operator<<(std::ostream& (*)(std::ostream&)) {
return *this;
}

template <typename T>
inline NullWriter& operator<<(const T&) {
return *this;
}

inline operator bool() {
return true;
}
};
class Writer : base::NoCopy {
public:
Writer(Level level, const char* file, base::type::LineNumber line,
const char* func, base::DispatchAction dispatchAction = base::DispatchAction::NormalLog,
base::type::VerboseLevel verboseLevel = 0) :
m_level(level), m_file(file), m_line(line), m_func(func), m_verboseLevel(verboseLevel),
m_logger(nullptr), m_proceed(false), m_dispatchAction(dispatchAction) {
}

virtual ~Writer(void) {
processDispatch();
}

template <typename T>
inline Writer& operator<<(const T& log) {
#if ELPP_LOGGING_ENABLED
if (m_proceed) {
m_messageBuilder << log;
}
#endif  
return *this;
}

inline Writer& operator<<(std::ostream& (*log)(std::ostream&)) {
#if ELPP_LOGGING_ENABLED
if (m_proceed) {
m_messageBuilder << log;
}
#endif  
return *this;
}

inline operator bool() {
return true;
}

Writer& construct(Logger* logger, bool needLock = true);
Writer& construct(int count, const char* loggerIds, ...);
protected:
Level m_level;
const char* m_file;
const base::type::LineNumber m_line;
const char* m_func;
base::type::VerboseLevel m_verboseLevel;
Logger* m_logger;
bool m_proceed;
base::MessageBuilder m_messageBuilder;
base::DispatchAction m_dispatchAction;
std::vector<std::string> m_loggerIds;
friend class el::Helpers;

void initializeLogger(const std::string& loggerId, bool lookup = true, bool needLock = true);
void processDispatch();
void triggerDispatch(void);
};
class PErrorWriter : public base::Writer {
public:
PErrorWriter(Level level, const char* file, base::type::LineNumber line,
const char* func, base::DispatchAction dispatchAction = base::DispatchAction::NormalLog,
base::type::VerboseLevel verboseLevel = 0) :
base::Writer(level, file, line, func, dispatchAction, verboseLevel) {
}

virtual ~PErrorWriter(void);
};
}  
#if ELPP_VARIADIC_TEMPLATES_SUPPORTED
template <typename T, typename... Args>
void Logger::log_(Level level, int vlevel, const char* s, const T& value, const Args&... args) {
base::MessageBuilder b;
b.initialize(this);
while (*s) {
if (*s == base::consts::kFormatSpecifierChar) {
if (*(s + 1) == base::consts::kFormatSpecifierChar) {
++s;
} else {
if (*(s + 1) == base::consts::kFormatSpecifierCharValue) {
++s;
b << value;
log_(level, vlevel, ++s, args...);
return;
}
}
}
b << *s++;
}
ELPP_INTERNAL_ERROR("Too many arguments provided. Unable to handle. Please provide more format specifiers", false);
}
template <typename T>
void Logger::log_(Level level, int vlevel, const T& log) {
if (level == Level::Verbose) {
if (ELPP->vRegistry()->allowed(vlevel, __FILE__)) {
base::Writer(Level::Verbose, "FILE", 0, "FUNCTION",
base::DispatchAction::NormalLog, vlevel).construct(this, false) << log;
} else {
stream().str(ELPP_LITERAL(""));
}
} else {
base::Writer(level, "FILE", 0, "FUNCTION").construct(this, false) << log;
}
}
template <typename T, typename... Args>
inline void Logger::log(Level level, const char* s, const T& value, const Args&... args) {
base::threading::ScopedLock scopedLock(lock());
log_(level, 0, s, value, args...);
}
template <typename T>
inline void Logger::log(Level level, const T& log) {
base::threading::ScopedLock scopedLock(lock());
log_(level, 0, log);
}
#  if ELPP_VERBOSE_LOG
template <typename T, typename... Args>
inline void Logger::verbose(int vlevel, const char* s, const T& value, const Args&... args) {
base::threading::ScopedLock scopedLock(lock());
log_(el::Level::Verbose, vlevel, s, value, args...);
}
template <typename T>
inline void Logger::verbose(int vlevel, const T& log) {
base::threading::ScopedLock scopedLock(lock());
log_(el::Level::Verbose, vlevel, log);
}
#  else
template <typename T, typename... Args>
inline void Logger::verbose(int, const char*, const T&, const Args&...) {
return;
}
template <typename T>
inline void Logger::verbose(int, const T&) {
return;
}
#  endif  
#  define LOGGER_LEVEL_WRITERS(FUNCTION_NAME, LOG_LEVEL)\
template <typename T, typename... Args>\
inline void Logger::FUNCTION_NAME(const char* s, const T& value, const Args&... args) {\
log(LOG_LEVEL, s, value, args...);\
}\
template <typename T>\
inline void Logger::FUNCTION_NAME(const T& value) {\
log(LOG_LEVEL, value);\
}
#  define LOGGER_LEVEL_WRITERS_DISABLED(FUNCTION_NAME, LOG_LEVEL)\
template <typename T, typename... Args>\
inline void Logger::FUNCTION_NAME(const char*, const T&, const Args&...) {\
return;\
}\
template <typename T>\
inline void Logger::FUNCTION_NAME(const T&) {\
return;\
}

#  if ELPP_INFO_LOG
LOGGER_LEVEL_WRITERS(info, Level::Info)
#  else
LOGGER_LEVEL_WRITERS_DISABLED(info, Level::Info)
#  endif 
#  if ELPP_DEBUG_LOG
LOGGER_LEVEL_WRITERS(debug, Level::Debug)
#  else
LOGGER_LEVEL_WRITERS_DISABLED(debug, Level::Debug)
#  endif 
#  if ELPP_WARNING_LOG
LOGGER_LEVEL_WRITERS(warn, Level::Warning)
#  else
LOGGER_LEVEL_WRITERS_DISABLED(warn, Level::Warning)
#  endif 
#  if ELPP_ERROR_LOG
LOGGER_LEVEL_WRITERS(error, Level::Error)
#  else
LOGGER_LEVEL_WRITERS_DISABLED(error, Level::Error)
#  endif 
#  if ELPP_FATAL_LOG
LOGGER_LEVEL_WRITERS(fatal, Level::Fatal)
#  else
LOGGER_LEVEL_WRITERS_DISABLED(fatal, Level::Fatal)
#  endif 
#  if ELPP_TRACE_LOG
LOGGER_LEVEL_WRITERS(trace, Level::Trace)
#  else
LOGGER_LEVEL_WRITERS_DISABLED(trace, Level::Trace)
#  endif 
#  undef LOGGER_LEVEL_WRITERS
#  undef LOGGER_LEVEL_WRITERS_DISABLED
#endif 
#if ELPP_COMPILER_MSVC
#  define ELPP_VARIADIC_FUNC_MSVC(variadicFunction, variadicArgs) variadicFunction variadicArgs
#  define ELPP_VARIADIC_FUNC_MSVC_RUN(variadicFunction, ...) ELPP_VARIADIC_FUNC_MSVC(variadicFunction, (__VA_ARGS__))
#  define el_getVALength(...) ELPP_VARIADIC_FUNC_MSVC_RUN(el_resolveVALength, 0, ## __VA_ARGS__,\
10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0)
#else
#  if ELPP_COMPILER_CLANG
#    define el_getVALength(...) el_resolveVALength(0, __VA_ARGS__, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0)
#  else
#    define el_getVALength(...) el_resolveVALength(0, ## __VA_ARGS__, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0)
#  endif 
#endif 
#define el_resolveVALength(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, N, ...) N
#define ELPP_WRITE_LOG(writer, level, dispatchAction, ...) \
writer(level, __FILE__, __LINE__, ELPP_FUNC, dispatchAction).construct(el_getVALength(__VA_ARGS__), __VA_ARGS__)
#define ELPP_WRITE_LOG_IF(writer, condition, level, dispatchAction, ...) if (condition) \
writer(level, __FILE__, __LINE__, ELPP_FUNC, dispatchAction).construct(el_getVALength(__VA_ARGS__), __VA_ARGS__)
#define ELPP_WRITE_LOG_EVERY_N(writer, occasion, level, dispatchAction, ...) \
ELPP->validateEveryNCounter(__FILE__, __LINE__, occasion) && \
writer(level, __FILE__, __LINE__, ELPP_FUNC, dispatchAction).construct(el_getVALength(__VA_ARGS__), __VA_ARGS__)
#define ELPP_WRITE_LOG_AFTER_N(writer, n, level, dispatchAction, ...) \
ELPP->validateAfterNCounter(__FILE__, __LINE__, n) && \
writer(level, __FILE__, __LINE__, ELPP_FUNC, dispatchAction).construct(el_getVALength(__VA_ARGS__), __VA_ARGS__)
#define ELPP_WRITE_LOG_N_TIMES(writer, n, level, dispatchAction, ...) \
ELPP->validateNTimesCounter(__FILE__, __LINE__, n) && \
writer(level, __FILE__, __LINE__, ELPP_FUNC, dispatchAction).construct(el_getVALength(__VA_ARGS__), __VA_ARGS__)
#if defined(ELPP_FEATURE_ALL) || defined(ELPP_FEATURE_PERFORMANCE_TRACKING)
class PerformanceTrackingData {
public:
enum class DataType : base::type::EnumType {
Checkpoint = 1, Complete = 2
};
explicit PerformanceTrackingData(DataType dataType) : m_performanceTracker(nullptr),
m_dataType(dataType), m_firstCheckpoint(false), m_file(""), m_line(0), m_func("") {}
inline const std::string* blockName(void) const;
inline const struct timeval* startTime(void) const;
inline const struct timeval* endTime(void) const;
inline const struct timeval* lastCheckpointTime(void) const;
inline const base::PerformanceTracker* performanceTracker(void) const {
return m_performanceTracker;
}
inline PerformanceTrackingData::DataType dataType(void) const {
return m_dataType;
}
inline bool firstCheckpoint(void) const {
return m_firstCheckpoint;
}
inline std::string checkpointId(void) const {
return m_checkpointId;
}
inline const char* file(void) const {
return m_file;
}
inline base::type::LineNumber line(void) const {
return m_line;
}
inline const char* func(void) const {
return m_func;
}
inline const base::type::string_t* formattedTimeTaken() const {
return &m_formattedTimeTaken;
}
inline const std::string& loggerId(void) const;
private:
base::PerformanceTracker* m_performanceTracker;
base::type::string_t m_formattedTimeTaken;
PerformanceTrackingData::DataType m_dataType;
bool m_firstCheckpoint;
std::string m_checkpointId;
const char* m_file;
base::type::LineNumber m_line;
const char* m_func;
inline void init(base::PerformanceTracker* performanceTracker, bool firstCheckpoint = false) {
m_performanceTracker = performanceTracker;
m_firstCheckpoint = firstCheckpoint;
}

friend class el::base::PerformanceTracker;
};
namespace base {
class PerformanceTracker : public base::threading::ThreadSafe, public Loggable {
public:
PerformanceTracker(const std::string& blockName,
base::TimestampUnit timestampUnit = base::TimestampUnit::Millisecond,
const std::string& loggerId = std::string(el::base::consts::kPerformanceLoggerId),
bool scopedLog = true, Level level = base::consts::kPerformanceTrackerDefaultLevel);
PerformanceTracker(const PerformanceTracker& t) :
m_blockName(t.m_blockName), m_timestampUnit(t.m_timestampUnit), m_loggerId(t.m_loggerId), m_scopedLog(t.m_scopedLog),
m_level(t.m_level), m_hasChecked(t.m_hasChecked), m_lastCheckpointId(t.m_lastCheckpointId), m_enabled(t.m_enabled),
m_startTime(t.m_startTime), m_endTime(t.m_endTime), m_lastCheckpointTime(t.m_lastCheckpointTime) {
}
virtual ~PerformanceTracker(void);
void checkpoint(const std::string& id = std::string(), const char* file = __FILE__,
base::type::LineNumber line = __LINE__,
const char* func = "");
inline Level level(void) const {
return m_level;
}
private:
std::string m_blockName;
base::TimestampUnit m_timestampUnit;
std::string m_loggerId;
bool m_scopedLog;
Level m_level;
bool m_hasChecked;
std::string m_lastCheckpointId;
bool m_enabled;
struct timeval m_startTime, m_endTime, m_lastCheckpointTime;

PerformanceTracker(void);

friend class el::PerformanceTrackingData;
friend class base::DefaultPerformanceTrackingCallback;

const inline base::type::string_t getFormattedTimeTaken() const {
return getFormattedTimeTaken(m_startTime);
}

const base::type::string_t getFormattedTimeTaken(struct timeval startTime) const;

virtual inline void log(el::base::type::ostream_t& os) const {
os << getFormattedTimeTaken();
}
};
class DefaultPerformanceTrackingCallback : public PerformanceTrackingCallback {
protected:
void handle(const PerformanceTrackingData* data) {
m_data = data;
base::type::stringstream_t ss;
if (m_data->dataType() == PerformanceTrackingData::DataType::Complete) {
ss << ELPP_LITERAL("Executed [") << m_data->blockName()->c_str() << ELPP_LITERAL("] in [") <<
*m_data->formattedTimeTaken() << ELPP_LITERAL("]");
} else {
ss << ELPP_LITERAL("Performance checkpoint");
if (!m_data->checkpointId().empty()) {
ss << ELPP_LITERAL(" [") << m_data->checkpointId().c_str() << ELPP_LITERAL("]");
}
ss << ELPP_LITERAL(" for block [") << m_data->blockName()->c_str() << ELPP_LITERAL("] : [") <<
*m_data->performanceTracker();
if (!ELPP->hasFlag(LoggingFlag::DisablePerformanceTrackingCheckpointComparison)
&& m_data->performanceTracker()->m_hasChecked) {
ss << ELPP_LITERAL(" ([") << *m_data->formattedTimeTaken() << ELPP_LITERAL("] from ");
if (m_data->performanceTracker()->m_lastCheckpointId.empty()) {
ss << ELPP_LITERAL("last checkpoint");
} else {
ss << ELPP_LITERAL("checkpoint '") << m_data->performanceTracker()->m_lastCheckpointId.c_str() << ELPP_LITERAL("'");
}
ss << ELPP_LITERAL(")]");
} else {
ss << ELPP_LITERAL("]");
}
}
el::base::Writer(m_data->performanceTracker()->level(), m_data->file(), m_data->line(), m_data->func()).construct(1,
m_data->loggerId().c_str()) << ss.str();
}
private:
const PerformanceTrackingData* m_data;
};
}  
inline const std::string* PerformanceTrackingData::blockName() const {
return const_cast<const std::string*>(&m_performanceTracker->m_blockName);
}
inline const struct timeval* PerformanceTrackingData::startTime() const {
return const_cast<const struct timeval*>(&m_performanceTracker->m_startTime);
}
inline const struct timeval* PerformanceTrackingData::endTime() const {
return const_cast<const struct timeval*>(&m_performanceTracker->m_endTime);
}
inline const struct timeval* PerformanceTrackingData::lastCheckpointTime() const {
return const_cast<const struct timeval*>(&m_performanceTracker->m_lastCheckpointTime);
}
inline const std::string& PerformanceTrackingData::loggerId(void) const {
return m_performanceTracker->m_loggerId;
}
#endif 
namespace base {
namespace debug {
#if defined(ELPP_FEATURE_ALL) || defined(ELPP_FEATURE_CRASH_LOG)
class StackTrace : base::NoCopy {
public:
static const unsigned int kMaxStack = 64;
static const unsigned int kStackStart = 2;  
class StackTraceEntry {
public:
StackTraceEntry(std::size_t index, const char* loc, const char* demang, const char* hex, const char* addr);
StackTraceEntry(std::size_t index, char* loc) :
m_index(index),
m_location(loc) {
}
std::size_t m_index;
std::string m_location;
std::string m_demangled;
std::string m_hex;
std::string m_addr;
friend std::ostream& operator<<(std::ostream& ss, const StackTraceEntry& si);

private:
StackTraceEntry(void);
};

StackTrace(void) {
generateNew();
}

virtual ~StackTrace(void) {
}

inline std::vector<StackTraceEntry>& getLatestStack(void) {
return m_stack;
}

friend std::ostream& operator<<(std::ostream& os, const StackTrace& st);

private:
std::vector<StackTraceEntry> m_stack;

void generateNew(void);
};
class CrashHandler : base::NoCopy {
public:
typedef void (*Handler)(int);

explicit CrashHandler(bool useDefault);
explicit CrashHandler(const Handler& cHandler) {
setHandler(cHandler);
}
void setHandler(const Handler& cHandler);

private:
Handler m_handler;
};
#else
class CrashHandler {
public:
explicit CrashHandler(bool) {}
};
#endif 
}  
}  
extern base::debug::CrashHandler elCrashHandler;
#define MAKE_LOGGABLE(ClassType, ClassInstance, OutputStreamInstance) \
el::base::type::ostream_t& operator<<(el::base::type::ostream_t& OutputStreamInstance, const ClassType& ClassInstance)
class SysLogInitializer {
public:
SysLogInitializer(const char* processIdent, int options = 0, int facility = 0) {
#if defined(ELPP_SYSLOG)
openlog(processIdent, options, facility);
#else
ELPP_UNUSED(processIdent);
ELPP_UNUSED(options);
ELPP_UNUSED(facility);
#endif  
}
virtual ~SysLogInitializer(void) {
#if defined(ELPP_SYSLOG)
closelog();
#endif  
}
};
#define ELPP_INITIALIZE_SYSLOG(id, opt, fac) el::SysLogInitializer elSyslogInit(id, opt, fac)
class Helpers : base::StaticClass {
public:
static inline void setStorage(base::type::StoragePointer storage) {
ELPP = storage;
}
static inline base::type::StoragePointer storage() {
return ELPP;
}
static inline void setArgs(int argc, char** argv) {
ELPP->setApplicationArguments(argc, argv);
}
static inline void setArgs(int argc, const char** argv) {
ELPP->setApplicationArguments(argc, const_cast<char**>(argv));
}
static inline void setThreadName(const std::string& name) {
ELPP->setThreadName(name);
}
static inline std::string getThreadName() {
return ELPP->getThreadName(base::threading::getCurrentThreadId());
}
#if defined(ELPP_FEATURE_ALL) || defined(ELPP_FEATURE_CRASH_LOG)
static inline void setCrashHandler(const el::base::debug::CrashHandler::Handler& crashHandler) {
el::elCrashHandler.setHandler(crashHandler);
}
static void crashAbort(int sig, const char* sourceFile = "", unsigned int long line = 0);
static void logCrashReason(int sig, bool stackTraceIfAvailable = false,
Level level = Level::Fatal, const char* logger = base::consts::kDefaultLoggerId);
#endif 
static inline void installPreRollOutCallback(const PreRollOutCallback& callback) {
ELPP->setPreRollOutCallback(callback);
}
static inline void uninstallPreRollOutCallback(void) {
ELPP->unsetPreRollOutCallback();
}
template <typename T>
static inline bool installLogDispatchCallback(const std::string& id) {
return ELPP->installLogDispatchCallback<T>(id);
}
template <typename T>
static inline void uninstallLogDispatchCallback(const std::string& id) {
ELPP->uninstallLogDispatchCallback<T>(id);
}
template <typename T>
static inline T* logDispatchCallback(const std::string& id) {
return ELPP->logDispatchCallback<T>(id);
}
#if defined(ELPP_FEATURE_ALL) || defined(ELPP_FEATURE_PERFORMANCE_TRACKING)
template <typename T>
static inline bool installPerformanceTrackingCallback(const std::string& id) {
return ELPP->installPerformanceTrackingCallback<T>(id);
}
template <typename T>
static inline void uninstallPerformanceTrackingCallback(const std::string& id) {
ELPP->uninstallPerformanceTrackingCallback<T>(id);
}
template <typename T>
static inline T* performanceTrackingCallback(const std::string& id) {
return ELPP->performanceTrackingCallback<T>(id);
}
#endif 
template <typename T>
static std::string convertTemplateToStdString(const T& templ) {
el::Logger* logger =
ELPP->registeredLoggers()->get(el::base::consts::kDefaultLoggerId);
if (logger == nullptr) {
return std::string();
}
base::MessageBuilder b;
b.initialize(logger);
logger->acquireLock();
b << templ;
#if defined(ELPP_UNICODE)
std::string s = std::string(logger->stream().str().begin(), logger->stream().str().end());
#else
std::string s = logger->stream().str();
#endif  
logger->stream().str(ELPP_LITERAL(""));
logger->releaseLock();
return s;
}
static inline const el::base::utils::CommandLineArgs* commandLineArgs(void) {
return ELPP->commandLineArgs();
}
static inline void installCustomFormatSpecifier(const CustomFormatSpecifier& customFormatSpecifier) {
ELPP->installCustomFormatSpecifier(customFormatSpecifier);
}
static inline bool uninstallCustomFormatSpecifier(const char* formatSpecifier) {
return ELPP->uninstallCustomFormatSpecifier(formatSpecifier);
}
static inline bool hasCustomFormatSpecifier(const char* formatSpecifier) {
return ELPP->hasCustomFormatSpecifier(formatSpecifier);
}
static inline void validateFileRolling(Logger* logger, Level level) {
if (logger == nullptr) return;
logger->m_typedConfigurations->validateFileRolling(level, ELPP->preRollOutCallback());
}
};
class Loggers : base::StaticClass {
public:
static Logger* getLogger(const std::string& identity, bool registerIfNotAvailable = true);
static void setDefaultLogBuilder(el::LogBuilderPtr& logBuilderPtr);
template <typename T>
static inline bool installLoggerRegistrationCallback(const std::string& id) {
return ELPP->registeredLoggers()->installLoggerRegistrationCallback<T>(id);
}
template <typename T>
static inline void uninstallLoggerRegistrationCallback(const std::string& id) {
ELPP->registeredLoggers()->uninstallLoggerRegistrationCallback<T>(id);
}
template <typename T>
static inline T* loggerRegistrationCallback(const std::string& id) {
return ELPP->registeredLoggers()->loggerRegistrationCallback<T>(id);
}
static bool unregisterLogger(const std::string& identity);
static bool hasLogger(const std::string& identity);
static Logger* reconfigureLogger(Logger* logger, const Configurations& configurations);
static Logger* reconfigureLogger(const std::string& identity, const Configurations& configurations);
static Logger* reconfigureLogger(const std::string& identity, ConfigurationType configurationType,
const std::string& value);
static void reconfigureAllLoggers(const Configurations& configurations);
static inline void reconfigureAllLoggers(ConfigurationType configurationType, const std::string& value) {
reconfigureAllLoggers(Level::Global, configurationType, value);
}
static void reconfigureAllLoggers(Level level, ConfigurationType configurationType,
const std::string& value);
static void setDefaultConfigurations(const Configurations& configurations,
bool reconfigureExistingLoggers = false);
static const Configurations* defaultConfigurations(void);
static const base::LogStreamsReferenceMap* logStreamsReference(void);
static base::TypedConfigurations defaultTypedConfigurations(void);
static std::vector<std::string>* populateAllLoggerIds(std::vector<std::string>* targetList);
static void configureFromGlobal(const char* globalConfigurationFilePath);
static bool configureFromArg(const char* argKey);
static void flushAll(void);
static inline void addFlag(LoggingFlag flag) {
ELPP->addFlag(flag);
}
static inline void removeFlag(LoggingFlag flag) {
ELPP->removeFlag(flag);
}
static inline bool hasFlag(LoggingFlag flag) {
return ELPP->hasFlag(flag);
}
class ScopedAddFlag {
public:
ScopedAddFlag(LoggingFlag flag) : m_flag(flag) {
Loggers::addFlag(m_flag);
}
~ScopedAddFlag(void) {
Loggers::removeFlag(m_flag);
}
private:
LoggingFlag m_flag;
};
class ScopedRemoveFlag {
public:
ScopedRemoveFlag(LoggingFlag flag) : m_flag(flag) {
Loggers::removeFlag(m_flag);
}
~ScopedRemoveFlag(void) {
Loggers::addFlag(m_flag);
}
private:
LoggingFlag m_flag;
};
static void setLoggingLevel(Level level) {
ELPP->setLoggingLevel(level);
}
static void setVerboseLevel(base::type::VerboseLevel level);
static base::type::VerboseLevel verboseLevel(void);
static void setVModules(const char* modules);
static void clearVModules(void);
};
class VersionInfo : base::StaticClass {
public:
static const std::string version(void);

static const std::string releaseDate(void);
};
}  
#undef VLOG_IS_ON
#define VLOG_IS_ON(verboseLevel) (ELPP->vRegistry()->allowed(verboseLevel, __FILE__))
#undef TIMED_BLOCK
#undef TIMED_SCOPE
#undef TIMED_SCOPE_IF
#undef TIMED_FUNC
#undef TIMED_FUNC_IF
#undef ELPP_MIN_UNIT
#if defined(ELPP_PERFORMANCE_MICROSECONDS)
#  define ELPP_MIN_UNIT el::base::TimestampUnit::Microsecond
#else
#  define ELPP_MIN_UNIT el::base::TimestampUnit::Millisecond
#endif  
#define TIMED_SCOPE_IF(obj, blockname, condition) el::base::type::PerformanceTrackerPtr obj( condition ? \
new el::base::PerformanceTracker(blockname, ELPP_MIN_UNIT) : nullptr )
#define TIMED_SCOPE(obj, blockname) TIMED_SCOPE_IF(obj, blockname, true)
#define TIMED_BLOCK(obj, blockName) for (struct { int i; el::base::type::PerformanceTrackerPtr timer; } obj = { 0, \
el::base::type::PerformanceTrackerPtr(new el::base::PerformanceTracker(blockName, ELPP_MIN_UNIT)) }; obj.i < 1; ++obj.i)
#define TIMED_FUNC_IF(obj,condition) TIMED_SCOPE_IF(obj, ELPP_FUNC, condition)
#define TIMED_FUNC(obj) TIMED_SCOPE(obj, ELPP_FUNC)
#undef PERFORMANCE_CHECKPOINT
#undef PERFORMANCE_CHECKPOINT_WITH_ID
#define PERFORMANCE_CHECKPOINT(obj) obj->checkpoint(std::string(), __FILE__, __LINE__, ELPP_FUNC)
#define PERFORMANCE_CHECKPOINT_WITH_ID(obj, id) obj->checkpoint(id, __FILE__, __LINE__, ELPP_FUNC)
#undef ELPP_COUNTER
#undef ELPP_COUNTER_POS
#define ELPP_COUNTER (ELPP->hitCounters()->getCounter(__FILE__, __LINE__))
#define ELPP_COUNTER_POS (ELPP_COUNTER == nullptr ? -1 : ELPP_COUNTER->hitCounts())
#undef INFO
#undef WARNING
#undef DEBUG
#undef ERROR
#undef FATAL
#undef TRACE
#undef VERBOSE
#undef CINFO
#undef CWARNING
#undef CDEBUG
#undef CFATAL
#undef CERROR
#undef CTRACE
#undef CVERBOSE
#undef CINFO_IF
#undef CWARNING_IF
#undef CDEBUG_IF
#undef CERROR_IF
#undef CFATAL_IF
#undef CTRACE_IF
#undef CVERBOSE_IF
#undef CINFO_EVERY_N
#undef CWARNING_EVERY_N
#undef CDEBUG_EVERY_N
#undef CERROR_EVERY_N
#undef CFATAL_EVERY_N
#undef CTRACE_EVERY_N
#undef CVERBOSE_EVERY_N
#undef CINFO_AFTER_N
#undef CWARNING_AFTER_N
#undef CDEBUG_AFTER_N
#undef CERROR_AFTER_N
#undef CFATAL_AFTER_N
#undef CTRACE_AFTER_N
#undef CVERBOSE_AFTER_N
#undef CINFO_N_TIMES
#undef CWARNING_N_TIMES
#undef CDEBUG_N_TIMES
#undef CERROR_N_TIMES
#undef CFATAL_N_TIMES
#undef CTRACE_N_TIMES
#undef CVERBOSE_N_TIMES
#if ELPP_INFO_LOG
#  define CINFO(writer, dispatchAction, ...) ELPP_WRITE_LOG(writer, el::Level::Info, dispatchAction, __VA_ARGS__)
#else
#  define CINFO(writer, dispatchAction, ...) el::base::NullWriter()
#endif  
#if ELPP_WARNING_LOG
#  define CWARNING(writer, dispatchAction, ...) ELPP_WRITE_LOG(writer, el::Level::Warning, dispatchAction, __VA_ARGS__)
#else
#  define CWARNING(writer, dispatchAction, ...) el::base::NullWriter()
#endif  
#if ELPP_DEBUG_LOG
#  define CDEBUG(writer, dispatchAction, ...) ELPP_WRITE_LOG(writer, el::Level::Debug, dispatchAction, __VA_ARGS__)
#else
#  define CDEBUG(writer, dispatchAction, ...) el::base::NullWriter()
#endif  
#if ELPP_ERROR_LOG
#  define CERROR(writer, dispatchAction, ...) ELPP_WRITE_LOG(writer, el::Level::Error, dispatchAction, __VA_ARGS__)
#else
#  define CERROR(writer, dispatchAction, ...) el::base::NullWriter()
#endif  
#if ELPP_FATAL_LOG
#  define CFATAL(writer, dispatchAction, ...) ELPP_WRITE_LOG(writer, el::Level::Fatal, dispatchAction, __VA_ARGS__)
#else
#  define CFATAL(writer, dispatchAction, ...) el::base::NullWriter()
#endif  
#if ELPP_TRACE_LOG
#  define CTRACE(writer, dispatchAction, ...) ELPP_WRITE_LOG(writer, el::Level::Trace, dispatchAction, __VA_ARGS__)
#else
#  define CTRACE(writer, dispatchAction, ...) el::base::NullWriter()
#endif  
#if ELPP_VERBOSE_LOG
#  define CVERBOSE(writer, vlevel, dispatchAction, ...) if (VLOG_IS_ON(vlevel)) writer(\
el::Level::Verbose, __FILE__, __LINE__, ELPP_FUNC, dispatchAction, vlevel).construct(el_getVALength(__VA_ARGS__), __VA_ARGS__)
#else
#  define CVERBOSE(writer, vlevel, dispatchAction, ...) el::base::NullWriter()
#endif  
#if ELPP_INFO_LOG
#  define CINFO_IF(writer, condition_, dispatchAction, ...) \
ELPP_WRITE_LOG_IF(writer, (condition_), el::Level::Info, dispatchAction, __VA_ARGS__)
#else
#  define CINFO_IF(writer, condition_, dispatchAction, ...) el::base::NullWriter()
#endif  
#if ELPP_WARNING_LOG
#  define CWARNING_IF(writer, condition_, dispatchAction, ...)\
ELPP_WRITE_LOG_IF(writer, (condition_), el::Level::Warning, dispatchAction, __VA_ARGS__)
#else
#  define CWARNING_IF(writer, condition_, dispatchAction, ...) el::base::NullWriter()
#endif  
#if ELPP_DEBUG_LOG
#  define CDEBUG_IF(writer, condition_, dispatchAction, ...)\
ELPP_WRITE_LOG_IF(writer, (condition_), el::Level::Debug, dispatchAction, __VA_ARGS__)
#else
#  define CDEBUG_IF(writer, condition_, dispatchAction, ...) el::base::NullWriter()
#endif  
#if ELPP_ERROR_LOG
#  define CERROR_IF(writer, condition_, dispatchAction, ...)\
ELPP_WRITE_LOG_IF(writer, (condition_), el::Level::Error, dispatchAction, __VA_ARGS__)
#else
#  define CERROR_IF(writer, condition_, dispatchAction, ...) el::base::NullWriter()
#endif  
#if ELPP_FATAL_LOG
#  define CFATAL_IF(writer, condition_, dispatchAction, ...)\
ELPP_WRITE_LOG_IF(writer, (condition_), el::Level::Fatal, dispatchAction, __VA_ARGS__)
#else
#  define CFATAL_IF(writer, condition_, dispatchAction, ...) el::base::NullWriter()
#endif  
#if ELPP_TRACE_LOG
#  define CTRACE_IF(writer, condition_, dispatchAction, ...)\
ELPP_WRITE_LOG_IF(writer, (condition_), el::Level::Trace, dispatchAction, __VA_ARGS__)
#else
#  define CTRACE_IF(writer, condition_, dispatchAction, ...) el::base::NullWriter()
#endif  
#if ELPP_VERBOSE_LOG
#  define CVERBOSE_IF(writer, condition_, vlevel, dispatchAction, ...) if (VLOG_IS_ON(vlevel) && (condition_)) writer( \
el::Level::Verbose, __FILE__, __LINE__, ELPP_FUNC, dispatchAction, vlevel).construct(el_getVALength(__VA_ARGS__), __VA_ARGS__)
#else
#  define CVERBOSE_IF(writer, condition_, vlevel, dispatchAction, ...) el::base::NullWriter()
#endif  
#if ELPP_INFO_LOG
#  define CINFO_EVERY_N(writer, occasion, dispatchAction, ...)\
ELPP_WRITE_LOG_EVERY_N(writer, occasion, el::Level::Info, dispatchAction, __VA_ARGS__)
#else
#  define CINFO_EVERY_N(writer, occasion, dispatchAction, ...) el::base::NullWriter()
#endif  
#if ELPP_WARNING_LOG
#  define CWARNING_EVERY_N(writer, occasion, dispatchAction, ...)\
ELPP_WRITE_LOG_EVERY_N(writer, occasion, el::Level::Warning, dispatchAction, __VA_ARGS__)
#else
#  define CWARNING_EVERY_N(writer, occasion, dispatchAction, ...) el::base::NullWriter()
#endif  
#if ELPP_DEBUG_LOG
#  define CDEBUG_EVERY_N(writer, occasion, dispatchAction, ...)\
ELPP_WRITE_LOG_EVERY_N(writer, occasion, el::Level::Debug, dispatchAction, __VA_ARGS__)
#else
#  define CDEBUG_EVERY_N(writer, occasion, dispatchAction, ...) el::base::NullWriter()
#endif  
#if ELPP_ERROR_LOG
#  define CERROR_EVERY_N(writer, occasion, dispatchAction, ...)\
ELPP_WRITE_LOG_EVERY_N(writer, occasion, el::Level::Error, dispatchAction, __VA_ARGS__)
#else
#  define CERROR_EVERY_N(writer, occasion, dispatchAction, ...) el::base::NullWriter()
#endif  
#if ELPP_FATAL_LOG
#  define CFATAL_EVERY_N(writer, occasion, dispatchAction, ...)\
ELPP_WRITE_LOG_EVERY_N(writer, occasion, el::Level::Fatal, dispatchAction, __VA_ARGS__)
#else
#  define CFATAL_EVERY_N(writer, occasion, dispatchAction, ...) el::base::NullWriter()
#endif  
#if ELPP_TRACE_LOG
#  define CTRACE_EVERY_N(writer, occasion, dispatchAction, ...)\
ELPP_WRITE_LOG_EVERY_N(writer, occasion, el::Level::Trace, dispatchAction, __VA_ARGS__)
#else
#  define CTRACE_EVERY_N(writer, occasion, dispatchAction, ...) el::base::NullWriter()
#endif  
#if ELPP_VERBOSE_LOG
#  define CVERBOSE_EVERY_N(writer, occasion, vlevel, dispatchAction, ...)\
CVERBOSE_IF(writer, ELPP->validateEveryNCounter(__FILE__, __LINE__, occasion), vlevel, dispatchAction, __VA_ARGS__)
#else
#  define CVERBOSE_EVERY_N(writer, occasion, vlevel, dispatchAction, ...) el::base::NullWriter()
#endif  
#if ELPP_INFO_LOG
#  define CINFO_AFTER_N(writer, n, dispatchAction, ...)\
ELPP_WRITE_LOG_AFTER_N(writer, n, el::Level::Info, dispatchAction, __VA_ARGS__)
#else
#  define CINFO_AFTER_N(writer, n, dispatchAction, ...) el::base::NullWriter()
#endif  
#if ELPP_WARNING_LOG
#  define CWARNING_AFTER_N(writer, n, dispatchAction, ...)\
ELPP_WRITE_LOG_AFTER_N(writer, n, el::Level::Warning, dispatchAction, __VA_ARGS__)
#else
#  define CWARNING_AFTER_N(writer, n, dispatchAction, ...) el::base::NullWriter()
#endif  
#if ELPP_DEBUG_LOG
#  define CDEBUG_AFTER_N(writer, n, dispatchAction, ...)\
ELPP_WRITE_LOG_AFTER_N(writer, n, el::Level::Debug, dispatchAction, __VA_ARGS__)
#else
#  define CDEBUG_AFTER_N(writer, n, dispatchAction, ...) el::base::NullWriter()
#endif  
#if ELPP_ERROR_LOG
#  define CERROR_AFTER_N(writer, n, dispatchAction, ...)\
ELPP_WRITE_LOG_AFTER_N(writer, n, el::Level::Error, dispatchAction, __VA_ARGS__)
#else
#  define CERROR_AFTER_N(writer, n, dispatchAction, ...) el::base::NullWriter()
#endif  
#if ELPP_FATAL_LOG
#  define CFATAL_AFTER_N(writer, n, dispatchAction, ...)\
ELPP_WRITE_LOG_AFTER_N(writer, n, el::Level::Fatal, dispatchAction, __VA_ARGS__)
#else
#  define CFATAL_AFTER_N(writer, n, dispatchAction, ...) el::base::NullWriter()
#endif  
#if ELPP_TRACE_LOG
#  define CTRACE_AFTER_N(writer, n, dispatchAction, ...)\
ELPP_WRITE_LOG_AFTER_N(writer, n, el::Level::Trace, dispatchAction, __VA_ARGS__)
#else
#  define CTRACE_AFTER_N(writer, n, dispatchAction, ...) el::base::NullWriter()
#endif  
#if ELPP_VERBOSE_LOG
#  define CVERBOSE_AFTER_N(writer, n, vlevel, dispatchAction, ...)\
CVERBOSE_IF(writer, ELPP->validateAfterNCounter(__FILE__, __LINE__, n), vlevel, dispatchAction, __VA_ARGS__)
#else
#  define CVERBOSE_AFTER_N(writer, n, vlevel, dispatchAction, ...) el::base::NullWriter()
#endif  
#if ELPP_INFO_LOG
#  define CINFO_N_TIMES(writer, n, dispatchAction, ...)\
ELPP_WRITE_LOG_N_TIMES(writer, n, el::Level::Info, dispatchAction, __VA_ARGS__)
#else
#  define CINFO_N_TIMES(writer, n, dispatchAction, ...) el::base::NullWriter()
#endif  
#if ELPP_WARNING_LOG
#  define CWARNING_N_TIMES(writer, n, dispatchAction, ...)\
ELPP_WRITE_LOG_N_TIMES(writer, n, el::Level::Warning, dispatchAction, __VA_ARGS__)
#else
#  define CWARNING_N_TIMES(writer, n, dispatchAction, ...) el::base::NullWriter()
#endif  
#if ELPP_DEBUG_LOG
#  define CDEBUG_N_TIMES(writer, n, dispatchAction, ...)\
ELPP_WRITE_LOG_N_TIMES(writer, n, el::Level::Debug, dispatchAction, __VA_ARGS__)
#else
#  define CDEBUG_N_TIMES(writer, n, dispatchAction, ...) el::base::NullWriter()
#endif  
#if ELPP_ERROR_LOG
#  define CERROR_N_TIMES(writer, n, dispatchAction, ...)\
ELPP_WRITE_LOG_N_TIMES(writer, n, el::Level::Error, dispatchAction, __VA_ARGS__)
#else
#  define CERROR_N_TIMES(writer, n, dispatchAction, ...) el::base::NullWriter()
#endif  
#if ELPP_FATAL_LOG
#  define CFATAL_N_TIMES(writer, n, dispatchAction, ...)\
ELPP_WRITE_LOG_N_TIMES(writer, n, el::Level::Fatal, dispatchAction, __VA_ARGS__)
#else
#  define CFATAL_N_TIMES(writer, n, dispatchAction, ...) el::base::NullWriter()
#endif  
#if ELPP_TRACE_LOG
#  define CTRACE_N_TIMES(writer, n, dispatchAction, ...)\
ELPP_WRITE_LOG_N_TIMES(writer, n, el::Level::Trace, dispatchAction, __VA_ARGS__)
#else
#  define CTRACE_N_TIMES(writer, n, dispatchAction, ...) el::base::NullWriter()
#endif  
#if ELPP_VERBOSE_LOG
#  define CVERBOSE_N_TIMES(writer, n, vlevel, dispatchAction, ...)\
CVERBOSE_IF(writer, ELPP->validateNTimesCounter(__FILE__, __LINE__, n), vlevel, dispatchAction, __VA_ARGS__)
#else
#  define CVERBOSE_N_TIMES(writer, n, vlevel, dispatchAction, ...) el::base::NullWriter()
#endif  
#undef CLOG
#undef CLOG_VERBOSE
#undef CVLOG
#undef CLOG_IF
#undef CLOG_VERBOSE_IF
#undef CVLOG_IF
#undef CLOG_EVERY_N
#undef CVLOG_EVERY_N
#undef CLOG_AFTER_N
#undef CVLOG_AFTER_N
#undef CLOG_N_TIMES
#undef CVLOG_N_TIMES
#define CLOG(LEVEL, ...)\
C##LEVEL(el::base::Writer, el::base::DispatchAction::NormalLog, __VA_ARGS__)
#define CVLOG(vlevel, ...) CVERBOSE(el::base::Writer, vlevel, el::base::DispatchAction::NormalLog, __VA_ARGS__)
#define CLOG_IF(condition, LEVEL, ...)\
C##LEVEL##_IF(el::base::Writer, condition, el::base::DispatchAction::NormalLog, __VA_ARGS__)
#define CVLOG_IF(condition, vlevel, ...)\
CVERBOSE_IF(el::base::Writer, condition, vlevel, el::base::DispatchAction::NormalLog, __VA_ARGS__)
#define CLOG_EVERY_N(n, LEVEL, ...)\
C##LEVEL##_EVERY_N(el::base::Writer, n, el::base::DispatchAction::NormalLog, __VA_ARGS__)
#define CVLOG_EVERY_N(n, vlevel, ...)\
CVERBOSE_EVERY_N(el::base::Writer, n, vlevel, el::base::DispatchAction::NormalLog, __VA_ARGS__)
#define CLOG_AFTER_N(n, LEVEL, ...)\
C##LEVEL##_AFTER_N(el::base::Writer, n, el::base::DispatchAction::NormalLog, __VA_ARGS__)
#define CVLOG_AFTER_N(n, vlevel, ...)\
CVERBOSE_AFTER_N(el::base::Writer, n, vlevel, el::base::DispatchAction::NormalLog, __VA_ARGS__)
#define CLOG_N_TIMES(n, LEVEL, ...)\
C##LEVEL##_N_TIMES(el::base::Writer, n, el::base::DispatchAction::NormalLog, __VA_ARGS__)
#define CVLOG_N_TIMES(n, vlevel, ...)\
CVERBOSE_N_TIMES(el::base::Writer, n, vlevel, el::base::DispatchAction::NormalLog, __VA_ARGS__)
#undef LOG
#undef VLOG
#undef LOG_IF
#undef VLOG_IF
#undef LOG_EVERY_N
#undef VLOG_EVERY_N
#undef LOG_AFTER_N
#undef VLOG_AFTER_N
#undef LOG_N_TIMES
#undef VLOG_N_TIMES
#undef ELPP_CURR_FILE_LOGGER_ID
#if defined(ELPP_DEFAULT_LOGGER)
#  define ELPP_CURR_FILE_LOGGER_ID ELPP_DEFAULT_LOGGER
#else
#  define ELPP_CURR_FILE_LOGGER_ID el::base::consts::kDefaultLoggerId
#endif
#undef ELPP_TRACE
#define ELPP_TRACE CLOG(TRACE, ELPP_CURR_FILE_LOGGER_ID)
#define LOG(LEVEL) CLOG(LEVEL, ELPP_CURR_FILE_LOGGER_ID)
#define VLOG(vlevel) CVLOG(vlevel, ELPP_CURR_FILE_LOGGER_ID)
#define LOG_IF(condition, LEVEL) CLOG_IF(condition, LEVEL, ELPP_CURR_FILE_LOGGER_ID)
#define VLOG_IF(condition, vlevel) CVLOG_IF(condition, vlevel, ELPP_CURR_FILE_LOGGER_ID)
#define LOG_EVERY_N(n, LEVEL) CLOG_EVERY_N(n, LEVEL, ELPP_CURR_FILE_LOGGER_ID)
#define VLOG_EVERY_N(n, vlevel) CVLOG_EVERY_N(n, vlevel, ELPP_CURR_FILE_LOGGER_ID)
#define LOG_AFTER_N(n, LEVEL) CLOG_AFTER_N(n, LEVEL, ELPP_CURR_FILE_LOGGER_ID)
#define VLOG_AFTER_N(n, vlevel) CVLOG_AFTER_N(n, vlevel, ELPP_CURR_FILE_LOGGER_ID)
#define LOG_N_TIMES(n, LEVEL) CLOG_N_TIMES(n, LEVEL, ELPP_CURR_FILE_LOGGER_ID)
#define VLOG_N_TIMES(n, vlevel) CVLOG_N_TIMES(n, vlevel, ELPP_CURR_FILE_LOGGER_ID)
#undef CPLOG
#undef CPLOG_IF
#undef PLOG
#undef PLOG_IF
#undef DCPLOG
#undef DCPLOG_IF
#undef DPLOG
#undef DPLOG_IF
#define CPLOG(LEVEL, ...)\
C##LEVEL(el::base::PErrorWriter, el::base::DispatchAction::NormalLog, __VA_ARGS__)
#define CPLOG_IF(condition, LEVEL, ...)\
C##LEVEL##_IF(el::base::PErrorWriter, condition, el::base::DispatchAction::NormalLog, __VA_ARGS__)
#define DCPLOG(LEVEL, ...)\
if (ELPP_DEBUG_LOG) C##LEVEL(el::base::PErrorWriter, el::base::DispatchAction::NormalLog, __VA_ARGS__)
#define DCPLOG_IF(condition, LEVEL, ...)\
C##LEVEL##_IF(el::base::PErrorWriter, (ELPP_DEBUG_LOG) && (condition), el::base::DispatchAction::NormalLog, __VA_ARGS__)
#define PLOG(LEVEL) CPLOG(LEVEL, ELPP_CURR_FILE_LOGGER_ID)
#define PLOG_IF(condition, LEVEL) CPLOG_IF(condition, LEVEL, ELPP_CURR_FILE_LOGGER_ID)
#define DPLOG(LEVEL) DCPLOG(LEVEL, ELPP_CURR_FILE_LOGGER_ID)
#define DPLOG_IF(condition, LEVEL) DCPLOG_IF(condition, LEVEL, ELPP_CURR_FILE_LOGGER_ID)
#undef CSYSLOG
#undef CSYSLOG_IF
#undef CSYSLOG_EVERY_N
#undef CSYSLOG_AFTER_N
#undef CSYSLOG_N_TIMES
#undef SYSLOG
#undef SYSLOG_IF
#undef SYSLOG_EVERY_N
#undef SYSLOG_AFTER_N
#undef SYSLOG_N_TIMES
#undef DCSYSLOG
#undef DCSYSLOG_IF
#undef DCSYSLOG_EVERY_N
#undef DCSYSLOG_AFTER_N
#undef DCSYSLOG_N_TIMES
#undef DSYSLOG
#undef DSYSLOG_IF
#undef DSYSLOG_EVERY_N
#undef DSYSLOG_AFTER_N
#undef DSYSLOG_N_TIMES
#if defined(ELPP_SYSLOG)
#  define CSYSLOG(LEVEL, ...)\
C##LEVEL(el::base::Writer, el::base::DispatchAction::SysLog, __VA_ARGS__)
#  define CSYSLOG_IF(condition, LEVEL, ...)\
C##LEVEL##_IF(el::base::Writer, condition, el::base::DispatchAction::SysLog, __VA_ARGS__)
#  define CSYSLOG_EVERY_N(n, LEVEL, ...) C##LEVEL##_EVERY_N(el::base::Writer, n, el::base::DispatchAction::SysLog, __VA_ARGS__)
#  define CSYSLOG_AFTER_N(n, LEVEL, ...) C##LEVEL##_AFTER_N(el::base::Writer, n, el::base::DispatchAction::SysLog, __VA_ARGS__)
#  define CSYSLOG_N_TIMES(n, LEVEL, ...) C##LEVEL##_N_TIMES(el::base::Writer, n, el::base::DispatchAction::SysLog, __VA_ARGS__)
#  define SYSLOG(LEVEL) CSYSLOG(LEVEL, el::base::consts::kSysLogLoggerId)
#  define SYSLOG_IF(condition, LEVEL) CSYSLOG_IF(condition, LEVEL, el::base::consts::kSysLogLoggerId)
#  define SYSLOG_EVERY_N(n, LEVEL) CSYSLOG_EVERY_N(n, LEVEL, el::base::consts::kSysLogLoggerId)
#  define SYSLOG_AFTER_N(n, LEVEL) CSYSLOG_AFTER_N(n, LEVEL, el::base::consts::kSysLogLoggerId)
#  define SYSLOG_N_TIMES(n, LEVEL) CSYSLOG_N_TIMES(n, LEVEL, el::base::consts::kSysLogLoggerId)
#  define DCSYSLOG(LEVEL, ...) if (ELPP_DEBUG_LOG) C##LEVEL(el::base::Writer, el::base::DispatchAction::SysLog, __VA_ARGS__)
#  define DCSYSLOG_IF(condition, LEVEL, ...)\
C##LEVEL##_IF(el::base::Writer, (ELPP_DEBUG_LOG) && (condition), el::base::DispatchAction::SysLog, __VA_ARGS__)
#  define DCSYSLOG_EVERY_N(n, LEVEL, ...)\
if (ELPP_DEBUG_LOG) C##LEVEL##_EVERY_N(el::base::Writer, n, el::base::DispatchAction::SysLog, __VA_ARGS__)
#  define DCSYSLOG_AFTER_N(n, LEVEL, ...)\
if (ELPP_DEBUG_LOG) C##LEVEL##_AFTER_N(el::base::Writer, n, el::base::DispatchAction::SysLog, __VA_ARGS__)
#  define DCSYSLOG_N_TIMES(n, LEVEL, ...)\
if (ELPP_DEBUG_LOG) C##LEVEL##_EVERY_N(el::base::Writer, n, el::base::DispatchAction::SysLog, __VA_ARGS__)
#  define DSYSLOG(LEVEL) DCSYSLOG(LEVEL, el::base::consts::kSysLogLoggerId)
#  define DSYSLOG_IF(condition, LEVEL) DCSYSLOG_IF(condition, LEVEL, el::base::consts::kSysLogLoggerId)
#  define DSYSLOG_EVERY_N(n, LEVEL) DCSYSLOG_EVERY_N(n, LEVEL, el::base::consts::kSysLogLoggerId)
#  define DSYSLOG_AFTER_N(n, LEVEL) DCSYSLOG_AFTER_N(n, LEVEL, el::base::consts::kSysLogLoggerId)
#  define DSYSLOG_N_TIMES(n, LEVEL) DCSYSLOG_N_TIMES(n, LEVEL, el::base::consts::kSysLogLoggerId)
#else
#  define CSYSLOG(LEVEL, ...) el::base::NullWriter()
#  define CSYSLOG_IF(condition, LEVEL, ...) el::base::NullWriter()
#  define CSYSLOG_EVERY_N(n, LEVEL, ...) el::base::NullWriter()
#  define CSYSLOG_AFTER_N(n, LEVEL, ...) el::base::NullWriter()
#  define CSYSLOG_N_TIMES(n, LEVEL, ...) el::base::NullWriter()
#  define SYSLOG(LEVEL) el::base::NullWriter()
#  define SYSLOG_IF(condition, LEVEL) el::base::NullWriter()
#  define SYSLOG_EVERY_N(n, LEVEL) el::base::NullWriter()
#  define SYSLOG_AFTER_N(n, LEVEL) el::base::NullWriter()
#  define SYSLOG_N_TIMES(n, LEVEL) el::base::NullWriter()
#  define DCSYSLOG(LEVEL, ...) el::base::NullWriter()
#  define DCSYSLOG_IF(condition, LEVEL, ...) el::base::NullWriter()
#  define DCSYSLOG_EVERY_N(n, LEVEL, ...) el::base::NullWriter()
#  define DCSYSLOG_AFTER_N(n, LEVEL, ...) el::base::NullWriter()
#  define DCSYSLOG_N_TIMES(n, LEVEL, ...) el::base::NullWriter()
#  define DSYSLOG(LEVEL) el::base::NullWriter()
#  define DSYSLOG_IF(condition, LEVEL) el::base::NullWriter()
#  define DSYSLOG_EVERY_N(n, LEVEL) el::base::NullWriter()
#  define DSYSLOG_AFTER_N(n, LEVEL) el::base::NullWriter()
#  define DSYSLOG_N_TIMES(n, LEVEL) el::base::NullWriter()
#endif  
#undef DCLOG
#undef DCVLOG
#undef DCLOG_IF
#undef DCVLOG_IF
#undef DCLOG_EVERY_N
#undef DCVLOG_EVERY_N
#undef DCLOG_AFTER_N
#undef DCVLOG_AFTER_N
#undef DCLOG_N_TIMES
#undef DCVLOG_N_TIMES
#define DCLOG(LEVEL, ...) if (ELPP_DEBUG_LOG) CLOG(LEVEL, __VA_ARGS__)
#define DCLOG_VERBOSE(vlevel, ...) if (ELPP_DEBUG_LOG) CLOG_VERBOSE(vlevel, __VA_ARGS__)
#define DCVLOG(vlevel, ...) if (ELPP_DEBUG_LOG) CVLOG(vlevel, __VA_ARGS__)
#define DCLOG_IF(condition, LEVEL, ...) if (ELPP_DEBUG_LOG) CLOG_IF(condition, LEVEL, __VA_ARGS__)
#define DCVLOG_IF(condition, vlevel, ...) if (ELPP_DEBUG_LOG) CVLOG_IF(condition, vlevel, __VA_ARGS__)
#define DCLOG_EVERY_N(n, LEVEL, ...) if (ELPP_DEBUG_LOG) CLOG_EVERY_N(n, LEVEL, __VA_ARGS__)
#define DCVLOG_EVERY_N(n, vlevel, ...) if (ELPP_DEBUG_LOG) CVLOG_EVERY_N(n, vlevel, __VA_ARGS__)
#define DCLOG_AFTER_N(n, LEVEL, ...) if (ELPP_DEBUG_LOG) CLOG_AFTER_N(n, LEVEL, __VA_ARGS__)
#define DCVLOG_AFTER_N(n, vlevel, ...) if (ELPP_DEBUG_LOG) CVLOG_AFTER_N(n, vlevel, __VA_ARGS__)
#define DCLOG_N_TIMES(n, LEVEL, ...) if (ELPP_DEBUG_LOG) CLOG_N_TIMES(n, LEVEL, __VA_ARGS__)
#define DCVLOG_N_TIMES(n, vlevel, ...) if (ELPP_DEBUG_LOG) CVLOG_N_TIMES(n, vlevel, __VA_ARGS__)
#if !defined(ELPP_NO_DEBUG_MACROS)
#undef DLOG
#undef DVLOG
#undef DLOG_IF
#undef DVLOG_IF
#undef DLOG_EVERY_N
#undef DVLOG_EVERY_N
#undef DLOG_AFTER_N
#undef DVLOG_AFTER_N
#undef DLOG_N_TIMES
#undef DVLOG_N_TIMES
#define DLOG(LEVEL) DCLOG(LEVEL, ELPP_CURR_FILE_LOGGER_ID)
#define DVLOG(vlevel) DCVLOG(vlevel, ELPP_CURR_FILE_LOGGER_ID)
#define DLOG_IF(condition, LEVEL) DCLOG_IF(condition, LEVEL, ELPP_CURR_FILE_LOGGER_ID)
#define DVLOG_IF(condition, vlevel) DCVLOG_IF(condition, vlevel, ELPP_CURR_FILE_LOGGER_ID)
#define DLOG_EVERY_N(n, LEVEL) DCLOG_EVERY_N(n, LEVEL, ELPP_CURR_FILE_LOGGER_ID)
#define DVLOG_EVERY_N(n, vlevel) DCVLOG_EVERY_N(n, vlevel, ELPP_CURR_FILE_LOGGER_ID)
#define DLOG_AFTER_N(n, LEVEL) DCLOG_AFTER_N(n, LEVEL, ELPP_CURR_FILE_LOGGER_ID)
#define DVLOG_AFTER_N(n, vlevel) DCVLOG_AFTER_N(n, vlevel, ELPP_CURR_FILE_LOGGER_ID)
#define DLOG_N_TIMES(n, LEVEL) DCLOG_N_TIMES(n, LEVEL, ELPP_CURR_FILE_LOGGER_ID)
#define DVLOG_N_TIMES(n, vlevel) DCVLOG_N_TIMES(n, vlevel, ELPP_CURR_FILE_LOGGER_ID)
#endif 
#if !defined(ELPP_NO_CHECK_MACROS)
#undef CCHECK
#undef CPCHECK
#undef CCHECK_EQ
#undef CCHECK_NE
#undef CCHECK_LT
#undef CCHECK_GT
#undef CCHECK_LE
#undef CCHECK_GE
#undef CCHECK_BOUNDS
#undef CCHECK_NOTNULL
#undef CCHECK_STRCASEEQ
#undef CCHECK_STRCASENE
#undef CHECK
#undef PCHECK
#undef CHECK_EQ
#undef CHECK_NE
#undef CHECK_LT
#undef CHECK_GT
#undef CHECK_LE
#undef CHECK_GE
#undef CHECK_BOUNDS
#undef CHECK_NOTNULL
#undef CHECK_STRCASEEQ
#undef CHECK_STRCASENE
#define CCHECK(condition, ...) CLOG_IF(!(condition), FATAL, __VA_ARGS__) << "Check failed: [" << #condition << "] "
#define CPCHECK(condition, ...) CPLOG_IF(!(condition), FATAL, __VA_ARGS__) << "Check failed: [" << #condition << "] "
#define CHECK(condition) CCHECK(condition, ELPP_CURR_FILE_LOGGER_ID)
#define PCHECK(condition) CPCHECK(condition, ELPP_CURR_FILE_LOGGER_ID)
#define CCHECK_EQ(a, b, ...) CCHECK(a == b, __VA_ARGS__)
#define CCHECK_NE(a, b, ...) CCHECK(a != b, __VA_ARGS__)
#define CCHECK_LT(a, b, ...) CCHECK(a < b, __VA_ARGS__)
#define CCHECK_GT(a, b, ...) CCHECK(a > b, __VA_ARGS__)
#define CCHECK_LE(a, b, ...) CCHECK(a <= b, __VA_ARGS__)
#define CCHECK_GE(a, b, ...) CCHECK(a >= b, __VA_ARGS__)
#define CCHECK_BOUNDS(val, min, max, ...) CCHECK(val >= min && val <= max, __VA_ARGS__)
#define CHECK_EQ(a, b) CCHECK_EQ(a, b, ELPP_CURR_FILE_LOGGER_ID)
#define CHECK_NE(a, b) CCHECK_NE(a, b, ELPP_CURR_FILE_LOGGER_ID)
#define CHECK_LT(a, b) CCHECK_LT(a, b, ELPP_CURR_FILE_LOGGER_ID)
#define CHECK_GT(a, b) CCHECK_GT(a, b, ELPP_CURR_FILE_LOGGER_ID)
#define CHECK_LE(a, b) CCHECK_LE(a, b, ELPP_CURR_FILE_LOGGER_ID)
#define CHECK_GE(a, b) CCHECK_GE(a, b, ELPP_CURR_FILE_LOGGER_ID)
#define CHECK_BOUNDS(val, min, max) CCHECK_BOUNDS(val, min, max, ELPP_CURR_FILE_LOGGER_ID)
#define CCHECK_NOTNULL(ptr, ...) CCHECK((ptr) != nullptr, __VA_ARGS__)
#define CCHECK_STREQ(str1, str2, ...) CLOG_IF(!el::base::utils::Str::cStringEq(str1, str2), FATAL, __VA_ARGS__) \
<< "Check failed: [" << #str1 << " == " << #str2 << "] "
#define CCHECK_STRNE(str1, str2, ...) CLOG_IF(el::base::utils::Str::cStringEq(str1, str2), FATAL, __VA_ARGS__) \
<< "Check failed: [" << #str1 << " != " << #str2 << "] "
#define CCHECK_STRCASEEQ(str1, str2, ...) CLOG_IF(!el::base::utils::Str::cStringCaseEq(str1, str2), FATAL, __VA_ARGS__) \
<< "Check failed: [" << #str1 << " == " << #str2 << "] "
#define CCHECK_STRCASENE(str1, str2, ...) CLOG_IF(el::base::utils::Str::cStringCaseEq(str1, str2), FATAL, __VA_ARGS__) \
<< "Check failed: [" << #str1 << " != " << #str2 << "] "
#define CHECK_NOTNULL(ptr) CCHECK_NOTNULL((ptr), ELPP_CURR_FILE_LOGGER_ID)
#define CHECK_STREQ(str1, str2) CCHECK_STREQ(str1, str2, ELPP_CURR_FILE_LOGGER_ID)
#define CHECK_STRNE(str1, str2) CCHECK_STRNE(str1, str2, ELPP_CURR_FILE_LOGGER_ID)
#define CHECK_STRCASEEQ(str1, str2) CCHECK_STRCASEEQ(str1, str2, ELPP_CURR_FILE_LOGGER_ID)
#define CHECK_STRCASENE(str1, str2) CCHECK_STRCASENE(str1, str2, ELPP_CURR_FILE_LOGGER_ID)
#undef DCCHECK
#undef DCCHECK_EQ
#undef DCCHECK_NE
#undef DCCHECK_LT
#undef DCCHECK_GT
#undef DCCHECK_LE
#undef DCCHECK_GE
#undef DCCHECK_BOUNDS
#undef DCCHECK_NOTNULL
#undef DCCHECK_STRCASEEQ
#undef DCCHECK_STRCASENE
#undef DCPCHECK
#undef DCHECK
#undef DCHECK_EQ
#undef DCHECK_NE
#undef DCHECK_LT
#undef DCHECK_GT
#undef DCHECK_LE
#undef DCHECK_GE
#undef DCHECK_BOUNDS_
#undef DCHECK_NOTNULL
#undef DCHECK_STRCASEEQ
#undef DCHECK_STRCASENE
#undef DPCHECK
#define DCCHECK(condition, ...) if (ELPP_DEBUG_LOG) CCHECK(condition, __VA_ARGS__)
#define DCCHECK_EQ(a, b, ...) if (ELPP_DEBUG_LOG) CCHECK_EQ(a, b, __VA_ARGS__)
#define DCCHECK_NE(a, b, ...) if (ELPP_DEBUG_LOG) CCHECK_NE(a, b, __VA_ARGS__)
#define DCCHECK_LT(a, b, ...) if (ELPP_DEBUG_LOG) CCHECK_LT(a, b, __VA_ARGS__)
#define DCCHECK_GT(a, b, ...) if (ELPP_DEBUG_LOG) CCHECK_GT(a, b, __VA_ARGS__)
#define DCCHECK_LE(a, b, ...) if (ELPP_DEBUG_LOG) CCHECK_LE(a, b, __VA_ARGS__)
#define DCCHECK_GE(a, b, ...) if (ELPP_DEBUG_LOG) CCHECK_GE(a, b, __VA_ARGS__)
#define DCCHECK_BOUNDS(val, min, max, ...) if (ELPP_DEBUG_LOG) CCHECK_BOUNDS(val, min, max, __VA_ARGS__)
#define DCCHECK_NOTNULL(ptr, ...) if (ELPP_DEBUG_LOG) CCHECK_NOTNULL((ptr), __VA_ARGS__)
#define DCCHECK_STREQ(str1, str2, ...) if (ELPP_DEBUG_LOG) CCHECK_STREQ(str1, str2, __VA_ARGS__)
#define DCCHECK_STRNE(str1, str2, ...) if (ELPP_DEBUG_LOG) CCHECK_STRNE(str1, str2, __VA_ARGS__)
#define DCCHECK_STRCASEEQ(str1, str2, ...) if (ELPP_DEBUG_LOG) CCHECK_STRCASEEQ(str1, str2, __VA_ARGS__)
#define DCCHECK_STRCASENE(str1, str2, ...) if (ELPP_DEBUG_LOG) CCHECK_STRCASENE(str1, str2, __VA_ARGS__)
#define DCPCHECK(condition, ...) if (ELPP_DEBUG_LOG) CPCHECK(condition, __VA_ARGS__)
#define DCHECK(condition) DCCHECK(condition, ELPP_CURR_FILE_LOGGER_ID)
#define DCHECK_EQ(a, b) DCCHECK_EQ(a, b, ELPP_CURR_FILE_LOGGER_ID)
#define DCHECK_NE(a, b) DCCHECK_NE(a, b, ELPP_CURR_FILE_LOGGER_ID)
#define DCHECK_LT(a, b) DCCHECK_LT(a, b, ELPP_CURR_FILE_LOGGER_ID)
#define DCHECK_GT(a, b) DCCHECK_GT(a, b, ELPP_CURR_FILE_LOGGER_ID)
#define DCHECK_LE(a, b) DCCHECK_LE(a, b, ELPP_CURR_FILE_LOGGER_ID)
#define DCHECK_GE(a, b) DCCHECK_GE(a, b, ELPP_CURR_FILE_LOGGER_ID)
#define DCHECK_BOUNDS(val, min, max) DCCHECK_BOUNDS(val, min, max, ELPP_CURR_FILE_LOGGER_ID)
#define DCHECK_NOTNULL(ptr) DCCHECK_NOTNULL((ptr), ELPP_CURR_FILE_LOGGER_ID)
#define DCHECK_STREQ(str1, str2) DCCHECK_STREQ(str1, str2, ELPP_CURR_FILE_LOGGER_ID)
#define DCHECK_STRNE(str1, str2) DCCHECK_STRNE(str1, str2, ELPP_CURR_FILE_LOGGER_ID)
#define DCHECK_STRCASEEQ(str1, str2) DCCHECK_STRCASEEQ(str1, str2, ELPP_CURR_FILE_LOGGER_ID)
#define DCHECK_STRCASENE(str1, str2) DCCHECK_STRCASENE(str1, str2, ELPP_CURR_FILE_LOGGER_ID)
#define DPCHECK(condition) DCPCHECK(condition, ELPP_CURR_FILE_LOGGER_ID)
#endif 
#if defined(ELPP_DISABLE_DEFAULT_CRASH_HANDLING)
#  define ELPP_USE_DEF_CRASH_HANDLER false
#else
#  define ELPP_USE_DEF_CRASH_HANDLER true
#endif  
#define ELPP_CRASH_HANDLER_INIT
#define ELPP_INIT_EASYLOGGINGPP(val) \
namespace el { \
namespace base { \
el::base::type::StoragePointer elStorage(val); \
} \
el::base::debug::CrashHandler elCrashHandler(ELPP_USE_DEF_CRASH_HANDLER); \
}

#if ELPP_ASYNC_LOGGING
#  define INITIALIZE_EASYLOGGINGPP ELPP_INIT_EASYLOGGINGPP(new el::base::Storage(el::LogBuilderPtr(new el::base::DefaultLogBuilder()),\
new el::base::AsyncDispatchWorker()))
#else
#  define INITIALIZE_EASYLOGGINGPP ELPP_INIT_EASYLOGGINGPP(new el::base::Storage(el::LogBuilderPtr(new el::base::DefaultLogBuilder())))
#endif  
#define INITIALIZE_NULL_EASYLOGGINGPP \
namespace el {\
namespace base {\
el::base::type::StoragePointer elStorage;\
}\
el::base::debug::CrashHandler elCrashHandler(ELPP_USE_DEF_CRASH_HANDLER);\
}
#define SHARE_EASYLOGGINGPP(initializedStorage)\
namespace el {\
namespace base {\
el::base::type::StoragePointer elStorage(initializedStorage);\
}\
el::base::debug::CrashHandler elCrashHandler(ELPP_USE_DEF_CRASH_HANDLER);\
}

#if defined(ELPP_UNICODE)
#  define START_EASYLOGGINGPP(argc, argv) el::Helpers::setArgs(argc, argv); std::locale::global(std::locale(""))
#else
#  define START_EASYLOGGINGPP(argc, argv) el::Helpers::setArgs(argc, argv)
#endif  
#endif 
