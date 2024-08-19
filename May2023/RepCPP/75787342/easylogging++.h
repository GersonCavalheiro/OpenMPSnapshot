#ifndef EASYLOGGINGPP_H
#define EASYLOGGINGPP_H
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
#  elif(ELPP_GCC_VERSION >= 40801)
#    define ELPP_CXX11 1
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
#  define ELPP_CLANG_VERSION (__clang_major__ * 10000 \
+ __clang_minor__ * 100 \
+ __clang_patchlevel__)
#  if (ELPP_CLANG_VERSION >= 30300)
#    define ELPP_CXX11 1
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
#if defined(ELPP_STACKTRACE_ON_CRASH)
#  if (ELPP_COMPILER_GCC && !ELPP_MINGW)
#    define ELPP_STACKTRACE 1
#  else
#      if ELPP_COMPILER_MSVC
#         pragma message("Stack trace not available for this compiler")
#      else
#         warning "Stack trace not available for this compiler";
#      endif  
#  endif  
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
#if ((!ELPP_MINGW && !ELPP_COMPILER_CLANG && defined(ELPP_CXX11)) || defined(ELPP_FORCE_USE_STD_THREAD))
#  define ELPP_USE_STD_THREADING 1
#else
#  define ELPP_USE_STD_THREADING 0
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
#define ELPP_LOGGING_ENABLED (!defined(ELPP_DISABLE_LOGS))
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
#   error "Easylogging++ 9.0+ is only compatible with C++0x (or higher) compliant compiler"
#endif  
#if defined(ELPP_CXX11)
#    define HAS_ENUM_CLASS
#    define HAS_RANGE_FOR
#endif
#if defined(HAS_ENUM_CLASS)
#define ENUMCLASS enum class
#else
#define ENUMCLASS enum
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
#include <array>
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
typedef unsigned short EnumType;
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
ENUMCLASS Level : base::type::EnumType {
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
static const char* convertToString(Level level) {
if (level == Level::Global) return "GLOBAL";
if (level == Level::Debug) return "DEBUG";
if (level == Level::Info) return "INFO";
if (level == Level::Warning) return "WARNING";
if (level == Level::Error) return "ERROR";
if (level == Level::Fatal) return "FATAL";
if (level == Level::Verbose) return "VERBOSE";
if (level == Level::Trace) return "TRACE";
return "UNKNOWN";
}
static Level convertFromString(const char* levelStr) {
if ((strcmp(levelStr, "GLOBAL") == 0) || (strcmp(levelStr, "global") == 0))
return Level::Global;
if ((strcmp(levelStr, "DEBUG") == 0) || (strcmp(levelStr, "debug") == 0))
return Level::Debug;
if ((strcmp(levelStr, "INFO") == 0) || (strcmp(levelStr, "info") == 0))
return Level::Info;
if ((strcmp(levelStr, "WARNING") == 0) || (strcmp(levelStr, "warning") == 0))
return Level::Warning;
if ((strcmp(levelStr, "ERROR") == 0) || (strcmp(levelStr, "error") == 0))
return Level::Error;
if ((strcmp(levelStr, "FATAL") == 0) || (strcmp(levelStr, "fatal") == 0))
return Level::Fatal;
if ((strcmp(levelStr, "VERBOSE") == 0) || (strcmp(levelStr, "verbose") == 0))
return Level::Verbose;
if ((strcmp(levelStr, "TRACE") == 0) || (strcmp(levelStr, "trace") == 0))
return Level::Trace;
return Level::Unknown;
}
static inline void forEachLevel(base::type::EnumType* startIndex, const std::function<bool(void)>& fn) {
base::type::EnumType lIndexMax = LevelHelper::kMaxValid;
do {
if (fn()) {
break;
}
*startIndex = static_cast<base::type::EnumType>(*startIndex << 1);
} while (*startIndex <= lIndexMax);
}
};
ENUMCLASS ConfigurationType : base::type::EnumType {
Enabled = 1,
ToFile = 2,
ToStandardOutput = 4,
Format = 8,
Filename = 16,
MillisecondsWidth = 32,
PerformanceTracking = 64,
MaxLogFileSize = 128,
LogFlushThreshold = 256,
Unknown_Type = 1010
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
static const char* convertToString(ConfigurationType configurationType) {
if (configurationType == ConfigurationType::Enabled) return "ENABLED";
if (configurationType == ConfigurationType::Filename) return "FILENAME";
if (configurationType == ConfigurationType::Format) return "FORMAT";
if (configurationType == ConfigurationType::ToFile) return "TO_FILE";
if (configurationType == ConfigurationType::ToStandardOutput) return "TO_STANDARD_OUTPUT";
if (configurationType == ConfigurationType::MillisecondsWidth) return "MILLISECONDS_WIDTH";
if (configurationType == ConfigurationType::PerformanceTracking) return "PERFORMANCE_TRACKING";
if (configurationType == ConfigurationType::MaxLogFileSize) return "MAX_LOG_FILE_SIZE";
if (configurationType == ConfigurationType::LogFlushThreshold) return "LOG_FLUSH_THRESHOLD";
return "UNKNOWN_TYPE";
}
static ConfigurationType convertFromString(const char* configStr) {
if ((strcmp(configStr, "ENABLED") == 0) || (strcmp(configStr, "enabled") == 0))
return ConfigurationType::Enabled;
if ((strcmp(configStr, "TO_FILE") == 0) || (strcmp(configStr, "to_file") == 0))
return ConfigurationType::ToFile;
if ((strcmp(configStr, "TO_STANDARD_OUTPUT") == 0) || (strcmp(configStr, "to_standard_output") == 0))
return ConfigurationType::ToStandardOutput;
if ((strcmp(configStr, "FORMAT") == 0) || (strcmp(configStr, "format") == 0))
return ConfigurationType::Format;
if ((strcmp(configStr, "FILENAME") == 0) || (strcmp(configStr, "filename") == 0))
return ConfigurationType::Filename;
if ((strcmp(configStr, "MILLISECONDS_WIDTH") == 0) || (strcmp(configStr, "milliseconds_width") == 0))
return ConfigurationType::MillisecondsWidth;
if ((strcmp(configStr, "PERFORMANCE_TRACKING") == 0) || (strcmp(configStr, "performance_tracking") == 0))
return ConfigurationType::PerformanceTracking;
if ((strcmp(configStr, "MAX_LOG_FILE_SIZE") == 0) || (strcmp(configStr, "max_log_file_size") == 0))
return ConfigurationType::MaxLogFileSize;
if ((strcmp(configStr, "LOG_FLUSH_THRESHOLD") == 0) || (strcmp(configStr, "log_flush_threshold") == 0))
return ConfigurationType::LogFlushThreshold;
return ConfigurationType::Unknown_Type;
}
static inline void forEachConfigType(base::type::EnumType* startIndex, const std::function<bool(void)>& fn) {
base::type::EnumType cIndexMax = ConfigurationTypeHelper::kMaxValid;
do {
if (fn()) {
break;
}
*startIndex = static_cast<base::type::EnumType>(*startIndex << 1);
} while (*startIndex <= cIndexMax);
}
};
ENUMCLASS LoggingFlag : base::type::EnumType {
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
static const base::type::char_t* kInfoLevelLogValue     =   ELPP_LITERAL("INFO ");
static const base::type::char_t* kDebugLevelLogValue    =   ELPP_LITERAL("DEBUG");
static const base::type::char_t* kWarningLevelLogValue  =   ELPP_LITERAL("WARN ");
static const base::type::char_t* kErrorLevelLogValue    =   ELPP_LITERAL("ERROR");
static const base::type::char_t* kFatalLevelLogValue    =   ELPP_LITERAL("FATAL");
static const base::type::char_t* kVerboseLevelLogValue  =   ELPP_LITERAL("VER");
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
static const char* kDefaultLoggerId                        =      "default";
static const char* kPerformanceLoggerId                    =      "performance";
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
static const unsigned int  kDefaultMillisecondsWidth       =      3;
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
static const el::Level kPerformanceTrackerDefaultLevel     =      el::Level::Info;
const struct {
double value;
const base::type::char_t* unit;
} kTimeFormats[] = {
{ 1000.0f, ELPP_LITERAL("mis") },
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
}  
}  
typedef std::function<void(const char*, std::size_t)> PreRollOutCallback;
namespace base {
static inline void defaultPreRollOutCallback(const char*, std::size_t) {}
ENUMCLASS TimestampUnit : base::type::EnumType {
Microsecond = 0, Millisecond = 1, Second = 2, Minute = 3, Hour = 4, Day = 5
};
ENUMCLASS FormatFlags : base::type::EnumType {
DateTime = 1<<1, LoggerId = 1<<2, File = 1<<3, Line = 1<<4, Location = 1<<5, Function = 1<<6,
User = 1<<7, Host = 1<<8, LogMessage = 1<<9, VerboseLevel = 1<<10, AppName = 1<<11, ThreadId = 1<<12,
Level = 1<<13, FileBase = 1<<14, LevelShort = 1<<15
};
class MillisecondsWidth {
public:
MillisecondsWidth(void) {
init(base::consts::kDefaultMillisecondsWidth);
}
explicit MillisecondsWidth(int width) {
init(width);
}
bool operator==(const MillisecondsWidth& msWidth) {
return m_width == msWidth.m_width && m_offset == msWidth.m_offset;
}
int m_width;
unsigned int m_offset;
private:
void init(int width) {
if (width < 1 || width > 6) {
width = base::consts::kDefaultMillisecondsWidth;
}
m_width = width;
switch (m_width) {
case 3:
m_offset = 1000;
break;
case 4:
m_offset = 100;
break;
case 5:
m_offset = 10;
break;
case 6:
m_offset = 1;
break;
default:
m_offset = 1000;
break;
}
}
};
namespace utils {
template <typename T>
static inline
typename std::enable_if<std::is_pointer<T*>::value, void>::type
safeDelete(T*& pointer) {
if (pointer == nullptr)
return;
delete pointer;
pointer = nullptr;
}
static inline const char* charPtrVal(const char* pointer) {
return pointer == nullptr ? base::consts::kNullPointer : pointer;
}
static inline void abort(int status, const std::string& reason = std::string()) {
ELPP_UNUSED(status);
ELPP_UNUSED(reason);
#if defined(ELPP_COMPILER_MSVC) && defined(_M_IX86) && defined(_DEBUG)
_asm int 3
#else
::abort();
#endif  
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
pthread_mutex_init(&m_underlyingMutex, nullptr);
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
return (TryEnterCriticalSection(&m_underlyingMutex) != 0);
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
static inline std::string getCurrentThreadId(void) {
std::stringstream ss;
#      if (ELPP_OS_WINDOWS)
ss << GetCurrentThreadId();
#      endif  
return ss.str();
}
static inline void msleep(int) {
}
typedef base::threading::internal::Mutex Mutex;
typedef base::threading::internal::ScopedLock<base::threading::Mutex> ScopedLock;
#  else
static inline std::string getCurrentThreadId(void) {
std::stringstream ss;
ss << std::this_thread::get_id();
return ss.str();
}
static inline void msleep(int ms) {
#      if ELPP_ASYNC_LOGGING
#         if defined(ELPP_NO_SLEEP_FOR)
usleep(ms * 1000);
#         else
std::this_thread::sleep_for(std::chrono::milliseconds(ms));
#         endif  
#      else
ELPP_UNUSED(ms);
#      endif  
}
typedef std::mutex Mutex;
typedef std::lock_guard<std::mutex> ScopedLock;
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
static inline std::string getCurrentThreadId(void) {
return std::string();
}
static inline void msleep(int) {
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
}  
namespace utils {
class File : base::StaticClass {
public:
static base::type::fstream_t* newFileStream(const std::string& filename) {
base::type::fstream_t *fs = new base::type::fstream_t(filename.c_str(),
base::type::fstream_t::out
#if !defined(ELPP_FRESH_LOG_FILE)
| base::type::fstream_t::app
#endif
);
#if defined(ELPP_UNICODE)
std::locale elppUnicodeLocale("");
#  if ELPP_OS_WINDOWS
std::locale elppUnicodeLocaleWindows(elppUnicodeLocale, new std::codecvt_utf8_utf16<wchar_t>);
elppUnicodeLocale = elppUnicodeLocaleWindows;
#  endif 
fs->imbue(elppUnicodeLocale);
#endif  
if (fs->is_open()) {
fs->flush();
} else {
base::utils::safeDelete(fs);
ELPP_INTERNAL_ERROR("Bad file [" << filename << "]", true);
}
return fs;
}

static std::size_t getSizeOfFile(base::type::fstream_t* fs) {
if (fs == nullptr) {
return 0;
}
std::streampos currPos = fs->tellg();
fs->seekg(0, fs->end);
std::size_t size = static_cast<std::size_t>(fs->tellg());
fs->seekg(currPos);
return size;
}

static inline bool pathExists(const char* path, bool considerFile = false) {
if (path == nullptr) {
return false;
}
#if ELPP_OS_UNIX
ELPP_UNUSED(considerFile);
struct stat st;
return (stat(path, &st) == 0);
#elif ELPP_OS_WINDOWS
DWORD fileType = GetFileAttributesA(path);
if (fileType == INVALID_FILE_ATTRIBUTES) {
return false;
}
return considerFile ? true : ((fileType & FILE_ATTRIBUTE_DIRECTORY) == 0 ? false : true);
#endif  
}

static bool createPath(const std::string& path) {
if (path.empty()) {
return false;
}
if (base::utils::File::pathExists(path.c_str())) {
return true;
}
int status = -1;

char* currPath = const_cast<char*>(path.c_str());
std::string builtPath = std::string();
#if ELPP_OS_UNIX
if (path[0] == '/') {
builtPath = "/";
}
currPath = STRTOK(currPath, base::consts::kFilePathSeperator, 0);
#elif ELPP_OS_WINDOWS
char* nextTok_ = nullptr;
currPath = STRTOK(currPath, base::consts::kFilePathSeperator, &nextTok_);
ELPP_UNUSED(nextTok_);
#endif  
while (currPath != nullptr) {
builtPath.append(currPath);
builtPath.append(base::consts::kFilePathSeperator);
#if ELPP_OS_UNIX
status = mkdir(builtPath.c_str(), ELPP_LOG_PERMS);
currPath = STRTOK(nullptr, base::consts::kFilePathSeperator, 0);
#elif ELPP_OS_WINDOWS
status = _mkdir(builtPath.c_str());
currPath = STRTOK(nullptr, base::consts::kFilePathSeperator, &nextTok_);
#endif  
}
if (status == -1) {
ELPP_INTERNAL_ERROR("Error while creating path [" << path << "]", true);
return false;
}
return true;
}
static std::string extractPathFromFilename(const std::string& fullPath,
const char* seperator = base::consts::kFilePathSeperator) {
if ((fullPath == "") || (fullPath.find(seperator) == std::string::npos)) {
return fullPath;
}
std::size_t lastSlashAt = fullPath.find_last_of(seperator);
if (lastSlashAt == 0) {
return std::string(seperator);
}
return fullPath.substr(0, lastSlashAt + 1);
}
static void buildStrippedFilename(const char* filename, char buff[],
std::size_t limit = base::consts::kSourceFilenameMaxLength) {
std::size_t sizeOfFilename = strlen(filename);
if (sizeOfFilename >= limit) {
filename += (sizeOfFilename - limit);
if (filename[0] != '.' && filename[1] != '.') {  
filename += 3;  
STRCAT(buff, "..", limit);
}
}
STRCAT(buff, filename, limit);
}
static void buildBaseFilename(const std::string& fullPath, char buff[],
std::size_t limit = base::consts::kSourceFilenameMaxLength,
const char* seperator = base::consts::kFilePathSeperator) {
const char *filename = fullPath.c_str();
std::size_t lastSlashAt = fullPath.find_last_of(seperator);
filename += lastSlashAt ? lastSlashAt+1 : 0;
std::size_t sizeOfFilename = strlen(filename);
if (sizeOfFilename >= limit) {
filename += (sizeOfFilename - limit);
if (filename[0] != '.' && filename[1] != '.') {  
filename += 3;  
STRCAT(buff, "..", limit);
}
}
STRCAT(buff, filename, limit);
}
};
class Str : base::StaticClass {
public:
static inline bool isDigit(char c) {
return c >= '0' && c <= '9';
}

static bool wildCardMatch(const char* str, const char* pattern) {
while (*pattern) {
switch (*pattern) {
case '?':
if (!*str)
return false;
++str;
++pattern;
break;
case '*':
if (wildCardMatch(str, pattern + 1))
return true;
if (*str && wildCardMatch(str + 1, pattern))
return true;
return false;
default:
if (*str++ != *pattern++)
return false;
break;
}
}
return !*str && !*pattern;
}

static inline std::string& ltrim(std::string& str) {
str.erase(str.begin(), std::find_if(str.begin(), str.end(), [](char c) {
return !std::isspace(c);
} ));
return str;
}

static inline std::string& rtrim(std::string& str) {
str.erase(std::find_if(str.rbegin(), str.rend(), [](char c) {
return !std::isspace(c);
}).base(), str.end());
return str;
}

static inline std::string& trim(std::string& str) {
return ltrim(rtrim(str));
}

static inline bool startsWith(const std::string& str, const std::string& start) {
return (str.length() >= start.length()) && (str.compare(0, start.length(), start) == 0);
}

static inline bool endsWith(const std::string& str, const std::string& end) {
return (str.length() >= end.length()) && (str.compare(str.length() - end.length(), end.length(), end) == 0);
}

static inline std::string& replaceAll(std::string& str, char replaceWhat, char replaceWith) {
std::replace(str.begin(), str.end(), replaceWhat, replaceWith);
return str;
}

static inline std::string& replaceAll(std::string& str, const std::string& replaceWhat,
const std::string& replaceWith) {
if (replaceWhat == replaceWith)
return str;
std::size_t foundAt = std::string::npos;
while ((foundAt = str.find(replaceWhat, foundAt + 1)) != std::string::npos) {
str.replace(foundAt, replaceWhat.length(), replaceWith);
}
return str;
}

static void replaceFirstWithEscape(base::type::string_t& str, const base::type::string_t& replaceWhat,
const base::type::string_t& replaceWith) {
std::size_t foundAt = base::type::string_t::npos;
while ((foundAt = str.find(replaceWhat, foundAt + 1)) != base::type::string_t::npos) {
if (foundAt > 0 && str[foundAt - 1] == base::consts::kFormatSpecifierChar) {
str.erase(foundAt > 0 ? foundAt - 1 : 0, 1);
++foundAt;
} else {
str.replace(foundAt, replaceWhat.length(), replaceWith);
return;
}
}
}
#if defined(ELPP_UNICODE)
static void replaceFirstWithEscape(base::type::string_t& str, const base::type::string_t& replaceWhat,
const std::string& replaceWith) {
replaceFirstWithEscape(str, replaceWhat, base::type::string_t(replaceWith.begin(), replaceWith.end()));
}
#endif  
static inline std::string& toUpper(std::string& str) {
std::transform(str.begin(), str.end(), str.begin(), ::toupper);
return str;
}

static inline bool cStringEq(const char* s1, const char* s2) {
if (s1 == nullptr && s2 == nullptr) return true;
if (s1 == nullptr || s2 == nullptr) return false;
return strcmp(s1, s2) == 0;
}

static bool cStringCaseEq(const char* s1, const char* s2) {
if (s1 == nullptr && s2 == nullptr) return true;
if (s1 == nullptr || s2 == nullptr) return false;
if (strlen(s1) != strlen(s2)) return false;
while (*s1 != '\0' && *s2 != '\0') {
if (::toupper(*s1) != ::toupper(*s2)) return false;
++s1;
++s2;
}
return true;
}

static inline bool contains(const char* str, char c) {
for (; *str; ++str) {
if (*str == c)
return true;
}
return false;
}

static inline char* convertAndAddToBuff(std::size_t n, int len, char* buf, const char* bufLim, bool zeroPadded = true) {
char localBuff[10] = "";
char* p = localBuff + sizeof(localBuff) - 2;
if (n > 0) {
for (; n > 0 && p > localBuff && len > 0; n /= 10, --len)
*--p = static_cast<char>(n % 10 + '0');
} else {
*--p = '0';
--len;
}
if (zeroPadded)
while (p > localBuff && len-- > 0) *--p = static_cast<char>('0');
return addToBuff(p, buf, bufLim);
}

static inline char* addToBuff(const char* str, char* buf, const char* bufLim) {
while ((buf < bufLim) && ((*buf = *str++) != '\0'))
++buf;
return buf;
}

static inline char* clearBuff(char buff[], std::size_t lim) {
STRCPY(buff, "", lim);
ELPP_UNUSED(lim);  
return buff;
}

static char* wcharPtrToCharPtr(const wchar_t* line) {
std::size_t len_ = wcslen(line) + 1;
char* buff_ = static_cast<char*>(malloc(len_ + 1));
#      if ELPP_OS_UNIX || (ELPP_OS_WINDOWS && !ELPP_CRT_DBG_WARNINGS)
std::wcstombs(buff_, line, len_);
#      elif ELPP_OS_WINDOWS
std::size_t convCount_ = 0;
mbstate_t mbState_;
::memset(static_cast<void*>(&mbState_), 0, sizeof(mbState_));
wcsrtombs_s(&convCount_, buff_, len_, &line, len_, &mbState_);
#      endif  
return buff_;
}
};
class OS : base::StaticClass {
public:
#if ELPP_OS_WINDOWS
static const char* getWindowsEnvironmentVariable(const char* varname) {
const DWORD bufferLen = 50;
static char buffer[bufferLen];
if (GetEnvironmentVariableA(varname, buffer, bufferLen)) {
return buffer;
}
return nullptr;
}
#endif  
#if ELPP_OS_ANDROID
static inline std::string getProperty(const char* prop) {
char propVal[PROP_VALUE_MAX + 1];
int ret = __system_property_get(prop, propVal);
return ret == 0 ? std::string() : std::string(propVal);
}

static std::string getDeviceName(void) {
std::stringstream ss;
std::string manufacturer = getProperty("ro.product.manufacturer");
std::string model = getProperty("ro.product.model");
if (manufacturer.empty() || model.empty()) {
return std::string();
}
ss << manufacturer << "-" << model;
return ss.str();
}
#endif  

static const std::string getBashOutput(const char* command) {
#if (ELPP_OS_UNIX && !ELPP_OS_ANDROID && !ELPP_CYGWIN)
if (command == nullptr) {
return std::string();
}
FILE* proc = nullptr;
if ((proc = popen(command, "r")) == nullptr) {
ELPP_INTERNAL_ERROR("\nUnable to run command [" << command << "]", true);
return std::string();
}
char hBuff[4096];
if (fgets(hBuff, sizeof(hBuff), proc) != nullptr) {
pclose(proc);
if (hBuff[strlen(hBuff) - 1] == '\n') {
hBuff[strlen(hBuff) - 1] = '\0';
}
return std::string(hBuff);
}
return std::string();
#else
ELPP_UNUSED(command);
return std::string();
#endif  
}

static std::string getEnvironmentVariable(const char* variableName, const char* defaultVal,
const char* alternativeBashCommand = nullptr) {
#if ELPP_OS_UNIX
const char* val = getenv(variableName);
#elif ELPP_OS_WINDOWS
const char* val = getWindowsEnvironmentVariable(variableName);
#endif  
if ((val == nullptr) || ((strcmp(val, "") == 0))) {
#if ELPP_OS_UNIX && defined(ELPP_FORCE_ENV_VAR_FROM_BASH)
std::string valBash = base::utils::OS::getBashOutput(alternativeBashCommand);
if (valBash.empty()) {
return std::string(defaultVal);
} else {
return valBash;
}
#elif ELPP_OS_WINDOWS || ELPP_OS_UNIX
ELPP_UNUSED(alternativeBashCommand);
return std::string(defaultVal);
#endif  
}
return std::string(val);
}
static inline std::string currentUser(void) {
#if ELPP_OS_UNIX && !ELPP_OS_ANDROID
return getEnvironmentVariable("USER", base::consts::kUnknownUser, "whoami");
#elif ELPP_OS_WINDOWS
return getEnvironmentVariable("USERNAME", base::consts::kUnknownUser);
#elif ELPP_OS_ANDROID
ELPP_UNUSED(base::consts::kUnknownUser);
return std::string("android");
#else
return std::string();
#endif  
}

static inline std::string currentHost(void) {
#if ELPP_OS_UNIX && !ELPP_OS_ANDROID
return getEnvironmentVariable("HOSTNAME", base::consts::kUnknownHost, "hostname");
#elif ELPP_OS_WINDOWS
return getEnvironmentVariable("COMPUTERNAME", base::consts::kUnknownHost);
#elif ELPP_OS_ANDROID
ELPP_UNUSED(base::consts::kUnknownHost);
return getDeviceName();
#else
return std::string();
#endif  
}
static inline bool termSupportsColor(void) {
std::string term = getEnvironmentVariable("TERM", "");
return term == "xterm" || term == "xterm-color" || term == "xterm-256color"
|| term == "screen" || term == "linux" || term == "cygwin"
|| term == "screen-256color";
}
};
extern std::string s_currentUser;
extern std::string s_currentHost;
extern bool s_termSupportsColor;
#define ELPP_INITI_BASIC_DECLR \
namespace el {\
namespace base {\
namespace utils {\
std::string s_currentUser = el::base::utils::OS::currentUser(); \
std::string s_currentHost = el::base::utils::OS::currentHost(); \
bool s_termSupportsColor = el::base::utils::OS::termSupportsColor(); \
}\
}\
}
class DateTime : base::StaticClass {
public:
static void gettimeofday(struct timeval* tv) {
#if ELPP_OS_WINDOWS
if (tv != nullptr) {
#  if ELPP_COMPILER_MSVC || defined(_MSC_EXTENSIONS)
const unsigned __int64 delta_ = 11644473600000000Ui64;
#  else
const unsigned __int64 delta_ = 11644473600000000ULL;
#  endif  
const double secOffSet = 0.000001;
const unsigned long usecOffSet = 1000000;
FILETIME fileTime;
GetSystemTimeAsFileTime(&fileTime);
unsigned __int64 present = 0;
present |= fileTime.dwHighDateTime;
present = present << 32;
present |= fileTime.dwLowDateTime;
present /= 10;  
present -= delta_;
tv->tv_sec = static_cast<long>(present * secOffSet);
tv->tv_usec = static_cast<long>(present % usecOffSet);
}
#else
::gettimeofday(tv, nullptr);
#endif  
}

static std::string getDateTime(const char* format, const base::MillisecondsWidth* msWidth) {
struct timeval currTime;
gettimeofday(&currTime);
return timevalToString(currTime, format, msWidth);
}

static std::string timevalToString(struct timeval tval, const char* format,
const el::base::MillisecondsWidth* msWidth) {
struct ::tm timeInfo;
buildTimeInfo(&tval, &timeInfo);
const int kBuffSize = 30;
char buff_[kBuffSize] = "";
parseFormat(buff_, kBuffSize, format, &timeInfo, static_cast<std::size_t>(tval.tv_usec / msWidth->m_offset),
msWidth);
return std::string(buff_);
}

static base::type::string_t formatTime(unsigned long long time, base::TimestampUnit timestampUnit) {
double result = static_cast<double>(time);
base::type::EnumType start = static_cast<base::type::EnumType>(timestampUnit);
const base::type::char_t* unit = base::consts::kTimeFormats[start].unit;
for (base::type::EnumType i = start; i < base::consts::kTimeFormatsCount - 1; ++i) {
if (result <= base::consts::kTimeFormats[i].value) {
break;
}
result /= base::consts::kTimeFormats[i].value;
unit = base::consts::kTimeFormats[i + 1].unit;
}
base::type::stringstream_t ss;
ss << result << " " << unit;
return ss.str();
}

static inline unsigned long long getTimeDifference(const struct timeval& endTime, const struct timeval& startTime,
base::TimestampUnit timestampUnit) {
if (timestampUnit == base::TimestampUnit::Microsecond) {
return static_cast<unsigned long long>(static_cast<unsigned long long>(1000000 * endTime.tv_sec + endTime.tv_usec) -
static_cast<unsigned long long>(1000000 * startTime.tv_sec + startTime.tv_usec));
} else {
return static_cast<unsigned long long>((((endTime.tv_sec - startTime.tv_sec) * 1000000) +
(endTime.tv_usec - startTime.tv_usec)) / 1000);
}
}


private:
static inline struct ::tm* buildTimeInfo(struct timeval* currTime, struct ::tm* timeInfo) {
#if ELPP_OS_UNIX
time_t rawTime = currTime->tv_sec;
::localtime_r(&rawTime, timeInfo);
return timeInfo;
#else
#  if ELPP_COMPILER_MSVC
ELPP_UNUSED(currTime);
time_t t;
_time64(&t);
localtime_s(timeInfo, &t);
return timeInfo;
#  else
time_t rawTime = currTime->tv_sec;
struct tm* tmInf = localtime(&rawTime);
*timeInfo = *tmInf;
return timeInfo;
#  endif  
#endif  
}
static char* parseFormat(char* buf, std::size_t bufSz, const char* format, const struct tm* tInfo,
std::size_t msec, const base::MillisecondsWidth* msWidth) {
const char* bufLim = buf + bufSz;
for (; *format; ++format) {
if (*format == base::consts::kFormatSpecifierChar) {
switch (*++format) {
case base::consts::kFormatSpecifierChar:  
break;
case '\0':  
--format;
break;
case 'd':  
buf = base::utils::Str::convertAndAddToBuff(tInfo->tm_mday, 2, buf, bufLim);
continue;
case 'a':  
buf = base::utils::Str::addToBuff(base::consts::kDaysAbbrev[tInfo->tm_wday], buf, bufLim);
continue;
case 'A':  
buf = base::utils::Str::addToBuff(base::consts::kDays[tInfo->tm_wday], buf, bufLim);
continue;
case 'M':  
buf = base::utils::Str::convertAndAddToBuff(tInfo->tm_mon + 1, 2, buf, bufLim);
continue;
case 'b':  
buf = base::utils::Str::addToBuff(base::consts::kMonthsAbbrev[tInfo->tm_mon], buf, bufLim);
continue;
case 'B':  
buf = base::utils::Str::addToBuff(base::consts::kMonths[tInfo->tm_mon], buf, bufLim);
continue;
case 'y':  
buf = base::utils::Str::convertAndAddToBuff(tInfo->tm_year + base::consts::kYearBase, 2, buf, bufLim);
continue;
case 'Y':  
buf = base::utils::Str::convertAndAddToBuff(tInfo->tm_year + base::consts::kYearBase, 4, buf, bufLim);
continue;
case 'h':  
buf = base::utils::Str::convertAndAddToBuff(tInfo->tm_hour % 12, 2, buf, bufLim);
continue;
case 'H':  
buf = base::utils::Str::convertAndAddToBuff(tInfo->tm_hour, 2, buf, bufLim);
continue;
case 'm':  
buf = base::utils::Str::convertAndAddToBuff(tInfo->tm_min, 2, buf, bufLim);
continue;
case 's':  
buf = base::utils::Str::convertAndAddToBuff(tInfo->tm_sec, 2, buf, bufLim);
continue;
case 'z':  
case 'g':
buf = base::utils::Str::convertAndAddToBuff(msec, msWidth->m_width, buf, bufLim);
continue;
case 'F':  
buf = base::utils::Str::addToBuff((tInfo->tm_hour >= 12) ? base::consts::kPm : base::consts::kAm, buf, bufLim);
continue;
default:
continue;
}
}
if (buf == bufLim) break;
*buf++ = *format;
}
return buf;
}
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
inline void setArgs(int argc, char** argv) {
m_params.clear();
m_paramsWithValue.clear();
if (argc == 0 || argv == nullptr) {
return;
}
m_argc = argc;
m_argv = argv;
for (int i = 1; i < m_argc; ++i) {
const char* v = (strstr(m_argv[i], "="));
if (v != nullptr && strlen(v) > 0) {
std::string key = std::string(m_argv[i]);
key = key.substr(0, key.find_first_of('='));
if (hasParamWithValue(key.c_str())) {
ELPP_INTERNAL_INFO(1, "Skipping [" << key << "] arg since it already has value ["
<< getParamValue(key.c_str()) << "]");
} else {
m_paramsWithValue.insert(std::make_pair(key, std::string(v + 1)));
}
}
if (v == nullptr) {
if (hasParam(m_argv[i])) {
ELPP_INTERNAL_INFO(1, "Skipping [" << m_argv[i] << "] arg since it already exists");
} else {
m_params.push_back(std::string(m_argv[i]));
}
}
}
}
inline bool hasParamWithValue(const char* paramKey) const {
return m_paramsWithValue.find(std::string(paramKey)) != m_paramsWithValue.end();
}
inline const char* getParamValue(const char* paramKey) const {
return m_paramsWithValue.find(std::string(paramKey))->second.c_str();
}
inline bool hasParam(const char* paramKey) const {
return std::find(m_params.begin(), m_params.end(), std::string(paramKey)) != m_params.end();
}
inline bool empty(void) const {
return m_params.empty() && m_paramsWithValue.empty();
}
inline std::size_t size(void) const {
return m_params.size() + m_paramsWithValue.size();
}
inline friend base::type::ostream_t& operator<<(base::type::ostream_t& os, const CommandLineArgs& c) {
for (int i = 1; i < c.m_argc; ++i) {
os << ELPP_LITERAL("[") << c.m_argv[i] << ELPP_LITERAL("]");
if (i < c.m_argc - 1) {
os << ELPP_LITERAL(" ");
}
}
return os;
}

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
virtual inline void unregisterAll(void) ELPP_FINAL {
if (!this->empty()) {
#ifdef HAS_RANGE_FOR
for (auto&& curr : this->list()) {
base::utils::safeDelete(curr.second);
#else
for (iterator curr = this->begin(); curr != this->end(); ++curr) {
base::utils::safeDelete(curr->second);
#endif
}
this->list().clear();
}
}

virtual inline void registerNew(const T_Key& uniqKey, T_Ptr* ptr) ELPP_FINAL {
unregister(uniqKey);
this->list().insert(std::make_pair(uniqKey, ptr));
}

inline void unregister(const T_Key& uniqKey) {
T_Ptr* existing = get(uniqKey);
if (existing != nullptr) {
base::utils::safeDelete(existing);
this->list().erase(uniqKey);
}
}

inline T_Ptr* get(const T_Key& uniqKey) {
iterator it = this->list().find(uniqKey);
return it == this->list().end()
? nullptr
: it->second;
}

private:
virtual inline void deepCopy(const AbstractRegistry<T_Ptr, std::map<T_Key, T_Ptr*>>& sr) ELPP_FINAL {
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

friend inline base::type::ostream_t& operator<<(base::type::ostream_t& os, const RegistryWithPred& sr) {
for (const_iterator it = sr.list().begin(); it != sr.list().end(); ++it) {
os << ELPP_LITERAL("    ") << **it << ELPP_LITERAL("\n");
}
return os;
}

protected:
virtual inline void unregisterAll(void) ELPP_FINAL {
if (!this->empty()) {
#ifdef HAS_RANGE_FOR
for (auto&& curr : this->list()) {
base::utils::safeDelete(curr);
#else
for (iterator curr = this->begin();
curr != this->end(); ++curr) {
base::utils::safeDelete(*curr);
#endif
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
inline T_Ptr* get(const T& arg1, const T2 arg2) {
iterator iter = std::find_if(this->list().begin(), this->list().end(), Pred(arg1, arg2));
if (iter != this->list().end() && *iter != nullptr) {
return *iter;
}
return nullptr;
}

private:
virtual inline void deepCopy(const AbstractRegistry<T_Ptr, std::vector<T_Ptr*>>& sr) {
for (const_iterator it = sr.list().begin(); it != sr.list().end(); ++it) {
registerNew(new T_Ptr(**it));
}
}
};
class Utils {
public:
template <typename T, typename TPtr>
static inline bool installCallback(const std::string& id, std::map<std::string, TPtr>* mapT) {
if (mapT->find(id) == mapT->end()) {
mapT->insert(std::make_pair(id, TPtr(new T())));
return true;
}
return false;
}

template <typename T, typename TPtr>
static inline void uninstallCallback(const std::string& id, std::map<std::string, TPtr>* mapT) {
if (mapT->find(id) != mapT->end()) {
mapT->erase(id);
}
}

template <typename T, typename TPtr>
static inline T* callback(const std::string& id, std::map<std::string, TPtr>* mapT) {
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
LogFormat(void) :
m_level(el::Level::Unknown),
m_userFormat(base::type::string_t()),
m_format(base::type::string_t()),
m_dateTimeFormat(std::string()),
m_flags(0x0) {
}

LogFormat(el::Level level, const base::type::string_t& format)
: m_level(level), m_userFormat(format) {
parseFromFormat(m_userFormat);
}

LogFormat(const LogFormat& logFormat) {
m_level = logFormat.m_level;
m_userFormat = logFormat.m_userFormat;
m_format = logFormat.m_format;
m_dateTimeFormat = logFormat.m_dateTimeFormat;
m_flags = logFormat.m_flags;
}

LogFormat(LogFormat&& logFormat) {
m_level = std::move(logFormat.m_level);
m_userFormat = std::move(logFormat.m_userFormat);
m_format = std::move(logFormat.m_format);
m_dateTimeFormat = std::move(logFormat.m_dateTimeFormat);
m_flags = std::move(logFormat.m_flags);
}

LogFormat& operator=(const LogFormat& logFormat) {
m_level = logFormat.m_level;
m_userFormat = logFormat.m_userFormat;
m_dateTimeFormat = logFormat.m_dateTimeFormat;
m_flags = logFormat.m_flags;
return *this;
}

virtual ~LogFormat(void) {
}

inline bool operator==(const LogFormat& other) {
return m_level == other.m_level && m_userFormat == other.m_userFormat && m_format == other.m_format &&
m_dateTimeFormat == other.m_dateTimeFormat && m_flags == other.m_flags;
}

void parseFromFormat(const base::type::string_t& userFormat) {
base::type::string_t formatCopy = userFormat;
m_flags = 0x0;
auto conditionalAddFlag = [&](const base::type::char_t* specifier, base::FormatFlags flag) {
std::size_t foundAt = base::type::string_t::npos;
while ((foundAt = formatCopy.find(specifier, foundAt + 1)) != base::type::string_t::npos) {
if (foundAt > 0 && formatCopy[foundAt - 1] == base::consts::kFormatSpecifierChar) {
if (hasFlag(flag)) {
formatCopy.erase(foundAt > 0 ? foundAt - 1 : 0, 1);
++foundAt;
}
} else {
if (!hasFlag(flag)) addFlag(flag);
}
}
};
conditionalAddFlag(base::consts::kAppNameFormatSpecifier, base::FormatFlags::AppName);
conditionalAddFlag(base::consts::kSeverityLevelFormatSpecifier, base::FormatFlags::Level);
conditionalAddFlag(base::consts::kSeverityLevelShortFormatSpecifier, base::FormatFlags::LevelShort);
conditionalAddFlag(base::consts::kLoggerIdFormatSpecifier, base::FormatFlags::LoggerId);
conditionalAddFlag(base::consts::kThreadIdFormatSpecifier, base::FormatFlags::ThreadId);
conditionalAddFlag(base::consts::kLogFileFormatSpecifier, base::FormatFlags::File);
conditionalAddFlag(base::consts::kLogFileBaseFormatSpecifier, base::FormatFlags::FileBase);
conditionalAddFlag(base::consts::kLogLineFormatSpecifier, base::FormatFlags::Line);
conditionalAddFlag(base::consts::kLogLocationFormatSpecifier, base::FormatFlags::Location);
conditionalAddFlag(base::consts::kLogFunctionFormatSpecifier, base::FormatFlags::Function);
conditionalAddFlag(base::consts::kCurrentUserFormatSpecifier, base::FormatFlags::User);
conditionalAddFlag(base::consts::kCurrentHostFormatSpecifier, base::FormatFlags::Host);
conditionalAddFlag(base::consts::kMessageFormatSpecifier, base::FormatFlags::LogMessage);
conditionalAddFlag(base::consts::kVerboseLevelFormatSpecifier, base::FormatFlags::VerboseLevel);
std::size_t dateIndex = std::string::npos;
if ((dateIndex = formatCopy.find(base::consts::kDateTimeFormatSpecifier)) != std::string::npos) {
while (dateIndex > 0 && formatCopy[dateIndex - 1] == base::consts::kFormatSpecifierChar) {
dateIndex = formatCopy.find(base::consts::kDateTimeFormatSpecifier, dateIndex + 1);
}
if (dateIndex != std::string::npos) {
addFlag(base::FormatFlags::DateTime);
updateDateFormat(dateIndex, formatCopy);
}
}
m_format = formatCopy;
updateFormatSpec();
}

inline el::Level level(void) const {
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
virtual void updateDateFormat(std::size_t index, base::type::string_t& currFormat) ELPP_FINAL {
if (hasFlag(base::FormatFlags::DateTime)) {
index += ELPP_STRLEN(base::consts::kDateTimeFormatSpecifier);
}
const base::type::char_t* ptr = currFormat.c_str() + index;
if ((currFormat.size() > index) && (ptr[0] == '{')) {
++ptr;
int count = 1;  
std::stringstream ss;
for (; *ptr; ++ptr, ++count) {
if (*ptr == '}') {
++count;  
break;
}
ss << *ptr;
}
currFormat.erase(index, count);
m_dateTimeFormat = ss.str();
} else {
if (hasFlag(base::FormatFlags::DateTime)) {
m_dateTimeFormat = std::string(base::consts::kDefaultDateTimeFormat);
}
}
}

virtual void updateFormatSpec(void) ELPP_FINAL {
if (m_level == el::Level::Debug) {
base::utils::Str::replaceFirstWithEscape(m_format, base::consts::kSeverityLevelFormatSpecifier,
base::consts::kDebugLevelLogValue);
base::utils::Str::replaceFirstWithEscape(m_format, base::consts::kSeverityLevelShortFormatSpecifier,
base::consts::kDebugLevelShortLogValue);
} else if (m_level == el::Level::Info) {
base::utils::Str::replaceFirstWithEscape(m_format, base::consts::kSeverityLevelFormatSpecifier,
base::consts::kInfoLevelLogValue);
base::utils::Str::replaceFirstWithEscape(m_format, base::consts::kSeverityLevelShortFormatSpecifier,
base::consts::kInfoLevelShortLogValue);
} else if (m_level == el::Level::Warning) {
base::utils::Str::replaceFirstWithEscape(m_format, base::consts::kSeverityLevelFormatSpecifier,
base::consts::kWarningLevelLogValue);
base::utils::Str::replaceFirstWithEscape(m_format, base::consts::kSeverityLevelShortFormatSpecifier,
base::consts::kWarningLevelShortLogValue);
} else if (m_level == el::Level::Error) {
base::utils::Str::replaceFirstWithEscape(m_format, base::consts::kSeverityLevelFormatSpecifier,
base::consts::kErrorLevelLogValue);
base::utils::Str::replaceFirstWithEscape(m_format, base::consts::kSeverityLevelShortFormatSpecifier,
base::consts::kErrorLevelShortLogValue);
} else if (m_level == el::Level::Fatal) {
base::utils::Str::replaceFirstWithEscape(m_format, base::consts::kSeverityLevelFormatSpecifier,
base::consts::kFatalLevelLogValue);
base::utils::Str::replaceFirstWithEscape(m_format, base::consts::kSeverityLevelShortFormatSpecifier,
base::consts::kFatalLevelShortLogValue);
} else if (m_level == el::Level::Verbose) {
base::utils::Str::replaceFirstWithEscape(m_format, base::consts::kSeverityLevelFormatSpecifier,
base::consts::kVerboseLevelLogValue);
base::utils::Str::replaceFirstWithEscape(m_format, base::consts::kSeverityLevelShortFormatSpecifier,
base::consts::kVerboseLevelShortLogValue);
} else if (m_level == el::Level::Trace) {
base::utils::Str::replaceFirstWithEscape(m_format, base::consts::kSeverityLevelFormatSpecifier,
base::consts::kTraceLevelLogValue);
base::utils::Str::replaceFirstWithEscape(m_format, base::consts::kSeverityLevelShortFormatSpecifier,
base::consts::kTraceLevelShortLogValue);
}
if (hasFlag(base::FormatFlags::User)) {
std::string s = base::utils::s_currentUser;
base::utils::Str::replaceFirstWithEscape(m_format, base::consts::kCurrentUserFormatSpecifier,
base::utils::s_currentUser);
}
if (hasFlag(base::FormatFlags::Host)) {
base::utils::Str::replaceFirstWithEscape(m_format, base::consts::kCurrentHostFormatSpecifier,
base::utils::s_currentHost);
}
}

inline void addFlag(base::FormatFlags flag) {
base::utils::addFlag(flag, &m_flags);
}

private:
el::Level m_level;
base::type::string_t m_userFormat;
base::type::string_t m_format;
std::string m_dateTimeFormat;
base::type::EnumType m_flags;
friend class el::Logger;  
};
}  
typedef std::function<const char*(void)> FormatSpecifierValueResolver;
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
Configuration(const Configuration& c) :
m_level(c.m_level),
m_configurationType(c.m_configurationType),
m_value(c.m_value) {
}

Configuration& operator=(const Configuration& c) {
m_level = c.m_level;
m_configurationType = c.m_configurationType;
m_value = c.m_value;
return *this;
}

virtual ~Configuration(void) {
}

Configuration(el::Level level, el::ConfigurationType configurationType, const std::string& value) :
m_level(level),
m_configurationType(configurationType),
m_value(value) {
}

inline el::Level level(void) const {
return m_level;
}

inline el::ConfigurationType configurationType(void) const {
return m_configurationType;
}

inline const std::string& value(void) const {
return m_value;
}

inline void setValue(const std::string& value) {
m_value = value;
}

virtual inline void log(el::base::type::ostream_t& os) const {
os << LevelHelper::convertToString(m_level)
<< ELPP_LITERAL(" ") << ConfigurationTypeHelper::convertToString(m_configurationType)
<< ELPP_LITERAL(" = ") << m_value.c_str();
}

class Predicate {
public:
Predicate(el::Level level, el::ConfigurationType configurationType) :
m_level(level),
m_configurationType(configurationType) {
}

inline bool operator()(const el::Configuration* conf) const {
return ((conf != nullptr) && (conf->level() == m_level) && (conf->configurationType() == m_configurationType));
}

private:
el::Level m_level;
el::ConfigurationType m_configurationType;
};

private:
el::Level m_level;
el::ConfigurationType m_configurationType;
std::string m_value;
};

class Configurations : public base::utils::RegistryWithPred<Configuration, Configuration::Predicate> {
public:
Configurations(void) :
m_configurationFile(std::string()),
m_isFromFile(false) {
}

Configurations(const std::string& configurationFile, bool useDefaultsForRemaining = true,
Configurations* base = nullptr) :
m_configurationFile(configurationFile),
m_isFromFile(false) {
parseFromFile(configurationFile, base);
if (useDefaultsForRemaining) {
setRemainingToDefault();
}
}

virtual ~Configurations(void) {
}

inline bool parseFromFile(const std::string& configurationFile, Configurations* base = nullptr) {
bool assertionPassed = true;
ELPP_ASSERT((assertionPassed = base::utils::File::pathExists(configurationFile.c_str(), true)),
"Configuration file [" << configurationFile << "] does not exist!");
if (!assertionPassed) {
return false;
}
bool success = Parser::parseFromFile(configurationFile, this, base);
m_isFromFile = success;
return success;
}

inline bool parseFromText(const std::string& configurationsString, Configurations* base = nullptr) {
bool success = Parser::parseFromText(configurationsString, this, base);
if (success) {
m_isFromFile = false;
}
return success;
}

inline void setFromBase(el::Configurations* base) {
if (base == nullptr || base == this) {
return;
}
base::threading::ScopedLock scopedLock(base->lock());
#ifdef HAS_RANGE_FOR
for (el::Configuration*& conf : base->list()) {
set(conf);
#else
for (iterator iter = base->begin(); iter != base->end(); ++iter) {
set(*iter);
#endif
}
}

bool hasConfiguration(ConfigurationType configurationType) {
base::type::EnumType lIndex = LevelHelper::kMinValid;
bool result = false;
LevelHelper::forEachLevel(&lIndex, [&](void) -> bool {
if (hasConfiguration(LevelHelper::castFromInt(lIndex), configurationType)) {
result = true;
}
return result;
});
return result;
}

inline bool hasConfiguration(el::Level level, el::ConfigurationType configurationType) {
base::threading::ScopedLock scopedLock(lock());
#if ELPP_COMPILER_INTEL
return RegistryWithPred::get(level, configurationType) != nullptr;
#else
return RegistryWithPred<Configuration, Configuration::Predicate>::get(level, configurationType) != nullptr;
#endif  
}

inline void set(el::Level level, el::ConfigurationType configurationType, const std::string& value) {
base::threading::ScopedLock scopedLock(lock());
unsafeSet(level, configurationType, value);  
if (level == el::Level::Global) {
unsafeSetGlobally(configurationType, value, false);  
}
}

inline void set(el::Configuration* conf) {
if (conf == nullptr) {
return;
}
set(conf->level(), conf->configurationType(), conf->value());
}

inline el::Configuration* get(el::Level level, el::ConfigurationType configurationType) {
base::threading::ScopedLock scopedLock(lock());
return RegistryWithPred<el::Configuration, el::Configuration::Predicate>::get(level, configurationType);
}

inline void setGlobally(el::ConfigurationType configurationType, const std::string& value) {
setGlobally(configurationType, value, false);
}

inline void clear(void) {
base::threading::ScopedLock scopedLock(lock());
unregisterAll();
}

inline const std::string& configurationFile(void) const {
return m_configurationFile;
}

void setToDefault(void) {
setGlobally(el::ConfigurationType::Enabled, std::string("true"), true);
#if !defined(ELPP_NO_DEFAULT_LOG_FILE)
setGlobally(ConfigurationType::Filename, std::string(base::consts::kDefaultLogFile), true);
#else
ELPP_UNUSED(base::consts::kDefaultLogFile);
#endif  
#if defined(ELPP_NO_LOG_TO_FILE)
setGlobally(ConfigurationType::ToFile, std::string("false"), true);
#else
setGlobally(ConfigurationType::ToFile, std::string("true"), true);
#endif 
setGlobally(ConfigurationType::ToStandardOutput, std::string("true"), true);
setGlobally(ConfigurationType::MillisecondsWidth, std::string("3"), true);
setGlobally(ConfigurationType::PerformanceTracking, std::string("true"), true);
setGlobally(ConfigurationType::MaxLogFileSize, std::string("0"), true);
setGlobally(ConfigurationType::LogFlushThreshold, std::string("0"), true);

setGlobally(ConfigurationType::Format, std::string("%datetime %level [%logger] %msg"), true);
set(Level::Debug, ConfigurationType::Format,
std::string("%datetime %level [%logger] [%user@%host] [%func] [%loc] %msg"));
set(Level::Error, ConfigurationType::Format, std::string("%datetime %level [%logger] %msg"));
set(Level::Fatal, ConfigurationType::Format, std::string("%datetime %level [%logger] %msg"));
set(Level::Verbose, ConfigurationType::Format, std::string("%datetime %level-%vlevel [%logger] %msg"));
set(Level::Trace, ConfigurationType::Format, std::string("%datetime %level [%logger] [%func] [%loc] %msg"));
}

void setRemainingToDefault(void) {
base::threading::ScopedLock scopedLock(lock());
#if defined(ELPP_NO_LOG_TO_FILE)
unsafeSetIfNotExist(Level::Global, ConfigurationType::Enabled, std::string("false"));
#else
unsafeSetIfNotExist(Level::Global, ConfigurationType::Enabled, std::string("true"));
#endif 
#if !defined(ELPP_NO_DEFAULT_LOG_FILE)
unsafeSetIfNotExist(Level::Global, ConfigurationType::Filename, std::string(base::consts::kDefaultLogFile));
#endif  
unsafeSetIfNotExist(Level::Global, ConfigurationType::ToStandardOutput, std::string("true"));
unsafeSetIfNotExist(Level::Global, ConfigurationType::MillisecondsWidth, std::string("3"));
unsafeSetIfNotExist(Level::Global, ConfigurationType::PerformanceTracking, std::string("true"));
unsafeSetIfNotExist(Level::Global, ConfigurationType::MaxLogFileSize, std::string("0"));
unsafeSetIfNotExist(Level::Global, ConfigurationType::Format, std::string("%datetime %level [%logger] %msg"));
unsafeSetIfNotExist(Level::Debug, ConfigurationType::Format,
std::string("%datetime %level [%logger] [%user@%host] [%func] [%loc] %msg"));
unsafeSetIfNotExist(Level::Error, ConfigurationType::Format, std::string("%datetime %level [%logger] %msg"));
unsafeSetIfNotExist(Level::Fatal, ConfigurationType::Format, std::string("%datetime %level [%logger] %msg"));
unsafeSetIfNotExist(Level::Verbose, ConfigurationType::Format, std::string("%datetime %level-%vlevel [%logger] %msg"));
unsafeSetIfNotExist(Level::Trace, ConfigurationType::Format,
std::string("%datetime %level [%logger] [%func] [%loc] %msg"));
}

class Parser : base::StaticClass {
public:
static bool parseFromFile(const std::string& configurationFile, Configurations* sender,
Configurations* base = nullptr) {
sender->setFromBase(base);
std::ifstream fileStream_(configurationFile.c_str(), std::ifstream::in);
ELPP_ASSERT(fileStream_.is_open(), "Unable to open configuration file [" << configurationFile << "] for parsing.");
bool parsedSuccessfully = false;
std::string line = std::string();
el::Level currLevel = el::Level::Unknown;
std::string currConfigStr = std::string();
std::string currLevelStr = std::string();
while (fileStream_.good()) {
std::getline(fileStream_, line);
parsedSuccessfully = parseLine(&line, &currConfigStr, &currLevelStr, &currLevel, sender);
ELPP_ASSERT(parsedSuccessfully, "Unable to parse configuration line: " << line);
}
return parsedSuccessfully;
}

static bool parseFromText(const std::string& configurationsString, Configurations* sender,
Configurations* base = nullptr) {
sender->setFromBase(base);
bool parsedSuccessfully = false;
std::stringstream ss(configurationsString);
std::string line = std::string();
el::Level currLevel = el::Level::Unknown;
std::string currConfigStr = std::string();
std::string currLevelStr = std::string();
while (std::getline(ss, line)) {
parsedSuccessfully = parseLine(&line, &currConfigStr, &currLevelStr, &currLevel, sender);
ELPP_ASSERT(parsedSuccessfully, "Unable to parse configuration line: " << line);
}
return parsedSuccessfully;
}

private:
friend class el::Loggers;
static void ignoreComments(std::string* line) {
std::size_t foundAt = 0;
std::size_t quotesStart = line->find("\"");
std::size_t quotesEnd = std::string::npos;
if (quotesStart != std::string::npos) {
quotesEnd = line->find("\"", quotesStart + 1);
while (quotesEnd != std::string::npos && line->at(quotesEnd - 1) == '\\') {
quotesEnd = line->find("\"", quotesEnd + 2);
}
}
if ((foundAt = line->find(base::consts::kConfigurationComment)) != std::string::npos) {
if (foundAt < quotesEnd) {
foundAt = line->find(base::consts::kConfigurationComment, quotesEnd + 1);
}
*line = line->substr(0, foundAt);
}
}
static inline bool isLevel(const std::string& line) {
return base::utils::Str::startsWith(line, std::string(base::consts::kConfigurationLevel));
}

static inline bool isComment(const std::string& line) {
return base::utils::Str::startsWith(line, std::string(base::consts::kConfigurationComment));
}

static inline bool isConfig(const std::string& line) {
std::size_t assignment = line.find('=');
return line != "" &&
((line[0] >= 'A' && line[0] <= 'Z') || (line[0] >= 'a' && line[0] <= 'z')) &&
(assignment != std::string::npos) &&
(line.size() > assignment);
}

static bool parseLine(std::string* line, std::string* currConfigStr, std::string* currLevelStr, Level* currLevel,
el::Configurations* conf) {
el::ConfigurationType currConfig = el::ConfigurationType::Unknown_Type;
std::string currValue = std::string();
*line = base::utils::Str::trim(*line);
if (isComment(*line)) return true;
ignoreComments(line);
*line = base::utils::Str::trim(*line);
if (line->empty()) {
return true;
}
if (isLevel(*line)) {
if (line->size() <= 2) {
return true;
}
*currLevelStr = line->substr(1, line->size() - 2);
*currLevelStr = base::utils::Str::toUpper(*currLevelStr);
*currLevelStr = base::utils::Str::trim(*currLevelStr);
*currLevel = LevelHelper::convertFromString(currLevelStr->c_str());
return true;
}
if (isConfig(*line)) {
std::size_t assignment = line->find('=');
*currConfigStr = line->substr(0, assignment);
*currConfigStr = base::utils::Str::toUpper(*currConfigStr);
*currConfigStr = base::utils::Str::trim(*currConfigStr);
currConfig = ConfigurationTypeHelper::convertFromString(currConfigStr->c_str());
currValue = line->substr(assignment + 1);
currValue = base::utils::Str::trim(currValue);
std::size_t quotesStart = currValue.find("\"", 0);
std::size_t quotesEnd = std::string::npos;
if (quotesStart != std::string::npos) {
quotesEnd = currValue.find("\"", quotesStart + 1);
while (quotesEnd != std::string::npos && currValue.at(quotesEnd - 1) == '\\') {
currValue = currValue.erase(quotesEnd - 1, 1);
quotesEnd = currValue.find("\"", quotesEnd + 2);
}
}
if (quotesStart != std::string::npos && quotesEnd != std::string::npos) {
ELPP_ASSERT((quotesStart < quotesEnd), "Configuration error - No ending quote found in ["
<< currConfigStr << "]");
ELPP_ASSERT((quotesStart + 1 != quotesEnd), "Empty configuration value for [" << currConfigStr << "]");
if ((quotesStart != quotesEnd) && (quotesStart + 1 != quotesEnd)) {
currValue = currValue.substr(quotesStart + 1, quotesEnd - 1);
}
}
}
ELPP_ASSERT(*currLevel != el::Level::Unknown, "Unrecognized severity level [" << *currLevelStr << "]");
ELPP_ASSERT(currConfig != el::ConfigurationType::Unknown_Type, "Unrecognized configuration [" << *currConfigStr << "]");
if (*currLevel == el::Level::Unknown || currConfig == el::ConfigurationType::Unknown_Type) {
return false;  
}
conf->set(*currLevel, currConfig, currValue);
return true;
}
};

private:
std::string m_configurationFile;
bool m_isFromFile;
friend class el::Loggers;

void unsafeSetIfNotExist(el::Level level, el::ConfigurationType configurationType, const std::string& value) {
el::Configuration* conf = RegistryWithPred<el::Configuration, el::Configuration::Predicate>::get(level, configurationType);
if (conf == nullptr) {
unsafeSet(level, configurationType, value);
}
}

void unsafeSet(el::Level level, el::ConfigurationType configurationType, const std::string& value) {
el::Configuration* conf = RegistryWithPred<el::Configuration, el::Configuration::Predicate>::get(level, configurationType);
if (conf == nullptr) {
registerNew(new Configuration(level, configurationType, value));
} else {
conf->setValue(value);
}
if (level == el::Level::Global) {
unsafeSetGlobally(configurationType, value, false);
}
}

void setGlobally(ConfigurationType configurationType, const std::string& value, bool includeGlobalLevel) {
if (includeGlobalLevel) {
set(Level::Global, configurationType, value);
}
base::type::EnumType lIndex = LevelHelper::kMinValid;
LevelHelper::forEachLevel(&lIndex, [&](void) -> bool {
Configurations::set(LevelHelper::castFromInt(lIndex), configurationType, value);
return false;  
});
}

void unsafeSetGlobally(ConfigurationType configurationType, const std::string& value, bool includeGlobalLevel) {
if (includeGlobalLevel) {
unsafeSet(Level::Global, configurationType, value);
}
base::type::EnumType lIndex = LevelHelper::kMinValid;
LevelHelper::forEachLevel(&lIndex, [&](void) -> bool  {
unsafeSet(LevelHelper::castFromInt(lIndex), configurationType, value);
return false;  
});
}
};

namespace base {
typedef std::shared_ptr<base::type::fstream_t> FileStreamPtr;
typedef std::map<std::string, FileStreamPtr> LogStreamsReferenceMap;
class TypedConfigurations : public base::threading::ThreadSafe {
public:
TypedConfigurations(el::Configurations* configurations, base::LogStreamsReferenceMap* logStreamsReference) {
m_configurations = configurations;
m_logStreamsReference = logStreamsReference;
build(m_configurations);
}

TypedConfigurations(const TypedConfigurations& other) {
this->m_configurations = other.m_configurations;
this->m_logStreamsReference = other.m_logStreamsReference;
build(m_configurations);
}

virtual ~TypedConfigurations(void) {
}

const Configurations* configurations(void) const {
return m_configurations;
}

inline bool enabled(el::Level level) {
return getConfigByVal<bool>(level, &m_enabledMap, "enabled");
}

inline bool toFile(el::Level level) {
return getConfigByVal<bool>(level, &m_toFileMap, "toFile");
}

inline const std::string& filename(el::Level level) {
return getConfigByRef<std::string>(level, &m_filenameMap, "filename");
}

inline bool toStandardOutput(el::Level level) {
return getConfigByVal<bool>(level, &m_toStandardOutputMap, "toStandardOutput");
}

inline const base::LogFormat& logFormat(el::Level level) {
return getConfigByRef<base::LogFormat>(level, &m_logFormatMap, "logFormat");
}

inline const base::MillisecondsWidth& millisecondsWidth(el::Level level = el::Level::Global) {
return getConfigByRef<base::MillisecondsWidth>(level, &m_millisecondsWidthMap, "millisecondsWidth");
}

inline bool performanceTracking(el::Level level = el::Level::Global) {
return getConfigByVal<bool>(level, &m_performanceTrackingMap, "performanceTracking");
}

inline base::type::fstream_t* fileStream(el::Level level) {
return getConfigByRef<base::FileStreamPtr>(level, &m_fileStreamMap, "fileStream").get();
}

inline std::size_t maxLogFileSize(el::Level level) {
return getConfigByVal<std::size_t>(level, &m_maxLogFileSizeMap, "maxLogFileSize");
}

inline std::size_t logFlushThreshold(el::Level level) {
return getConfigByVal<std::size_t>(level, &m_logFlushThresholdMap, "logFlushThreshold");
}

private:
Configurations* m_configurations;
std::map<el::Level, bool> m_enabledMap;
std::map<el::Level, bool> m_toFileMap;
std::map<el::Level, std::string> m_filenameMap;
std::map<el::Level, bool> m_toStandardOutputMap;
std::map<el::Level, base::LogFormat> m_logFormatMap;
std::map<el::Level, base::MillisecondsWidth> m_millisecondsWidthMap;
std::map<el::Level, bool> m_performanceTrackingMap;
std::map<el::Level, base::FileStreamPtr> m_fileStreamMap;
std::map<el::Level, std::size_t> m_maxLogFileSizeMap;
std::map<el::Level, std::size_t> m_logFlushThresholdMap;
base::LogStreamsReferenceMap* m_logStreamsReference;

friend class el::Helpers;
friend class el::base::MessageBuilder;
friend class el::base::Writer;
friend class el::base::DefaultLogDispatchCallback;
friend class el::base::LogDispatcher;

template <typename Conf_T>
inline Conf_T getConfigByVal(el::Level level, const std::map<el::Level, Conf_T>* confMap, const char* confName) {
base::threading::ScopedLock scopedLock(lock());
return unsafeGetConfigByVal(level, confMap, confName);  
}

template <typename Conf_T>
inline Conf_T& getConfigByRef(el::Level level, std::map<el::Level, Conf_T>* confMap, const char* confName) {
base::threading::ScopedLock scopedLock(lock());
return unsafeGetConfigByRef(level, confMap, confName);  
}

template <typename Conf_T>
inline Conf_T unsafeGetConfigByVal(el::Level level, const std::map<el::Level, Conf_T>* confMap, const char* confName) {
ELPP_UNUSED(confName);
typename std::map<el::Level, Conf_T>::const_iterator it = confMap->find(level);
if (it == confMap->end()) {
try {
return confMap->at(el::Level::Global);
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
inline Conf_T& unsafeGetConfigByRef(el::Level level, std::map<el::Level, Conf_T>* confMap, const char* confName) {
ELPP_UNUSED(confName);
typename std::map<el::Level, Conf_T>::iterator it = confMap->find(level);
if (it == confMap->end()) {
try {
return confMap->at(el::Level::Global);
} catch (...) {
ELPP_INTERNAL_ERROR("Unable to get configuration [" << confName << "] for level ["
<< LevelHelper::convertToString(level) << "]"
<< std::endl << "Please ensure you have properly configured logger.", false);
}
}
return it->second;
}

template <typename Conf_T>
void setValue(el::Level level, const Conf_T& value, std::map<el::Level, Conf_T>* confMap, bool includeGlobalLevel = true) {
if (confMap->empty() && includeGlobalLevel) {
confMap->insert(std::make_pair(el::Level::Global, value));
return;
}
typename std::map<el::Level, Conf_T>::iterator it = confMap->find(el::Level::Global);
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

void build(Configurations* configurations) {
base::threading::ScopedLock scopedLock(lock());
auto getBool = [] (std::string boolStr) -> bool {  
base::utils::Str::trim(boolStr);
return (boolStr == "TRUE" || boolStr == "true" || boolStr == "1");
};
std::vector<Configuration*> withFileSizeLimit;
for (Configurations::const_iterator it = configurations->begin(); it != configurations->end(); ++it) {
Configuration* conf = *it;
if (conf->configurationType() == ConfigurationType::Enabled) {
setValue(conf->level(), getBool(conf->value()), &m_enabledMap);
} else if (conf->configurationType() == ConfigurationType::ToFile) {
setValue(conf->level(), getBool(conf->value()), &m_toFileMap);
} else if (conf->configurationType() == ConfigurationType::ToStandardOutput) {
setValue(conf->level(), getBool(conf->value()), &m_toStandardOutputMap);
} else if (conf->configurationType() == ConfigurationType::Filename) {
} else if (conf->configurationType() == ConfigurationType::Format) {
setValue(conf->level(), base::LogFormat(conf->level(),
base::type::string_t(conf->value().begin(), conf->value().end())), &m_logFormatMap);
} else if (conf->configurationType() == ConfigurationType::MillisecondsWidth) {
setValue(el::Level::Global,
base::MillisecondsWidth(static_cast<int>(getULong(conf->value()))), &m_millisecondsWidthMap);
} else if (conf->configurationType() == ConfigurationType::PerformanceTracking) {
setValue(el::Level::Global, getBool(conf->value()), &m_performanceTrackingMap);
} else if (conf->configurationType() == ConfigurationType::MaxLogFileSize) {
setValue(conf->level(), static_cast<std::size_t>(getULong(conf->value())), &m_maxLogFileSizeMap);
#if !defined(ELPP_NO_DEFAULT_LOG_FILE)
withFileSizeLimit.push_back(conf);
#endif  
} else if (conf->configurationType() == ConfigurationType::LogFlushThreshold) {
setValue(conf->level(), static_cast<std::size_t>(getULong(conf->value())), &m_logFlushThresholdMap);
}
}
for (Configurations::const_iterator it = configurations->begin(); it != configurations->end(); ++it) {
Configuration* conf = *it;
if (conf->configurationType() == ConfigurationType::Filename) {
insertFile(conf->level(), conf->value());
}
}
for (std::vector<Configuration*>::iterator conf = withFileSizeLimit.begin();
conf != withFileSizeLimit.end(); ++conf) {
unsafeValidateFileRolling((*conf)->level(), base::defaultPreRollOutCallback);
}
}

unsigned long getULong(std::string confVal) {
bool valid = true;
base::utils::Str::trim(confVal);
valid = !confVal.empty() && std::find_if(confVal.begin(), confVal.end(),
[](char c) {
return !base::utils::Str::isDigit(c);
}) == confVal.end();
if (!valid) {
valid = false;
ELPP_ASSERT(valid, "Configuration value not a valid integer [" << confVal << "]");
return 0;
}
return atol(confVal.c_str());
}

std::string resolveFilename(const std::string& filename) {
std::string resultingFilename = filename;
std::size_t dateIndex = std::string::npos;
std::string dateTimeFormatSpecifierStr = std::string(base::consts::kDateTimeFormatSpecifierForFilename);
if ((dateIndex = resultingFilename.find(dateTimeFormatSpecifierStr.c_str())) != std::string::npos) {
while (dateIndex > 0 && resultingFilename[dateIndex - 1] == base::consts::kFormatSpecifierChar) {
dateIndex = resultingFilename.find(dateTimeFormatSpecifierStr.c_str(), dateIndex + 1);
}
if (dateIndex != std::string::npos) {
const char* ptr = resultingFilename.c_str() + dateIndex;
ptr += dateTimeFormatSpecifierStr.size();
std::string fmt;
if ((resultingFilename.size() > dateIndex) && (ptr[0] == '{')) {
++ptr;
int count = 1;  
std::stringstream ss;
for (; *ptr; ++ptr, ++count) {
if (*ptr == '}') {
++count;  
break;
}
ss << *ptr;
}
resultingFilename.erase(dateIndex + dateTimeFormatSpecifierStr.size(), count);
fmt = ss.str();
} else {
fmt = std::string(base::consts::kDefaultDateTimeFormatInFilename);
}
base::MillisecondsWidth msWidth(3);
std::string now = base::utils::DateTime::getDateTime(fmt.c_str(), &msWidth);
base::utils::Str::replaceAll(now, '/', '-'); 
base::utils::Str::replaceAll(resultingFilename, dateTimeFormatSpecifierStr, now);
}
}
return resultingFilename;
}

void insertFile(el::Level level, const std::string& fullFilename) {
#if defined(ELPP_NO_LOG_TO_FILE)
setValue(level, false, &m_toFileMap);
ELPP_UNUSED(fullFilename);
m_fileStreamMap.insert(std::make_pair(level, base::FileStreamPtr(nullptr)));
return;
#endif
std::string resolvedFilename = resolveFilename(fullFilename);
if (resolvedFilename.empty()) {
std::cerr << "Could not load empty file for logging, please re-check your configurations for level ["
<< LevelHelper::convertToString(level) << "]";
}
std::string filePath = base::utils::File::extractPathFromFilename(resolvedFilename, base::consts::kFilePathSeperator);
if (filePath.size() < resolvedFilename.size()) {
base::utils::File::createPath(filePath);
}
auto create = [&](el::Level level) {
base::LogStreamsReferenceMap::iterator filestreamIter = m_logStreamsReference->find(resolvedFilename);
base::type::fstream_t* fs = nullptr;
if (filestreamIter == m_logStreamsReference->end()) {
fs = base::utils::File::newFileStream(resolvedFilename);
m_filenameMap.insert(std::make_pair(level, resolvedFilename));
m_fileStreamMap.insert(std::make_pair(level, base::FileStreamPtr(fs)));
m_logStreamsReference->insert(std::make_pair(resolvedFilename, base::FileStreamPtr(m_fileStreamMap.at(level))));
} else {
m_filenameMap.insert(std::make_pair(level, filestreamIter->first));
m_fileStreamMap.insert(std::make_pair(level, base::FileStreamPtr(filestreamIter->second)));
fs = filestreamIter->second.get();
}
if (fs == nullptr) {
ELPP_INTERNAL_ERROR("Setting [TO_FILE] of ["
<< LevelHelper::convertToString(level) << "] to FALSE", false);
setValue(level, false, &m_toFileMap);
}
};
create(m_filenameMap.empty() && m_fileStreamMap.empty() ? el::Level::Global : level);
}

bool unsafeValidateFileRolling(el::Level level, const PreRollOutCallback& PreRollOutCallback) {
base::type::fstream_t* fs = unsafeGetConfigByRef(level, &m_fileStreamMap, "fileStream").get();
if (fs == nullptr) {
return true;
}
std::size_t maxLogFileSize = unsafeGetConfigByVal(level, &m_maxLogFileSizeMap, "maxLogFileSize");
std::size_t currFileSize = base::utils::File::getSizeOfFile(fs);
if (maxLogFileSize != 0 && currFileSize >= maxLogFileSize) {
std::string fname = unsafeGetConfigByRef(level, &m_filenameMap, "filename");
ELPP_INTERNAL_INFO(1, "Truncating log file [" << fname << "] as a result of configurations for level ["
<< LevelHelper::convertToString(level) << "]");
fs->close();
PreRollOutCallback(fname.c_str(), currFileSize);
fs->open(fname, std::fstream::out | std::fstream::trunc);
return true;
}
return false;
}

bool validateFileRolling(el::Level level, const PreRollOutCallback& PreRollOutCallback) {
base::threading::ScopedLock scopedLock(lock());
return unsafeValidateFileRolling(level, PreRollOutCallback);
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
m_filename = hitCounter.m_filename;
m_lineNumber = hitCounter.m_lineNumber;
m_hitCounts = hitCounter.m_hitCounts;
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
bool validateEveryN(const char* filename, base::type::LineNumber lineNumber, std::size_t n) {
base::threading::ScopedLock scopedLock(lock());
base::HitCounter* counter = get(filename, lineNumber);
if (counter == nullptr) {
registerNew(counter = new base::HitCounter(filename, lineNumber));
}
counter->validateHitCounts(n);
bool result = (n >= 1 && counter->hitCounts() != 0 && counter->hitCounts() % n == 0);
return result;
}

bool validateAfterN(const char* filename, base::type::LineNumber lineNumber, std::size_t n) {
base::threading::ScopedLock scopedLock(lock());
base::HitCounter* counter = get(filename, lineNumber);
if (counter == nullptr) {
registerNew(counter = new base::HitCounter(filename, lineNumber));
}
if (counter->hitCounts() >= n)
return true;
counter->increment();
return false;
}

bool validateNTimes(const char* filename, base::type::LineNumber lineNumber, std::size_t n) {
base::threading::ScopedLock scopedLock(lock());
base::HitCounter* counter = get(filename, lineNumber);
if (counter == nullptr) {
registerNew(counter = new base::HitCounter(filename, lineNumber));
}
counter->increment();
if (counter->hitCounts() <= n)
return true;
return false;
}

inline const base::HitCounter* getCounter(const char* filename, base::type::LineNumber lineNumber) {
base::threading::ScopedLock scopedLock(lock());
return get(filename, lineNumber);
}
};
ENUMCLASS DispatchAction : base::type::EnumType {
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
virtual ~LogBuilder(void) {
ELPP_INTERNAL_INFO(3, "Destroying log builder...")
}
virtual base::type::string_t build(const LogMessage* logMessage, bool appendNewLine) const = 0;
void convertToColoredOutput(base::type::string_t* logLine, el::Level level) {
if (!base::utils::s_termSupportsColor) return;
const base::type::char_t* resetColor = ELPP_LITERAL("\x1b[0m");
if (level == Level::Error || level == Level::Fatal)
*logLine = ELPP_LITERAL("\x1b[31m") + *logLine + resetColor;
else if (level == Level::Warning)
*logLine = ELPP_LITERAL("\x1b[33m") + *logLine + resetColor;
else if (level == Level::Debug)
*logLine = ELPP_LITERAL("\x1b[32m") + *logLine + resetColor;
else if (level == Level::Info)
*logLine = ELPP_LITERAL("\x1b[36m") + *logLine + resetColor;
else if (level == Level::Trace)
*logLine = ELPP_LITERAL("\x1b[35m") + *logLine + resetColor;
}
private:
friend class el::base::DefaultLogDispatchCallback;
};
typedef std::shared_ptr<LogBuilder> LogBuilderPtr;
class Logger : public base::threading::ThreadSafe, public Loggable {
public:
Logger(const std::string& id, base::LogStreamsReferenceMap* logStreamsReference) :
m_id(id),
m_typedConfigurations(nullptr),
m_parentApplicationName(std::string()),
m_isConfigured(false),
m_logStreamsReference(logStreamsReference) {
initUnflushedCount();
}

Logger(const std::string& id, const Configurations& configurations, base::LogStreamsReferenceMap* logStreamsReference) :
m_id(id),
m_typedConfigurations(nullptr),
m_parentApplicationName(std::string()),
m_isConfigured(false),
m_logStreamsReference(logStreamsReference) {
initUnflushedCount();
configure(configurations);
}

Logger(const Logger& logger) {
base::utils::safeDelete(m_typedConfigurations);
m_id = logger.m_id;
m_typedConfigurations = logger.m_typedConfigurations;
m_parentApplicationName = logger.m_parentApplicationName;
m_isConfigured = logger.m_isConfigured;
m_configurations = logger.m_configurations;
m_unflushedCount = logger.m_unflushedCount;
m_logStreamsReference = logger.m_logStreamsReference;
}

Logger& operator=(const Logger& logger) {
base::utils::safeDelete(m_typedConfigurations);
m_id = logger.m_id;
m_typedConfigurations = logger.m_typedConfigurations;
m_parentApplicationName = logger.m_parentApplicationName;
m_isConfigured = logger.m_isConfigured;
m_configurations = logger.m_configurations;
m_unflushedCount = logger.m_unflushedCount;
m_logStreamsReference = logger.m_logStreamsReference;
return *this;
}

virtual ~Logger(void) {
base::utils::safeDelete(m_typedConfigurations);
}

virtual inline void log(el::base::type::ostream_t& os) const {
os << m_id.c_str();
}

void configure(const Configurations& configurations) {
m_isConfigured = false;  
initUnflushedCount();
if (m_typedConfigurations != nullptr) {
Configurations* c = const_cast<Configurations*>(m_typedConfigurations->configurations());
if (c->hasConfiguration(Level::Global, ConfigurationType::Filename)) {
flush();
}
}
base::threading::ScopedLock scopedLock(lock());
if (m_configurations != configurations) {
m_configurations.setFromBase(const_cast<Configurations*>(&configurations));
}
base::utils::safeDelete(m_typedConfigurations);
m_typedConfigurations = new base::TypedConfigurations(&m_configurations, m_logStreamsReference);
resolveLoggerFormatSpec();
m_isConfigured = true;
}

inline void reconfigure(void) {
ELPP_INTERNAL_INFO(1, "Reconfiguring logger [" << m_id << "]");
configure(m_configurations);
}

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

static inline bool isValidId(const std::string& id) {
for (std::string::const_iterator it = id.begin(); it != id.end(); ++it) {
if (!base::utils::Str::contains(base::consts::kValidLoggerIdSymbols, *it)) {
return false;
}
}
return true;
}
inline void flush(void) {
ELPP_INTERNAL_INFO(3, "Flushing logger [" << m_id << "] all levels");
base::threading::ScopedLock scopedLock(lock());
base::type::EnumType lIndex = LevelHelper::kMinValid;
LevelHelper::forEachLevel(&lIndex, [&](void) -> bool {
flush(LevelHelper::castFromInt(lIndex), nullptr);
return false;
});
}

inline void flush(el::Level level, base::type::fstream_t* fs) {
if (fs == nullptr && m_typedConfigurations->toFile(level)) {
fs = m_typedConfigurations->fileStream(level);
}
if (fs != nullptr) {
fs->flush();
m_unflushedCount.find(level)->second = 0;
}
}

inline bool isFlushNeeded(el::Level level) {
return ++m_unflushedCount.find(level)->second >= m_typedConfigurations->logFlushThreshold(level);
}

inline LogBuilder* logBuilder(void) const {
return m_logBuilder.get();
}

inline void setLogBuilder(const LogBuilderPtr& logBuilder) {
m_logBuilder = logBuilder;
}

inline bool enabled(el::Level level) const {
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

void initUnflushedCount(void) {
m_unflushedCount.clear();
base::type::EnumType lIndex = LevelHelper::kMinValid;
LevelHelper::forEachLevel(&lIndex, [&](void) -> bool {
m_unflushedCount.insert(std::make_pair(LevelHelper::castFromInt(lIndex), 0));
return false;
});
}

inline base::type::stringstream_t& stream(void) {
return m_stream;
}

void resolveLoggerFormatSpec(void) const {
base::type::EnumType lIndex = LevelHelper::kMinValid;
LevelHelper::forEachLevel(&lIndex, [&](void) -> bool {
base::LogFormat* logFormat =
const_cast<base::LogFormat*>(&m_typedConfigurations->logFormat(LevelHelper::castFromInt(lIndex)));
base::utils::Str::replaceFirstWithEscape(logFormat->m_format, base::consts::kLoggerIdFormatSpecifier, m_id);
return false;
});
}
};
namespace base {
class RegisteredLoggers : public base::utils::Registry<Logger, std::string> {
public:
explicit RegisteredLoggers(const LogBuilderPtr& defaultLogBuilder) :
m_defaultLogBuilder(defaultLogBuilder) {
m_defaultConfigurations.setToDefault();
}

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

Logger* get(const std::string& id, bool forceCreation = true) {
base::threading::ScopedLock scopedLock(lock());
Logger* logger_ = base::utils::Registry<Logger, std::string>::get(id);
if (logger_ == nullptr && forceCreation) {
bool validId = Logger::isValidId(id);
if (!validId) {
ELPP_ASSERT(validId, "Invalid logger ID [" << id << "]. Not registering this logger.");
return nullptr;
}
logger_ = new Logger(id, m_defaultConfigurations, &m_logStreamsReference);
logger_->m_logBuilder = m_defaultLogBuilder;
registerNew(id, logger_);
LoggerRegistrationCallback* callback = nullptr;
#ifdef HAS_RANGE_FOR
for (const std::pair<std::string, base::type::LoggerRegistrationCallbackPtr>& h
: m_loggerRegistrationCallbacks) {
callback = h.second.get();
#else
for (std::map<std::string, base::type::LoggerRegistrationCallbackPtr>::iterator
h = m_loggerRegistrationCallbacks.begin();
h != m_loggerRegistrationCallbacks.end(); ++h) {
callback = h->second.get();
#endif
if (callback != nullptr && callback->enabled()) {
callback->acquireLock();
callback->handle(logger_);
callback->releaseLock();
}
}
}
return logger_;
}

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

bool remove(const std::string& id) {
if (id == "default") {
return false;
}
Logger* logger = base::utils::Registry<Logger, std::string>::get(id);
if (logger != nullptr) {
unregister(logger);
}
return true;
}

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

inline void unsafeFlushAll(void) {
ELPP_INTERNAL_INFO(1, "Flushing all log files");
for (base::LogStreamsReferenceMap::iterator it = m_logStreamsReference.begin();
it != m_logStreamsReference.end(); ++it) {
if (it->second.get() == nullptr) continue;
it->second->flush();
}
}
};
class VRegistry : base::NoCopy, public base::threading::ThreadSafe {
public:
explicit VRegistry(base::type::VerboseLevel level, base::type::EnumType* pFlags) : m_level(level), m_pFlags(pFlags) {
}

inline void setLevel(base::type::VerboseLevel level) {
base::threading::ScopedLock scopedLock(lock());
if (level > 9)
m_level = base::consts::kMaxVerboseLevel;
else
m_level = level;
}

inline base::type::VerboseLevel level(void) const {
return m_level;
}

inline void clearModules(void) {
base::threading::ScopedLock scopedLock(lock());
m_modules.clear();
}

void setModules(const char* modules) {
base::threading::ScopedLock scopedLock(lock());
auto addSuffix = [](std::stringstream& ss, const char* sfx, const char* prev) {
if (prev != nullptr && base::utils::Str::endsWith(ss.str(), std::string(prev))) {
std::string chr(ss.str().substr(0, ss.str().size() - strlen(prev)));
ss.str(std::string(""));
ss << chr;
}
if (base::utils::Str::endsWith(ss.str(), std::string(sfx))) {
std::string chr(ss.str().substr(0, ss.str().size() - strlen(sfx)));
ss.str(std::string(""));
ss << chr;
}
ss << sfx;
};
auto insert = [&](std::stringstream& ss, base::type::VerboseLevel level) {
if (!base::utils::hasFlag(el::LoggingFlag::DisableVModulesExtensions, *m_pFlags)) {
addSuffix(ss, ".h", nullptr);
m_modules.insert(std::make_pair(ss.str(), level));
addSuffix(ss, ".c", ".h");
m_modules.insert(std::make_pair(ss.str(), level));
addSuffix(ss, ".cpp", ".c");
m_modules.insert(std::make_pair(ss.str(), level));
addSuffix(ss, ".cc", ".cpp");
m_modules.insert(std::make_pair(ss.str(), level));
addSuffix(ss, ".cxx", ".cc");
m_modules.insert(std::make_pair(ss.str(), level));
addSuffix(ss, ".-inl.h", ".cxx");
m_modules.insert(std::make_pair(ss.str(), level));
addSuffix(ss, ".hxx", ".-inl.h");
m_modules.insert(std::make_pair(ss.str(), level));
addSuffix(ss, ".hpp", ".hxx");
m_modules.insert(std::make_pair(ss.str(), level));
addSuffix(ss, ".hh", ".hpp");
}
m_modules.insert(std::make_pair(ss.str(), level));
};
bool isMod = true;
bool isLevel = false;
std::stringstream ss;
int level = -1;
for (; *modules; ++modules) {
switch (*modules) {
case '=':
isLevel = true;
isMod = false;
break;
case ',':
isLevel = false;
isMod = true;
if (!ss.str().empty() && level != -1) {
insert(ss, level);
ss.str(std::string(""));
level = -1;
}
break;
default:
if (isMod) {
ss << *modules;
} else if (isLevel) {
if (isdigit(*modules)) {
level = static_cast<base::type::VerboseLevel>(*modules) - 48;
}
}
break;
}
}
if (!ss.str().empty() && level != -1) {
insert(ss, level);
}
}

bool allowed(base::type::VerboseLevel vlevel, const char* file) {
base::threading::ScopedLock scopedLock(lock());
if (m_modules.empty() || file == nullptr) {
return vlevel <= m_level;
} else {
std::map<std::string, base::type::VerboseLevel>::iterator it = m_modules.begin();
for (; it != m_modules.end(); ++it) {
if (base::utils::Str::wildCardMatch(file, it->first.c_str())) {
return vlevel <= it->second;
}
}
if (base::utils::hasFlag(el::LoggingFlag::AllowVerboseIfModuleNotSpecified, *m_pFlags)) {
return true;
}
return false;
}
}

inline const std::map<std::string, base::type::VerboseLevel>& modules(void) const {
return m_modules;
}

void setFromArgs(const base::utils::CommandLineArgs* commandLineArgs) {
if (commandLineArgs->hasParam("-v") || commandLineArgs->hasParam("--verbose") ||
commandLineArgs->hasParam("-V") || commandLineArgs->hasParam("--VERBOSE")) {
setLevel(base::consts::kMaxVerboseLevel);
} else if (commandLineArgs->hasParamWithValue("--v")) {
setLevel(atoi(commandLineArgs->getParamValue("--v")));
} else if (commandLineArgs->hasParamWithValue("--V")) {
setLevel(atoi(commandLineArgs->getParamValue("--V")));
} else if ((commandLineArgs->hasParamWithValue("-vmodule")) && vModulesEnabled()) {
setModules(commandLineArgs->getParamValue("-vmodule"));
} else if (commandLineArgs->hasParamWithValue("-VMODULE") && vModulesEnabled()) {
setModules(commandLineArgs->getParamValue("-VMODULE"));
}
}

inline bool vModulesEnabled(void) {
return !base::utils::hasFlag(el::LoggingFlag::DisableVModules, *m_pFlags);
}

private:
base::type::VerboseLevel m_level;
base::type::EnumType* m_pFlags;
std::map<std::string, base::type::VerboseLevel> m_modules;
};
}  
class LogMessage {
public:
LogMessage(el::Level level, const std::string& file, base::type::LineNumber line, const std::string& func,
base::type::VerboseLevel verboseLevel, Logger* logger) :
m_level(level), m_file(file), m_line(line), m_func(func),
m_verboseLevel(verboseLevel), m_logger(logger), m_message(logger->stream().str()) {
}
inline el::Level level(void) const {
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
el::Level m_level;
std::string m_file;
base::type::LineNumber m_line;
std::string m_func;
base::type::VerboseLevel m_verboseLevel;
el::Logger* m_logger;
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
Storage(const LogBuilderPtr& defaultLogBuilder, base::IWorker* asyncDispatchWorker) :
#else
explicit Storage(const LogBuilderPtr& defaultLogBuilder) :
#endif  
m_registeredHitCounters(new base::RegisteredHitCounters()),
m_registeredLoggers(new base::RegisteredLoggers(defaultLogBuilder)),
m_flags(0x0),
m_vRegistry(new base::VRegistry(0, &m_flags)),
#if ELPP_ASYNC_LOGGING
m_asyncLogQueue(new base::AsyncLogQueue()),
m_asyncDispatchWorker(asyncDispatchWorker),
#endif  
m_preRollOutCallback(base::defaultPreRollOutCallback) {
m_registeredLoggers->get(std::string(base::consts::kDefaultLoggerId));
Logger* performanceLogger = m_registeredLoggers->get(std::string(base::consts::kPerformanceLoggerId));
performanceLogger->configurations()->setGlobally(ConfigurationType::Format, std::string("%datetime %level %msg"));
performanceLogger->reconfigure();
#if defined(ELPP_SYSLOG)
Logger* sysLogLogger = m_registeredLoggers->get(std::string(base::consts::kSysLogLoggerId));
sysLogLogger->configurations()->setGlobally(ConfigurationType::Format, std::string("%level: %msg"));
sysLogLogger->reconfigure();
#endif 
addFlag(el::LoggingFlag::AllowVerboseIfModuleNotSpecified);
#if ELPP_ASYNC_LOGGING
installLogDispatchCallback<base::AsyncLogDispatchCallback>(std::string("AsyncLogDispatchCallback"));
#else
installLogDispatchCallback<base::DefaultLogDispatchCallback>(std::string("DefaultLogDispatchCallback"));
#endif  
#if defined(ELPP_FEATURE_ALL) || defined(ELPP_FEATURE_PERFORMANCE_TRACKING)
installPerformanceTrackingCallback<base::DefaultPerformanceTrackingCallback>
(std::string("DefaultPerformanceTrackingCallback"));
#endif 
ELPP_INTERNAL_INFO(1, "Easylogging++ has been initialized");
#if ELPP_ASYNC_LOGGING
m_asyncDispatchWorker->start();
#endif  
}

virtual ~Storage(void) {
ELPP_INTERNAL_INFO(4, "Destroying storage");
#if ELPP_ASYNC_LOGGING
ELPP_INTERNAL_INFO(5, "Replacing log dispatch callback to synchronous");
uninstallLogDispatchCallback<base::AsyncLogDispatchCallback>(std::string("AsyncLogDispatchCallback"));
installLogDispatchCallback<base::DefaultLogDispatchCallback>(std::string("DefaultLogDispatchCallback"));
ELPP_INTERNAL_INFO(5, "Destroying asyncDispatchWorker");
base::utils::safeDelete(m_asyncDispatchWorker);
ELPP_INTERNAL_INFO(5, "Destroying asyncLogQueue");
base::utils::safeDelete(m_asyncLogQueue);
#endif  
ELPP_INTERNAL_INFO(5, "Destroying registeredHitCounters");
base::utils::safeDelete(m_registeredHitCounters);
ELPP_INTERNAL_INFO(5, "Destroying registeredLoggers");
base::utils::safeDelete(m_registeredLoggers);
ELPP_INTERNAL_INFO(5, "Destroying vRegistry");
base::utils::safeDelete(m_vRegistry);
}

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

inline bool hasCustomFormatSpecifier(const char* formatSpecifier) {
base::threading::ScopedLock scopedLock(lock());
return std::find(m_customFormatSpecifiers.begin(), m_customFormatSpecifiers.end(),
formatSpecifier) != m_customFormatSpecifiers.end();
}

void installCustomFormatSpecifier(const CustomFormatSpecifier& customFormatSpecifier) {
if (hasCustomFormatSpecifier(customFormatSpecifier.formatSpecifier())) {
return;
}
base::threading::ScopedLock scopedLock(lock());
m_customFormatSpecifiers.push_back(customFormatSpecifier);
}

bool uninstallCustomFormatSpecifier(const char* formatSpecifier) {
base::threading::ScopedLock scopedLock(lock());
std::vector<CustomFormatSpecifier>::iterator it = std::find(m_customFormatSpecifiers.begin(),
m_customFormatSpecifiers.end(), formatSpecifier);
if (it != m_customFormatSpecifiers.end() && strcmp(formatSpecifier, it->formatSpecifier()) == 0) {
m_customFormatSpecifiers.erase(it);
return true;
}
return false;
}

const std::vector<CustomFormatSpecifier>* customFormatSpecifiers(void) const {
return &m_customFormatSpecifiers;
}

inline void setLoggingLevel(el::Level level) {
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
std::vector<CustomFormatSpecifier> m_customFormatSpecifiers;
el::Level m_loggingLevel;

friend class el::Helpers;
friend class el::base::DefaultLogDispatchCallback;
friend class el::LogBuilder;
friend class el::base::MessageBuilder;
friend class el::base::Writer;
friend class el::base::PerformanceTracker;
friend class el::base::LogDispatcher;

void setApplicationArguments(int argc, char** argv) {
m_commandLineArgs.setArgs(argc, argv);
m_vRegistry->setFromArgs(commandLineArgs());
#if !defined(ELPP_DISABLE_LOG_FILE_FROM_ARG)
if (m_commandLineArgs.hasParamWithValue(base::consts::kDefaultLogFileParam)) {
Configurations c;
c.setGlobally(ConfigurationType::Filename,
std::string(m_commandLineArgs.getParamValue(base::consts::kDefaultLogFileParam)));
registeredLoggers()->setDefaultConfigurations(c);
for (base::RegisteredLoggers::iterator it = registeredLoggers()->begin();
it != registeredLoggers()->end(); ++it) {
it->second->configure(c);
}
}
#endif  
#if defined(ELPP_LOGGING_FLAGS_FROM_ARG)
if (m_commandLineArgs.hasParamWithValue(base::consts::kLoggingFlagsParam)) {
m_flags = atoi(m_commandLineArgs.getParamValue(base::consts::kLoggingFlagsParam));
}
#endif  
}

inline void setApplicationArguments(int argc, const char** argv) {
setApplicationArguments(argc, const_cast<char**>(argv));
}
};
extern ELPP_EXPORT base::type::StoragePointer elStorage;
#define ELPP el::base::elStorage
class DefaultLogDispatchCallback : public LogDispatchCallback {
protected:
void handle(const LogDispatchData* data) {
m_data = data;
dispatch(m_data->logMessage()->logger()->logBuilder()->build(m_data->logMessage(),
m_data->dispatchAction() == base::DispatchAction::NormalLog));
}
private:
const LogDispatchData* m_data;
void dispatch(base::type::string_t&& logLine) {
if (m_data->dispatchAction() == base::DispatchAction::NormalLog) {
if (m_data->logMessage()->logger()->m_typedConfigurations->toFile(m_data->logMessage()->level())) {
base::type::fstream_t* fs = m_data->logMessage()->logger()->m_typedConfigurations->fileStream(
m_data->logMessage()->level());
if (fs != nullptr) {
fs->write(logLine.c_str(), logLine.size());
if (fs->fail()) {
ELPP_INTERNAL_ERROR("Unable to write log to file ["
<< m_data->logMessage()->logger()->m_typedConfigurations->filename(m_data->logMessage()->level()) << "].\n"
<< "Few possible reasons (could be something else):\n" << "      * Permission denied\n"
<< "      * Disk full\n" << "      * Disk is not writable", true);
} else {
if (ELPP->hasFlag(el::LoggingFlag::ImmediateFlush)
|| (m_data->logMessage()->logger()->isFlushNeeded(m_data->logMessage()->level()))) {
m_data->logMessage()->logger()->flush(m_data->logMessage()->level(), fs);
}
}
} else {
ELPP_INTERNAL_ERROR("Log file for [" << LevelHelper::convertToString(m_data->logMessage()->level()) << "] "
<< "has not been configured but [TO_FILE] is configured to TRUE. [Logger ID: "
<< m_data->logMessage()->logger()->id() << "]", false);
}
}
if (m_data->logMessage()->logger()->m_typedConfigurations->toStandardOutput(m_data->logMessage()->level())) {
if (ELPP->hasFlag(el::LoggingFlag::ColoredTerminalOutput))
m_data->logMessage()->logger()->logBuilder()->convertToColoredOutput(&logLine, m_data->logMessage()->level());
ELPP_COUT << ELPP_COUT_LINE(logLine);
}
}
#if defined(ELPP_SYSLOG)
else if (m_data->dispatchAction() == base::DispatchAction::SysLog) {
int sysLogPriority = 0;
if (m_data->logMessage()->level() == Level::Fatal)
sysLogPriority = LOG_EMERG;
else if (m_data->logMessage()->level() == Level::Error)
sysLogPriority = LOG_ERR;
else if (m_data->logMessage()->level() == Level::Warning)
sysLogPriority = LOG_WARNING;
else if (m_data->logMessage()->level() == Level::Info)
sysLogPriority = LOG_INFO;
else if (m_data->logMessage()->level() == Level::Debug)
sysLogPriority = LOG_DEBUG;
else
sysLogPriority = LOG_NOTICE;
#  if defined(ELPP_UNICODE)
char* line = base::utils::Str::wcharPtrToCharPtr(logLine.c_str());
syslog(sysLogPriority, "%s", line);
free(line);
#  else
syslog(sysLogPriority, "%s", logLine.c_str());
#  endif
}
#endif  
}
};
#if ELPP_ASYNC_LOGGING
class AsyncLogDispatchCallback : public LogDispatchCallback {
protected:
void handle(const LogDispatchData* data) {
base::type::string_t logLine = data->logMessage()->logger()->logBuilder()->build(data->logMessage(),
data->dispatchAction() == base::DispatchAction::NormalLog);
if (data->dispatchAction() == base::DispatchAction::NormalLog
&& data->logMessage()->logger()->typedConfigurations()->toStandardOutput(data->logMessage()->level())) {
if (ELPP->hasFlag(el::LoggingFlag::ColoredTerminalOutput))
data->logMessage()->logger()->logBuilder()->convertToColoredOutput(&logLine, data->logMessage()->level());
ELPP_COUT << ELPP_COUT_LINE(logLine);
}
if (data->logMessage()->logger()->typedConfigurations()->toFile(data->logMessage()->level())) {
ELPP->asyncLogQueue()->push(AsyncLogItem(*(data->logMessage()), *data, logLine));
}
}
};
class AsyncDispatchWorker : public base::IWorker, public base::threading::ThreadSafe {
public:
AsyncDispatchWorker() {
setContinueRunning(false);
}

virtual ~AsyncDispatchWorker() {
setContinueRunning(false);
ELPP_INTERNAL_INFO(6, "Stopping dispatch worker - Cleaning log queue");
clean();
ELPP_INTERNAL_INFO(6, "Log queue cleaned");
}

inline bool clean(void) {
std::mutex m;
std::unique_lock<std::mutex> lk(m);
cv.wait(lk, [] { return !ELPP->asyncLogQueue()->empty(); });
emptyQueue();
lk.unlock();
cv.notify_one();
return ELPP->asyncLogQueue()->empty();
}

inline void emptyQueue(void) {
while (!ELPP->asyncLogQueue()->empty()) {
AsyncLogItem data = ELPP->asyncLogQueue()->next();
handle(&data);
base::threading::msleep(100);
}
}

virtual inline void start(void) {
base::threading::msleep(5000); 
setContinueRunning(true);
std::thread t1(&AsyncDispatchWorker::run, this);
t1.join();
}

void handle(AsyncLogItem* logItem) {
LogDispatchData* data = logItem->data();
LogMessage* logMessage = logItem->logMessage();
Logger* logger = logMessage->logger();
base::TypedConfigurations* conf = logger->typedConfigurations();
base::type::string_t logLine = logItem->logLine();
if (data->dispatchAction() == base::DispatchAction::NormalLog) {
if (conf->toFile(logMessage->level())) {
base::type::fstream_t* fs = conf->fileStream(logMessage->level());
if (fs != nullptr) {
fs->write(logLine.c_str(), logLine.size());
if (fs->fail()) {
ELPP_INTERNAL_ERROR("Unable to write log to file ["
<< conf->filename(logMessage->level()) << "].\n"
<< "Few possible reasons (could be something else):\n" << "      * Permission denied\n"
<< "      * Disk full\n" << "      * Disk is not writable", true);
} else {
if (ELPP->hasFlag(el::LoggingFlag::ImmediateFlush) || (logger->isFlushNeeded(logMessage->level()))) {
logger->flush(logMessage->level(), fs);
}
}
} else {
ELPP_INTERNAL_ERROR("Log file for [" << LevelHelper::convertToString(logMessage->level()) << "] "
<< "has not been configured but [TO_FILE] is configured to TRUE. [Logger ID: " << logger->id() << "]", false);
}
}
}
#  if defined(ELPP_SYSLOG)
else if (data->dispatchAction() == base::DispatchAction::SysLog) {
int sysLogPriority = 0;
if (logMessage->level() == Level::Fatal)
sysLogPriority = LOG_EMERG;
else if (logMessage->level() == Level::Error)
sysLogPriority = LOG_ERR;
else if (logMessage->level() == Level::Warning)
sysLogPriority = LOG_WARNING;
else if (logMessage->level() == Level::Info)
sysLogPriority = LOG_INFO;
else if (logMessage->level() == Level::Debug)
sysLogPriority = LOG_DEBUG;
else
sysLogPriority = LOG_NOTICE;
#      if defined(ELPP_UNICODE)
char* line = base::utils::Str::wcharPtrToCharPtr(logLine.c_str());
syslog(sysLogPriority, "%s", line);
free(line);
#      else
syslog(sysLogPriority, "%s", logLine.c_str());
#      endif
}
#  endif  
}

void run(void) {
while (continueRunning()) {
emptyQueue();
base::threading::msleep(10); 
}
}

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
base::type::string_t build(const el::LogMessage* logMessage, bool appendNewLine) const {
base::TypedConfigurations* tc = logMessage->logger()->typedConfigurations();
const base::LogFormat* logFormat = &tc->logFormat(logMessage->level());
base::type::string_t logLine = logFormat->format();
char buff[base::consts::kSourceFilenameMaxLength + base::consts::kSourceLineMaxLength] = "";
const char* bufLim = buff + sizeof(buff);
if (logFormat->hasFlag(base::FormatFlags::AppName)) {
base::utils::Str::replaceFirstWithEscape(logLine, base::consts::kAppNameFormatSpecifier,
logMessage->logger()->parentApplicationName());
}
if (logFormat->hasFlag(base::FormatFlags::ThreadId)) {
base::utils::Str::replaceFirstWithEscape(logLine, base::consts::kThreadIdFormatSpecifier,
base::threading::getCurrentThreadId());
}
if (logFormat->hasFlag(base::FormatFlags::DateTime)) {
base::utils::Str::replaceFirstWithEscape(logLine, base::consts::kDateTimeFormatSpecifier,
base::utils::DateTime::getDateTime(logFormat->dateTimeFormat().c_str(),
&tc->millisecondsWidth(logMessage->level())));
}
if (logFormat->hasFlag(base::FormatFlags::Function)) {
base::utils::Str::replaceFirstWithEscape(logLine, base::consts::kLogFunctionFormatSpecifier, logMessage->func());
}
if (logFormat->hasFlag(base::FormatFlags::File)) {
base::utils::Str::clearBuff(buff, base::consts::kSourceFilenameMaxLength);
base::utils::File::buildStrippedFilename(logMessage->file().c_str(), buff);
base::utils::Str::replaceFirstWithEscape(logLine, base::consts::kLogFileFormatSpecifier, std::string(buff));
}
if (logFormat->hasFlag(base::FormatFlags::FileBase)) {
base::utils::Str::clearBuff(buff, base::consts::kSourceFilenameMaxLength);
base::utils::File::buildBaseFilename(logMessage->file(), buff);
base::utils::Str::replaceFirstWithEscape(logLine, base::consts::kLogFileBaseFormatSpecifier, std::string(buff));
}
if (logFormat->hasFlag(base::FormatFlags::Line)) {
char* buf = base::utils::Str::clearBuff(buff, base::consts::kSourceLineMaxLength);
buf = base::utils::Str::convertAndAddToBuff(logMessage->line(), base::consts::kSourceLineMaxLength, buf, bufLim, false);
base::utils::Str::replaceFirstWithEscape(logLine, base::consts::kLogLineFormatSpecifier, std::string(buff));
}
if (logFormat->hasFlag(base::FormatFlags::Location)) {
char* buf = base::utils::Str::clearBuff(buff,
base::consts::kSourceFilenameMaxLength + base::consts::kSourceLineMaxLength);
base::utils::File::buildStrippedFilename(logMessage->file().c_str(), buff);
buf = base::utils::Str::addToBuff(buff, buf, bufLim);
buf = base::utils::Str::addToBuff(":", buf, bufLim);
buf = base::utils::Str::convertAndAddToBuff(logMessage->line(),  base::consts::kSourceLineMaxLength, buf, bufLim,
false);
base::utils::Str::replaceFirstWithEscape(logLine, base::consts::kLogLocationFormatSpecifier, std::string(buff));
}
if (logMessage->level() == el::Level::Verbose && logFormat->hasFlag(base::FormatFlags::VerboseLevel)) {
char* buf = base::utils::Str::clearBuff(buff, 1);
buf = base::utils::Str::convertAndAddToBuff(logMessage->verboseLevel(), 1, buf, bufLim, false);
base::utils::Str::replaceFirstWithEscape(logLine, base::consts::kVerboseLevelFormatSpecifier, std::string(buff));
}
if (logFormat->hasFlag(base::FormatFlags::LogMessage)) {
base::utils::Str::replaceFirstWithEscape(logLine, base::consts::kMessageFormatSpecifier, logMessage->message());
}
#if !defined(ELPP_DISABLE_CUSTOM_FORMAT_SPECIFIERS)
for (std::vector<CustomFormatSpecifier>::const_iterator it = ELPP->customFormatSpecifiers()->begin();
it != ELPP->customFormatSpecifiers()->end(); ++it) {
std::string fs(it->formatSpecifier());
base::type::string_t wcsFormatSpecifier(fs.begin(), fs.end());
base::utils::Str::replaceFirstWithEscape(logLine, wcsFormatSpecifier, std::string(it->resolver()()));
}
#endif  
if (appendNewLine) logLine += ELPP_LITERAL("\n");
return logLine;
}
};
class LogDispatcher : base::NoCopy {
public:
LogDispatcher(bool proceed, el::LogMessage&& logMessage, base::DispatchAction dispatchAction) :
m_proceed(proceed),
m_logMessage(std::move(logMessage)),
m_dispatchAction(std::move(dispatchAction)) {
}

void dispatch(void) {
if (m_proceed && m_dispatchAction == base::DispatchAction::None) {
m_proceed = false;
}
if (!m_proceed) {
return;
}
base::threading::ScopedLock scopedLock(ELPP->lock());
base::TypedConfigurations* tc = m_logMessage.logger()->m_typedConfigurations;
if (ELPP->hasFlag(el::LoggingFlag::StrictLogFileSizeCheck)) {
tc->validateFileRolling(m_logMessage.level(), ELPP->preRollOutCallback());
}
LogDispatchCallback* callback = nullptr;
LogDispatchData data;
#ifdef HAS_RANGE_FOR
for (const std::pair<std::string, base::type::LogDispatchCallbackPtr>& h
: ELPP->m_logDispatchCallbacks) {
callback = h.second.get();
#else
for (std::map<std::string, base::type::LogDispatchCallbackPtr>::iterator
h = ELPP->m_logDispatchCallbacks.begin();
h != ELPP->m_logDispatchCallbacks.end(); ++h) {
callback = h->second.get();
#endif
if (callback != nullptr && callback->enabled()) {
data.setLogMessage(&m_logMessage);
data.setDispatchAction(m_dispatchAction);
callback->acquireLock();
callback->handle(&data);
callback->releaseLock();
}
}
}

private:
bool m_proceed;
el::LogMessage m_logMessage;
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
void initialize(Logger* logger) {
m_logger = logger;
m_containerLogSeperator = ELPP->hasFlag(el::LoggingFlag::NewLineForContainer) ?
ELPP_LITERAL("\n    ") : ELPP_LITERAL(", ");
}

#  define ELPP_SIMPLE_LOG(LOG_TYPE)\
inline MessageBuilder& operator<<(LOG_TYPE msg) {\
m_logger->stream() << msg;\
if (ELPP->hasFlag(el::LoggingFlag::AutoSpacing)) {\
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
inline MessageBuilder& operator<<(const wchar_t* msg) {
if (msg == nullptr) {
m_logger->stream() << base::consts::kNullPointer;
return *this;
}
#  if defined(ELPP_UNICODE)
m_logger->stream() << msg;
#  else
char* buff_ = base::utils::Str::wcharPtrToCharPtr(msg);
m_logger->stream() << buff_;
free(buff_);
#  endif
if (ELPP->hasFlag(el::LoggingFlag::AutoSpacing)) {
m_logger->stream() << " ";
}
return *this;
}
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
inline MessageBuilder& operator<<(const std::pair<First, Second>& pair_) {
m_logger->stream() << ELPP_LITERAL("(");
operator << (static_cast<First>(pair_.first));
m_logger->stream() << ELPP_LITERAL(", ");
operator << (static_cast<Second>(pair_.second));
m_logger->stream() << ELPP_LITERAL(")");
return *this;
}
template <std::size_t Size>
inline MessageBuilder& operator<<(const std::bitset<Size>& bitset_) {
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
inline MessageBuilder& operator<<(const QPair<First, Second>& pair_) {
m_logger->stream() << ELPP_LITERAL("(");
operator << (static_cast<First>(pair_.first));
m_logger->stream() << ELPP_LITERAL(", ");
operator << (static_cast<Second>(pair_.second));
m_logger->stream() << ELPP_LITERAL(")");
return *this;
}
template <typename K, typename V>
inline MessageBuilder& operator<<(const QMap<K, V>& map_) {
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
inline MessageBuilder& operator<<(const QHash<K, V>& hash_) {
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
inline MessageBuilder& writeIterator(Iterator begin_, Iterator end_, std::size_t size_) {
m_logger->stream() << ELPP_LITERAL("[");
for (std::size_t i = 0; begin_ != end_ && i < base::consts::kMaxLogPerContainer; ++i, ++begin_) {
operator << (*begin_);
m_logger->stream() << ((i < size_ - 1) ? m_containerLogSeperator : ELPP_LITERAL(""));
}
if (begin_ != end_) {
m_logger->stream() << ELPP_LITERAL("...");
}
m_logger->stream() << ELPP_LITERAL("]");
if (ELPP->hasFlag(el::LoggingFlag::AutoSpacing)) {
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
Writer(el::Level level, const char* file, base::type::LineNumber line,
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

Writer& construct(Logger* logger, bool needLock = true) {
m_logger = logger;
initializeLogger(logger->id(), false, needLock);
m_messageBuilder.initialize(m_logger);
return *this;
}

Writer& construct(int count, const char* loggerIds, ...) {
if (ELPP->hasFlag(el::LoggingFlag::MultiLoggerSupport)) {
va_list loggersList;
va_start(loggersList, loggerIds);
const char* id = loggerIds;
for (int i = 0; i < count; ++i) {
m_loggerIds.push_back(std::string(id));
id = va_arg(loggersList, const char*);
}
va_end(loggersList);
initializeLogger(m_loggerIds.at(0));
} else {
initializeLogger(std::string(loggerIds));
}
m_messageBuilder.initialize(m_logger);
return *this;
}
protected:
el::Level m_level;
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

void initializeLogger(const std::string& loggerId, bool lookup = true, bool needLock = true) {
if (lookup) {
m_logger = ELPP->registeredLoggers()->get(loggerId, ELPP->hasFlag(el::LoggingFlag::CreateLoggerAutomatically));
}
if (m_logger == nullptr) {
ELPP->acquireLock();
if (!ELPP->registeredLoggers()->has(std::string(base::consts::kDefaultLoggerId))) {
ELPP->registeredLoggers()->get(std::string(base::consts::kDefaultLoggerId));
}
ELPP->releaseLock();  
Writer(el::Level::Debug, m_file, m_line, m_func).construct(1, base::consts::kDefaultLoggerId)
<< "Logger [" << loggerId << "] is not registered yet!";
m_proceed = false;
} else {
if (needLock) {
m_logger->acquireLock();  
}
if (ELPP->hasFlag(el::LoggingFlag::HierarchicalLogging)) {
m_proceed = m_level == el::Level::Verbose ? m_logger->enabled(m_level) :
LevelHelper::castToInt(m_level) >= LevelHelper::castToInt(ELPP->m_loggingLevel);
} else {
m_proceed = m_logger->enabled(m_level);
}
}
}

void processDispatch() {
#if ELPP_LOGGING_ENABLED
if (ELPP->hasFlag(el::LoggingFlag::MultiLoggerSupport)) {
bool firstDispatched = false;
base::type::string_t logMessage;
std::size_t i = 0;
do {
if (m_proceed) {
if (firstDispatched) {
m_logger->stream() << logMessage;
} else {
firstDispatched = true;
if (m_loggerIds.size() > 1) {
logMessage = m_logger->stream().str();
}
}
triggerDispatch();
} else if (m_logger != nullptr) {
m_logger->stream().str(ELPP_LITERAL(""));
m_logger->releaseLock();
}
if (i + 1 < m_loggerIds.size()) {
initializeLogger(m_loggerIds.at(i + 1));
}
} while (++i < m_loggerIds.size());
} else {
if (m_proceed) {
triggerDispatch();
} else if (m_logger != nullptr) {
m_logger->stream().str(ELPP_LITERAL(""));
m_logger->releaseLock();
}
}
#else
if (m_logger != nullptr) {
m_logger->stream().str(ELPP_LITERAL(""));
m_logger->releaseLock();
}
#endif 
}

void triggerDispatch(void) {
if (m_proceed) {
base::LogDispatcher(m_proceed, el::LogMessage(m_level, m_file, m_line, m_func, m_verboseLevel,
m_logger), m_dispatchAction).dispatch();
}
if (m_logger != nullptr) {
m_logger->stream().str(ELPP_LITERAL(""));
m_logger->releaseLock();
}
if (m_proceed && m_level == el::Level::Fatal
&& !ELPP->hasFlag(el::LoggingFlag::DisableApplicationAbortOnFatalLog)) {
base::Writer(el::Level::Warning, m_file, m_line, m_func).construct(1, base::consts::kDefaultLoggerId)
<< "Aborting application. Reason: Fatal log at [" << m_file << ":" << m_line << "]";
std::stringstream reasonStream;
reasonStream << "Fatal log at [" << m_file << ":" << m_line << "]"
<< " If you wish to disable 'abort on fatal log' please use "
<< "el::Helpers::addFlag(el::LoggingFlag::DisableApplicationAbortOnFatalLog)";
base::utils::abort(1, reasonStream.str());
}
m_proceed = false;
}
};
class PErrorWriter : public base::Writer {
public:
PErrorWriter(el::Level level, const char* file, base::type::LineNumber line,
const char* func, base::DispatchAction dispatchAction = base::DispatchAction::NormalLog,
base::type::VerboseLevel verboseLevel = 0) :
base::Writer(level, file, line, func, dispatchAction, verboseLevel) {
}

virtual ~PErrorWriter(void) {
if (m_proceed) {
#if ELPP_COMPILER_MSVC
char buff[256];
strerror_s(buff, 256, errno);
m_logger->stream() << ": " << buff << " [" << errno << "]";
#else
m_logger->stream() << ": " << strerror(errno) << " [" << errno << "]";
#endif
}
}
};
}  
#if ELPP_VARIADIC_TEMPLATES_SUPPORTED
template <typename T, typename... Args>
void Logger::log_(el::Level level, int vlevel, const char* s, const T& value, const Args&... args) {
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
inline void Logger::log_(el::Level level, int vlevel, const T& log) {
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
void Logger::log(el::Level level, const char* s, const T& value, const Args&... args) {
base::threading::ScopedLock scopedLock(lock());
log_(level, 0, s, value, args...);
}
template <typename T>
inline void Logger::log(el::Level level, const T& log) {
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
#undef ELPP_CURR_FILE_PERFORMANCE_LOGGER
#if defined(ELPP_PERFORMANCE_LOGGER)
#  define ELPP_CURR_FILE_PERFORMANCE_LOGGER ELPP_PERFORMANCE_LOGGER
#else
#  define ELPP_CURR_FILE_PERFORMANCE_LOGGER el::base::consts::kPerformanceLoggerId
#endif
#if defined(ELPP_FEATURE_ALL) || defined(ELPP_FEATURE_PERFORMANCE_TRACKING)
class PerformanceTrackingData {
public:
ENUMCLASS DataType : base::type::EnumType {
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
const std::string& loggerId = std::string(ELPP_CURR_FILE_PERFORMANCE_LOGGER),
bool scopedLog = true, Level level = base::consts::kPerformanceTrackerDefaultLevel) :
m_blockName(blockName), m_timestampUnit(timestampUnit), m_loggerId(loggerId), m_scopedLog(scopedLog),
m_level(level), m_hasChecked(false), m_lastCheckpointId(std::string()), m_enabled(false) {
#if !defined(ELPP_DISABLE_PERFORMANCE_TRACKING) && ELPP_LOGGING_ENABLED
el::Logger* loggerPtr = ELPP->registeredLoggers()->get(loggerId, false);
m_enabled = loggerPtr != nullptr && loggerPtr->m_typedConfigurations->performanceTracking(m_level);
if (m_enabled) {
base::utils::DateTime::gettimeofday(&m_startTime);
}
#endif  
}
PerformanceTracker(const PerformanceTracker& t) :
m_blockName(t.m_blockName), m_timestampUnit(t.m_timestampUnit), m_loggerId(t.m_loggerId), m_scopedLog(t.m_scopedLog),
m_level(t.m_level), m_hasChecked(t.m_hasChecked), m_lastCheckpointId(t.m_lastCheckpointId), m_enabled(t.m_enabled),
m_startTime(t.m_startTime), m_endTime(t.m_endTime), m_lastCheckpointTime(t.m_lastCheckpointTime) {
}
virtual ~PerformanceTracker(void) {
#if !defined(ELPP_DISABLE_PERFORMANCE_TRACKING) && ELPP_LOGGING_ENABLED
if (m_enabled) {
base::threading::ScopedLock scopedLock(lock());
if (m_scopedLog) {
base::utils::DateTime::gettimeofday(&m_endTime);
base::type::string_t formattedTime = getFormattedTimeTaken();
PerformanceTrackingData data(PerformanceTrackingData::DataType::Complete);
data.init(this);
data.m_formattedTimeTaken = formattedTime;
PerformanceTrackingCallback* callback = nullptr;
for (const std::pair<std::string, base::type::PerformanceTrackingCallbackPtr>& h
: ELPP->m_performanceTrackingCallbacks) {
callback = h.second.get();
if (callback != nullptr && callback->enabled()) {
callback->acquireLock();
callback->handle(&data);
callback->releaseLock();
}
}
}
}
#endif  
}
void checkpoint(const std::string& id = std::string(), const char* file = __FILE__,
base::type::LineNumber line = __LINE__,
const char* func = "") {
#if !defined(ELPP_DISABLE_PERFORMANCE_TRACKING) && ELPP_LOGGING_ENABLED
if (m_enabled) {
base::threading::ScopedLock scopedLock(lock());
base::utils::DateTime::gettimeofday(&m_endTime);
base::type::string_t formattedTime = m_hasChecked ? getFormattedTimeTaken(m_lastCheckpointTime) : ELPP_LITERAL("");
PerformanceTrackingData data(PerformanceTrackingData::DataType::Checkpoint);
data.init(this);
data.m_checkpointId = id;
data.m_file = file;
data.m_line = line;
data.m_func = func;
data.m_formattedTimeTaken = formattedTime;
PerformanceTrackingCallback* callback = nullptr;
for (const std::pair<std::string, base::type::PerformanceTrackingCallbackPtr>& h
: ELPP->m_performanceTrackingCallbacks) {
callback = h.second.get();
if (callback != nullptr && callback->enabled()) {
callback->acquireLock();
callback->handle(&data);
callback->releaseLock();
}
}
base::utils::DateTime::gettimeofday(&m_lastCheckpointTime);
m_hasChecked = true;
m_lastCheckpointId = id;
}
#endif  
ELPP_UNUSED(id);
ELPP_UNUSED(file);
ELPP_UNUSED(line);
ELPP_UNUSED(func);
}
inline el::Level level(void) const {
return m_level;
}
private:
std::string m_blockName;
base::TimestampUnit m_timestampUnit;
std::string m_loggerId;
bool m_scopedLog;
el::Level m_level;
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

const base::type::string_t getFormattedTimeTaken(struct timeval startTime) const {
if (ELPP->hasFlag(el::LoggingFlag::FixedTimeFormat)) {
base::type::stringstream_t ss;
ss << base::utils::DateTime::getTimeDifference(m_endTime,
startTime, m_timestampUnit) << " " << base::consts::kTimeFormats[static_cast<base::type::EnumType>
(m_timestampUnit)].unit;
return ss.str();
}
return base::utils::DateTime::formatTime(base::utils::DateTime::getTimeDifference(m_endTime,
startTime, m_timestampUnit), m_timestampUnit);
}

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
if (!ELPP->hasFlag(el::LoggingFlag::DisablePerformanceTrackingCheckpointComparison)
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
static const std::size_t kMaxStack = 64;
static const std::size_t kStackStart = 2;  
class StackTraceEntry {
public:
StackTraceEntry(std::size_t index, const char* loc, const char* demang, const char* hex, const char* addr) {
m_index = index;
m_location = std::string(loc);
m_demangled = std::string(demang);
m_hex = std::string(hex);
m_addr = std::string(addr);
}
StackTraceEntry(std::size_t index, char* loc) {
m_index = index;
m_location = std::string(loc);
}
std::size_t m_index;
std::string m_location;
std::string m_demangled;
std::string m_hex;
std::string m_addr;
friend std::ostream& operator<<(std::ostream& ss, const StackTraceEntry& si) {
ss << "[" << si.m_index << "] " << si.m_location << (si.m_demangled.empty() ? "" : ":") << si.m_demangled
<< (si.m_hex.empty() ? "" : "+") << si.m_hex << si.m_addr;
return ss;
}

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

friend inline std::ostream& operator<<(std::ostream& os, const StackTrace& st) {
std::vector<StackTraceEntry>::const_iterator it = st.m_stack.begin();
while (it != st.m_stack.end()) {
os << "    " << *it++ << "\n";
}
return os;
}

private:
std::vector<StackTraceEntry> m_stack;

void generateNew(void) {
#if ELPP_STACKTRACE
m_stack.clear();
void* stack[kMaxStack];
std::size_t size = backtrace(stack, kMaxStack);
char** strings = backtrace_symbols(stack, size);
if (size > kStackStart) {  
for (std::size_t i = kStackStart; i < size; ++i) {
char* mangName = nullptr;
char* hex = nullptr;
char* addr = nullptr;
for (char* c = strings[i]; *c; ++c) {
switch (*c) {
case '(':
mangName = c;
break;
case '+':
hex = c;
break;
case ')':
addr = c;
break;
default:
break;
}
}
if (mangName != nullptr && hex != nullptr && addr != nullptr && mangName < hex) {
*mangName++ = '\0';
*hex++ = '\0';
*addr++ = '\0';
int status = 0;
char* demangName = abi::__cxa_demangle(mangName, 0, 0, &status);
if (status == 0) {
StackTraceEntry entry(i - 1, strings[i], demangName, hex, addr);
m_stack.push_back(entry);
} else {
StackTraceEntry entry(i - 1, strings[i], mangName, hex, addr);
m_stack.push_back(entry);
}
free(demangName);
} else {
StackTraceEntry entry(i - 1, strings[i]);
m_stack.push_back(entry);
}
}
}
free(strings);
#else
ELPP_INTERNAL_INFO(1, "Stacktrace generation not supported for selected compiler");
#endif  
}
};
static std::string crashReason(int sig) {
std::stringstream ss;
bool foundReason = false;
for (int i = 0; i < base::consts::kCrashSignalsCount; ++i) {
if (base::consts::kCrashSignals[i].numb == sig) {
ss << "Application has crashed due to [" << base::consts::kCrashSignals[i].name << "] signal";
if (ELPP->hasFlag(el::LoggingFlag::LogDetailedCrashReason)) {
ss << std::endl <<
"    " << base::consts::kCrashSignals[i].brief << std::endl <<
"    " << base::consts::kCrashSignals[i].detail;
}
foundReason = true;
}
}
if (!foundReason) {
ss << "Application has crashed due to unknown signal [" << sig << "]";
}
return ss.str();
}
static void logCrashReason(int sig, bool stackTraceIfAvailable, el::Level level, const char* logger) {
std::stringstream ss;
ss << "CRASH HANDLED; ";
ss << crashReason(sig);
#if ELPP_STACKTRACE
if (stackTraceIfAvailable) {
ss << std::endl << "    ======= Backtrace: =========" << std::endl << base::debug::StackTrace();
}
#else
ELPP_UNUSED(stackTraceIfAvailable);
#endif  
ELPP_WRITE_LOG(el::base::Writer, level, base::DispatchAction::NormalLog, logger) << ss.str();
}
static inline void crashAbort(int sig) {
base::utils::abort(sig);
}
static inline void defaultCrashHandler(int sig) {
base::debug::logCrashReason(sig, true, el::Level::Fatal, base::consts::kDefaultLoggerId);
base::debug::crashAbort(sig);
}
class CrashHandler : base::NoCopy {
public:
typedef void (*Handler)(int);

explicit CrashHandler(bool useDefault) {
if (useDefault) {
setHandler(defaultCrashHandler);
}
}
explicit CrashHandler(const Handler& cHandler) {
setHandler(cHandler);
}
void setHandler(const Handler& cHandler) {
m_handler = cHandler;
#if defined(ELPP_HANDLE_SIGABRT)
int i = 0;  
#else
int i = 1;
#endif  
for (; i < base::consts::kCrashSignalsCount; ++i) {
m_handler = signal(base::consts::kCrashSignals[i].numb, cHandler);
}
}

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
#if defined(ELPP_FEATURE_ALL) || defined(ELPP_FEATURE_CRASH_LOG)
static inline void setCrashHandler(const el::base::debug::CrashHandler::Handler& crashHandler) {
el::elCrashHandler.setHandler(crashHandler);
}
static void crashAbort(int sig, const char* sourceFile = "", unsigned int long line = 0) {
std::stringstream ss;
ss << base::debug::crashReason(sig).c_str();
ss << " - [Called el::Helpers::crashAbort(" << sig << ")]";
if (sourceFile != nullptr && strlen(sourceFile) > 0) {
ss << " - Source: " << sourceFile;
if (line > 0)
ss << ":" << line;
else
ss << " (line number not specified)";
}
base::utils::abort(sig, ss.str());
}
static inline void logCrashReason(int sig, bool stackTraceIfAvailable = false,
el::Level level = Level::Fatal, const char* logger = base::consts::kDefaultLoggerId) {
el::base::debug::logCrashReason(sig, stackTraceIfAvailable, level, logger);
}
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
static inline Logger* getLogger(const std::string& identity, bool registerIfNotAvailable = true) {
base::threading::ScopedLock scopedLock(ELPP->lock());
return ELPP->registeredLoggers()->get(identity, registerIfNotAvailable);
}
static inline void setDefaultLogBuilder(el::LogBuilderPtr& logBuilderPtr) {
ELPP->registeredLoggers()->setDefaultLogBuilder(logBuilderPtr);
}
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
static inline bool unregisterLogger(const std::string& identity) {
base::threading::ScopedLock scopedLock(ELPP->lock());
return ELPP->registeredLoggers()->remove(identity);
}
static inline bool hasLogger(const std::string& identity) {
base::threading::ScopedLock scopedLock(ELPP->lock());
return ELPP->registeredLoggers()->has(identity);
}
static inline Logger* reconfigureLogger(Logger* logger, const Configurations& configurations) {
if (!logger) return nullptr;
logger->configure(configurations);
return logger;
}
static inline Logger* reconfigureLogger(const std::string& identity, const Configurations& configurations) {
return Loggers::reconfigureLogger(Loggers::getLogger(identity), configurations);
}
static inline Logger* reconfigureLogger(const std::string& identity, ConfigurationType configurationType,
const std::string& value) {
Logger* logger = Loggers::getLogger(identity);
if (logger == nullptr) {
return nullptr;
}
logger->configurations()->set(Level::Global, configurationType, value);
logger->reconfigure();
return logger;
}
static inline void reconfigureAllLoggers(const Configurations& configurations) {
for (base::RegisteredLoggers::iterator it = ELPP->registeredLoggers()->begin();
it != ELPP->registeredLoggers()->end(); ++it) {
Loggers::reconfigureLogger(it->second, configurations);
}
}
static inline void reconfigureAllLoggers(ConfigurationType configurationType, const std::string& value) {
reconfigureAllLoggers(Level::Global, configurationType, value);
}
static inline void reconfigureAllLoggers(Level level, ConfigurationType configurationType,
const std::string& value) {
for (base::RegisteredLoggers::iterator it = ELPP->registeredLoggers()->begin();
it != ELPP->registeredLoggers()->end(); ++it) {
Logger* logger = it->second;
logger->configurations()->set(level, configurationType, value);
logger->reconfigure();
}
}
static inline void setDefaultConfigurations(const Configurations& configurations,
bool reconfigureExistingLoggers = false) {
ELPP->registeredLoggers()->setDefaultConfigurations(configurations);
if (reconfigureExistingLoggers) {
Loggers::reconfigureAllLoggers(configurations);
}
}
static inline const Configurations* defaultConfigurations(void) {
return ELPP->registeredLoggers()->defaultConfigurations();
}
static inline const base::LogStreamsReferenceMap* logStreamsReference(void) {
return ELPP->registeredLoggers()->logStreamsReference();
}
static base::TypedConfigurations defaultTypedConfigurations(void) {
return base::TypedConfigurations(
ELPP->registeredLoggers()->defaultConfigurations(),
ELPP->registeredLoggers()->logStreamsReference());
}
static inline std::vector<std::string>* populateAllLoggerIds(std::vector<std::string>* targetList) {
targetList->clear();
for (base::RegisteredLoggers::iterator it = ELPP->registeredLoggers()->list().begin();
it != ELPP->registeredLoggers()->list().end(); ++it) {
targetList->push_back(it->first);
}
return targetList;
}
static void configureFromGlobal(const char* globalConfigurationFilePath) {
std::ifstream gcfStream(globalConfigurationFilePath, std::ifstream::in);
ELPP_ASSERT(gcfStream.is_open(), "Unable to open global configuration file [" << globalConfigurationFilePath
<< "] for parsing.");
std::string line = std::string();
std::stringstream ss;
Logger* logger = nullptr;
auto configure = [&](void) {
ELPP_INTERNAL_INFO(8, "Configuring logger: '" << logger->id() << "' with configurations \n" << ss.str()
<< "\n--------------");
Configurations c;
c.parseFromText(ss.str());
logger->configure(c);
};
while (gcfStream.good()) {
std::getline(gcfStream, line);
ELPP_INTERNAL_INFO(1, "Parsing line: " << line);
base::utils::Str::trim(line);
if (Configurations::Parser::isComment(line)) continue;
Configurations::Parser::ignoreComments(&line);
base::utils::Str::trim(line);
if (line.size() > 2 && base::utils::Str::startsWith(line, std::string(base::consts::kConfigurationLoggerId))) {
if (!ss.str().empty() && logger != nullptr) {
configure();
}
ss.str(std::string(""));
line = line.substr(2);
base::utils::Str::trim(line);
if (line.size() > 1) {
ELPP_INTERNAL_INFO(1, "Getting logger: '" << line << "'");
logger = getLogger(line);
}
} else {
ss << line << "\n";
}
}
if (!ss.str().empty() && logger != nullptr) {
configure();
}
}
static inline bool configureFromArg(const char* argKey) {
#if defined(ELPP_DISABLE_CONFIGURATION_FROM_PROGRAM_ARGS)
ELPP_UNUSED(argKey);
#else
if (!Helpers::commandLineArgs()->hasParamWithValue(argKey)) {
return false;
}
configureFromGlobal(Helpers::commandLineArgs()->getParamValue(argKey));
#endif  
return true;
}
static inline void flushAll(void) {
ELPP->registeredLoggers()->flushAll();
}
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
static inline void setLoggingLevel(Level level) {
ELPP->setLoggingLevel(level);
}
static inline void setVerboseLevel(base::type::VerboseLevel level) {
ELPP->vRegistry()->setLevel(level);
}
static inline base::type::VerboseLevel verboseLevel(void) {
return ELPP->vRegistry()->level();
}
static inline void setVModules(const char* modules) {
if (ELPP->vRegistry()->vModulesEnabled()) {
ELPP->vRegistry()->setModules(modules);
}
}
static inline void clearVModules(void) {
ELPP->vRegistry()->clearModules();
}
};
class VersionInfo : base::StaticClass {
public:
static inline const std::string version(void) {
return std::string("9.89");
}
static inline const std::string releaseDate(void) {
return std::string("31-12-2016 1531hrs");
}
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
#if defined(ELPP_DISABLE_DEFAULT_CRASH_HANDLING)
#  define ELPP_USE_DEF_CRASH_HANDLER false
#else
#  define ELPP_USE_DEF_CRASH_HANDLER true
#endif  
#define ELPP_CRASH_HANDLER_INIT
#define ELPP_INIT_EASYLOGGINGPP(val) \
ELPP_INITI_BASIC_DECLR \
namespace el { \
namespace base { \
el::base::type::StoragePointer elStorage(val); \
} \
el::base::debug::CrashHandler elCrashHandler(ELPP_USE_DEF_CRASH_HANDLER); \
}

#if ELPP_ASYNC_LOGGING
#  define INITIALIZE_EASYLOGGINGPP\
ELPP_INIT_EASYLOGGINGPP(new el::base::Storage(el::LogBuilderPtr(new el::base::DefaultLogBuilder()),\
new el::base::AsyncDispatchWorker()))\

#else
#  define INITIALIZE_EASYLOGGINGPP ELPP_INIT_EASYLOGGINGPP(new el::base::Storage(el::LogBuilderPtr(new el::base::DefaultLogBuilder())))
#endif  
#define INITIALIZE_NULL_EASYLOGGINGPP \
ELPP_INITI_BASIC_DECLR \
namespace el { \
namespace base { \
el::base::type::StoragePointer elStorage; \
} \
el::base::debug::CrashHandler elCrashHandler(ELPP_USE_DEF_CRASH_HANDLER); \
}
#define SHARE_EASYLOGGINGPP(initializedStorage) \
namespace el { \
namespace base { \
el::base::type::StoragePointer elStorage(initializedStorage); \
} \
el::base::debug::CrashHandler elCrashHandler(ELPP_USE_DEF_CRASH_HANDLER); \
}

#if defined(ELPP_UNICODE)
#  define START_EASYLOGGINGPP(argc, argv) el::Helpers::setArgs(argc, argv); std::locale::global(std::locale(""))
#else
#  define START_EASYLOGGINGPP(argc, argv) el::Helpers::setArgs(argc, argv)
#endif  
#endif 
