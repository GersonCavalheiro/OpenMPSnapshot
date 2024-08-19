


#ifndef GTEST_INCLUDE_GTEST_GTEST_H_
#define GTEST_INCLUDE_GTEST_GTEST_H_

#include <cstddef>
#include <limits>
#include <memory>
#include <ostream>
#include <type_traits>
#include <vector>



#ifndef GTEST_INCLUDE_GTEST_INTERNAL_GTEST_INTERNAL_H_
#define GTEST_INCLUDE_GTEST_INTERNAL_GTEST_INTERNAL_H_



#ifndef GTEST_INCLUDE_GTEST_INTERNAL_GTEST_PORT_H_
#define GTEST_INCLUDE_GTEST_INTERNAL_GTEST_PORT_H_






#include <ctype.h>   
#include <stddef.h>  
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cerrno>
#include <cstdint>
#include <limits>
#include <type_traits>

#ifndef _WIN32_WCE
# include <sys/types.h>
# include <sys/stat.h>
#endif  

#if defined __APPLE__
# include <AvailabilityMacros.h>
# include <TargetConditionals.h>
#endif

#include <iostream>  
#include <locale>
#include <memory>
#include <string>  
#include <tuple>
#include <vector>  


#ifndef GTEST_INCLUDE_GTEST_INTERNAL_CUSTOM_GTEST_PORT_H_
#define GTEST_INCLUDE_GTEST_INTERNAL_CUSTOM_GTEST_PORT_H_

#endif  

#ifndef GTEST_INCLUDE_GTEST_INTERNAL_GTEST_PORT_ARCH_H_
#define GTEST_INCLUDE_GTEST_INTERNAL_GTEST_PORT_ARCH_H_

#ifdef __CYGWIN__
# define GTEST_OS_CYGWIN 1
# elif defined(__MINGW__) || defined(__MINGW32__) || defined(__MINGW64__)
#  define GTEST_OS_WINDOWS_MINGW 1
#  define GTEST_OS_WINDOWS 1
#elif defined _WIN32
# define GTEST_OS_WINDOWS 1
# ifdef _WIN32_WCE
#  define GTEST_OS_WINDOWS_MOBILE 1
# elif defined(WINAPI_FAMILY)
#  include <winapifamily.h>
#  if WINAPI_FAMILY_PARTITION(WINAPI_PARTITION_DESKTOP)
#   define GTEST_OS_WINDOWS_DESKTOP 1
#  elif WINAPI_FAMILY_PARTITION(WINAPI_PARTITION_PHONE_APP)
#   define GTEST_OS_WINDOWS_PHONE 1
#  elif WINAPI_FAMILY_PARTITION(WINAPI_PARTITION_APP)
#   define GTEST_OS_WINDOWS_RT 1
#  elif WINAPI_FAMILY_PARTITION(WINAPI_PARTITION_TV_TITLE)
#   define GTEST_OS_WINDOWS_PHONE 1
#   define GTEST_OS_WINDOWS_TV_TITLE 1
#  else
#   define GTEST_OS_WINDOWS_DESKTOP 1
#  endif
# else
#  define GTEST_OS_WINDOWS_DESKTOP 1
# endif  
#elif defined __OS2__
# define GTEST_OS_OS2 1
#elif defined __APPLE__
# define GTEST_OS_MAC 1
# if TARGET_OS_IPHONE
#  define GTEST_OS_IOS 1
# endif
#elif defined __DragonFly__
# define GTEST_OS_DRAGONFLY 1
#elif defined __FreeBSD__
# define GTEST_OS_FREEBSD 1
#elif defined __Fuchsia__
# define GTEST_OS_FUCHSIA 1
#elif defined(__GLIBC__) && defined(__FreeBSD_kernel__)
# define GTEST_OS_GNU_KFREEBSD 1
#elif defined __linux__
# define GTEST_OS_LINUX 1
# if defined __ANDROID__
#  define GTEST_OS_LINUX_ANDROID 1
# endif
#elif defined __MVS__
# define GTEST_OS_ZOS 1
#elif defined(__sun) && defined(__SVR4)
# define GTEST_OS_SOLARIS 1
#elif defined(_AIX)
# define GTEST_OS_AIX 1
#elif defined(__hpux)
# define GTEST_OS_HPUX 1
#elif defined __native_client__
# define GTEST_OS_NACL 1
#elif defined __NetBSD__
# define GTEST_OS_NETBSD 1
#elif defined __OpenBSD__
# define GTEST_OS_OPENBSD 1
#elif defined __QNX__
# define GTEST_OS_QNX 1
#elif defined(__HAIKU__)
#define GTEST_OS_HAIKU 1
#elif defined ESP8266
#define GTEST_OS_ESP8266 1
#elif defined ESP32
#define GTEST_OS_ESP32 1
#endif  

#endif  

#if !defined(GTEST_DEV_EMAIL_)
# define GTEST_DEV_EMAIL_ "googletestframework@@googlegroups.com"
# define GTEST_FLAG_PREFIX_ "gtest_"
# define GTEST_FLAG_PREFIX_DASH_ "gtest-"
# define GTEST_FLAG_PREFIX_UPPER_ "GTEST_"
# define GTEST_NAME_ "Google Test"
# define GTEST_PROJECT_URL_ "https:
#endif  

#if !defined(GTEST_INIT_GOOGLE_TEST_NAME_)
# define GTEST_INIT_GOOGLE_TEST_NAME_ "testing::InitGoogleTest"
#endif  

#ifdef __GNUC__
# define GTEST_GCC_VER_ \
(__GNUC__*10000 + __GNUC_MINOR__*100 + __GNUC_PATCHLEVEL__)
#endif  

#if defined(_MSC_VER)
# define GTEST_DISABLE_MSC_WARNINGS_PUSH_(warnings) \
__pragma(warning(push))                        \
__pragma(warning(disable: warnings))
# define GTEST_DISABLE_MSC_WARNINGS_POP_()          \
__pragma(warning(pop))
#else
# define GTEST_DISABLE_MSC_WARNINGS_PUSH_(warnings)
# define GTEST_DISABLE_MSC_WARNINGS_POP_()
#endif

#ifdef __clang__
# define GTEST_DISABLE_MSC_DEPRECATED_PUSH_()                         \
_Pragma("clang diagnostic push")                                  \
_Pragma("clang diagnostic ignored \"-Wdeprecated-declarations\"") \
_Pragma("clang diagnostic ignored \"-Wdeprecated-implementations\"")
#define GTEST_DISABLE_MSC_DEPRECATED_POP_() \
_Pragma("clang diagnostic pop")
#else
# define GTEST_DISABLE_MSC_DEPRECATED_PUSH_() \
GTEST_DISABLE_MSC_WARNINGS_PUSH_(4996)
# define GTEST_DISABLE_MSC_DEPRECATED_POP_() \
GTEST_DISABLE_MSC_WARNINGS_POP_()
#endif

#if GTEST_OS_WINDOWS
# if !GTEST_OS_WINDOWS_MOBILE
#  include <direct.h>
#  include <io.h>
# endif
#if GTEST_OS_WINDOWS_MINGW && !defined(__MINGW64_VERSION_MAJOR)
typedef struct _CRITICAL_SECTION GTEST_CRITICAL_SECTION;
#else
typedef struct _RTL_CRITICAL_SECTION GTEST_CRITICAL_SECTION;
#endif
#else
# include <unistd.h>
# include <strings.h>
#endif  

#if GTEST_OS_LINUX_ANDROID
#  include <android/api-level.h>  
#endif

#ifndef GTEST_HAS_POSIX_RE
# if GTEST_OS_LINUX_ANDROID
#  define GTEST_HAS_POSIX_RE (__ANDROID_API__ >= 9)
# else
#  define GTEST_HAS_POSIX_RE (!GTEST_OS_WINDOWS)
# endif
#endif

#if GTEST_USES_PCRE

#elif GTEST_HAS_POSIX_RE

# include <regex.h>  

# define GTEST_USES_POSIX_RE 1

#elif GTEST_OS_WINDOWS

# define GTEST_USES_SIMPLE_RE 1

#else

# define GTEST_USES_SIMPLE_RE 1

#endif  

#ifndef GTEST_HAS_EXCEPTIONS
# if defined(_MSC_VER) && defined(_CPPUNWIND)
#  define GTEST_HAS_EXCEPTIONS 1
# elif defined(__BORLANDC__)
#  ifndef _HAS_EXCEPTIONS
#   define _HAS_EXCEPTIONS 1
#  endif  
#  define GTEST_HAS_EXCEPTIONS _HAS_EXCEPTIONS
# elif defined(__clang__)
#  define GTEST_HAS_EXCEPTIONS (__EXCEPTIONS && __has_feature(cxx_exceptions))
# elif defined(__GNUC__) && __EXCEPTIONS
#  define GTEST_HAS_EXCEPTIONS 1
# elif defined(__SUNPRO_CC)
#  define GTEST_HAS_EXCEPTIONS 1
# elif defined(__IBMCPP__) && __EXCEPTIONS
#  define GTEST_HAS_EXCEPTIONS 1
# elif defined(__HP_aCC)
#  define GTEST_HAS_EXCEPTIONS 1
# else
#  define GTEST_HAS_EXCEPTIONS 0
# endif  
#endif  

#ifndef GTEST_HAS_STD_WSTRING
#define GTEST_HAS_STD_WSTRING                                         \
(!(GTEST_OS_LINUX_ANDROID || GTEST_OS_CYGWIN || GTEST_OS_SOLARIS || \
GTEST_OS_HAIKU || GTEST_OS_ESP32 || GTEST_OS_ESP8266))

#endif  

#ifndef GTEST_HAS_RTTI

# ifdef _MSC_VER

#ifdef _CPPRTTI  
#   define GTEST_HAS_RTTI 1
#  else
#   define GTEST_HAS_RTTI 0
#  endif

# elif defined(__GNUC__)

#  ifdef __GXX_RTTI
#   if GTEST_OS_LINUX_ANDROID && defined(_STLPORT_MAJOR) && \
!defined(__EXCEPTIONS)
#    define GTEST_HAS_RTTI 0
#   else
#    define GTEST_HAS_RTTI 1
#   endif  
#  else
#   define GTEST_HAS_RTTI 0
#  endif  

# elif defined(__clang__)

#  define GTEST_HAS_RTTI __has_feature(cxx_rtti)

# elif defined(__IBMCPP__) && (__IBMCPP__ >= 900)

#  ifdef __RTTI_ALL__
#   define GTEST_HAS_RTTI 1
#  else
#   define GTEST_HAS_RTTI 0
#  endif

# else

#  define GTEST_HAS_RTTI 1

# endif  

#endif  

#if GTEST_HAS_RTTI
# include <typeinfo>
#endif

#ifndef GTEST_HAS_PTHREAD
#define GTEST_HAS_PTHREAD                                                      \
(GTEST_OS_LINUX || GTEST_OS_MAC || GTEST_OS_HPUX || GTEST_OS_QNX ||          \
GTEST_OS_FREEBSD || GTEST_OS_NACL || GTEST_OS_NETBSD || GTEST_OS_FUCHSIA || \
GTEST_OS_DRAGONFLY || GTEST_OS_GNU_KFREEBSD || GTEST_OS_OPENBSD ||          \
GTEST_OS_HAIKU)
#endif  

#if GTEST_HAS_PTHREAD
# include <pthread.h>  

# include <time.h>  
#endif

#ifndef GTEST_HAS_CLONE

# if GTEST_OS_LINUX && !defined(__ia64__)
#  if GTEST_OS_LINUX_ANDROID
#    if defined(__LP64__) || \
(defined(__arm__) && __ANDROID_API__ >= 9) || \
(defined(__mips__) && __ANDROID_API__ >= 12) || \
(defined(__i386__) && __ANDROID_API__ >= 17)
#     define GTEST_HAS_CLONE 1
#    else
#     define GTEST_HAS_CLONE 0
#    endif
#  else
#   define GTEST_HAS_CLONE 1
#  endif
# else
#  define GTEST_HAS_CLONE 0
# endif  

#endif  

#ifndef GTEST_HAS_STREAM_REDIRECTION
#if GTEST_OS_WINDOWS_MOBILE || GTEST_OS_WINDOWS_PHONE || \
GTEST_OS_WINDOWS_RT || GTEST_OS_ESP8266
#  define GTEST_HAS_STREAM_REDIRECTION 0
# else
#  define GTEST_HAS_STREAM_REDIRECTION 1
# endif  
#endif  

#if (GTEST_OS_LINUX || GTEST_OS_CYGWIN || GTEST_OS_SOLARIS ||             \
(GTEST_OS_MAC && !GTEST_OS_IOS) ||                                   \
(GTEST_OS_WINDOWS_DESKTOP && _MSC_VER) || GTEST_OS_WINDOWS_MINGW ||  \
GTEST_OS_AIX || GTEST_OS_HPUX || GTEST_OS_OPENBSD || GTEST_OS_QNX || \
GTEST_OS_FREEBSD || GTEST_OS_NETBSD || GTEST_OS_FUCHSIA ||           \
GTEST_OS_DRAGONFLY || GTEST_OS_GNU_KFREEBSD || GTEST_OS_HAIKU)
# define GTEST_HAS_DEATH_TEST 1
#endif


#if defined(__GNUC__) || defined(_MSC_VER) || defined(__SUNPRO_CC) || \
defined(__IBMCPP__) || defined(__HP_aCC)
# define GTEST_HAS_TYPED_TEST 1
# define GTEST_HAS_TYPED_TEST_P 1
#endif

#define GTEST_WIDE_STRING_USES_UTF16_ \
(GTEST_OS_WINDOWS || GTEST_OS_CYGWIN || GTEST_OS_AIX || GTEST_OS_OS2)

#if GTEST_OS_LINUX || GTEST_OS_GNU_KFREEBSD || GTEST_OS_DRAGONFLY || \
GTEST_OS_FREEBSD || GTEST_OS_NETBSD || GTEST_OS_OPENBSD
# define GTEST_CAN_STREAM_RESULTS_ 1
#endif


#ifdef __INTEL_COMPILER
# define GTEST_AMBIGUOUS_ELSE_BLOCKER_
#else
# define GTEST_AMBIGUOUS_ELSE_BLOCKER_ switch (0) case 0: default:  
#endif

#if defined(__GNUC__) && !defined(COMPILER_ICC)
# define GTEST_ATTRIBUTE_UNUSED_ __attribute__ ((unused))
#elif defined(__clang__)
# if __has_attribute(unused)
#  define GTEST_ATTRIBUTE_UNUSED_ __attribute__ ((unused))
# endif
#endif
#ifndef GTEST_ATTRIBUTE_UNUSED_
# define GTEST_ATTRIBUTE_UNUSED_
#endif

#if (defined(__GNUC__) || defined(__clang__)) && !defined(COMPILER_ICC)
# if defined(__MINGW_PRINTF_FORMAT)
#  define GTEST_ATTRIBUTE_PRINTF_(string_index, first_to_check) \
__attribute__((__format__(__MINGW_PRINTF_FORMAT, string_index, \
first_to_check)))
# else
#  define GTEST_ATTRIBUTE_PRINTF_(string_index, first_to_check) \
__attribute__((__format__(__printf__, string_index, first_to_check)))
# endif
#else
# define GTEST_ATTRIBUTE_PRINTF_(string_index, first_to_check)
#endif


#define GTEST_DISALLOW_ASSIGN_(type) \
type& operator=(type const &) = delete

#define GTEST_DISALLOW_COPY_AND_ASSIGN_(type) \
type(type const&) = delete;                 \
type& operator=(type const&) = delete

#define GTEST_DISALLOW_MOVE_ASSIGN_(type) \
type& operator=(type &&) noexcept = delete

#define GTEST_DISALLOW_MOVE_AND_ASSIGN_(type) \
type(type&&) noexcept = delete;             \
type& operator=(type&&) noexcept = delete

#if defined(__GNUC__) && !defined(COMPILER_ICC)
# define GTEST_MUST_USE_RESULT_ __attribute__ ((warn_unused_result))
#else
# define GTEST_MUST_USE_RESULT_
#endif  

# define GTEST_INTENTIONAL_CONST_COND_PUSH_() \
GTEST_DISABLE_MSC_WARNINGS_PUSH_(4127)
# define GTEST_INTENTIONAL_CONST_COND_POP_() \
GTEST_DISABLE_MSC_WARNINGS_POP_()

#ifndef GTEST_HAS_SEH

# if defined(_MSC_VER) || defined(__BORLANDC__)
#  define GTEST_HAS_SEH 1
# else
#  define GTEST_HAS_SEH 0
# endif

#endif  

#ifndef GTEST_IS_THREADSAFE

#define GTEST_IS_THREADSAFE                                                 \
(GTEST_HAS_MUTEX_AND_THREAD_LOCAL_ ||                                     \
(GTEST_OS_WINDOWS && !GTEST_OS_WINDOWS_PHONE && !GTEST_OS_WINDOWS_RT) || \
GTEST_HAS_PTHREAD)

#endif  

#ifndef GTEST_API_

#ifdef _MSC_VER
# if GTEST_LINKED_AS_SHARED_LIBRARY
#  define GTEST_API_ __declspec(dllimport)
# elif GTEST_CREATE_SHARED_LIBRARY
#  define GTEST_API_ __declspec(dllexport)
# endif
#elif __GNUC__ >= 4 || defined(__clang__)
# define GTEST_API_ __attribute__((visibility ("default")))
#endif  

#endif  

#ifndef GTEST_API_
# define GTEST_API_
#endif  

#ifndef GTEST_DEFAULT_DEATH_TEST_STYLE
# define GTEST_DEFAULT_DEATH_TEST_STYLE  "fast"
#endif  

#ifdef __GNUC__
# define GTEST_NO_INLINE_ __attribute__((noinline))
#else
# define GTEST_NO_INLINE_
#endif

#if !defined(GTEST_HAS_CXXABI_H_)
# if defined(__GLIBCXX__) || (defined(_LIBCPP_VERSION) && !defined(_MSC_VER))
#  define GTEST_HAS_CXXABI_H_ 1
# else
#  define GTEST_HAS_CXXABI_H_ 0
# endif
#endif

#if defined(__clang__)
# if __has_feature(memory_sanitizer)
#  define GTEST_ATTRIBUTE_NO_SANITIZE_MEMORY_ \
__attribute__((no_sanitize_memory))
# else
#  define GTEST_ATTRIBUTE_NO_SANITIZE_MEMORY_
# endif  
#else
# define GTEST_ATTRIBUTE_NO_SANITIZE_MEMORY_
#endif  

#if defined(__clang__)
# if __has_feature(address_sanitizer)
#  define GTEST_ATTRIBUTE_NO_SANITIZE_ADDRESS_ \
__attribute__((no_sanitize_address))
# else
#  define GTEST_ATTRIBUTE_NO_SANITIZE_ADDRESS_
# endif  
#else
# define GTEST_ATTRIBUTE_NO_SANITIZE_ADDRESS_
#endif  

#if defined(__clang__)
# if __has_feature(hwaddress_sanitizer)
#  define GTEST_ATTRIBUTE_NO_SANITIZE_HWADDRESS_ \
__attribute__((no_sanitize("hwaddress")))
# else
#  define GTEST_ATTRIBUTE_NO_SANITIZE_HWADDRESS_
# endif  
#else
# define GTEST_ATTRIBUTE_NO_SANITIZE_HWADDRESS_
#endif  

#if defined(__clang__)
# if __has_feature(thread_sanitizer)
#  define GTEST_ATTRIBUTE_NO_SANITIZE_THREAD_ \
__attribute__((no_sanitize_thread))
# else
#  define GTEST_ATTRIBUTE_NO_SANITIZE_THREAD_
# endif  
#else
# define GTEST_ATTRIBUTE_NO_SANITIZE_THREAD_
#endif  

namespace testing {

class Message;

using std::get;
using std::make_tuple;
using std::tuple;
using std::tuple_element;
using std::tuple_size;

namespace internal {

class Secret;

#define GTEST_COMPILE_ASSERT_(expr, msg) static_assert(expr, #msg)

GTEST_API_ bool IsTrue(bool condition);


#if GTEST_USES_PCRE
#elif GTEST_USES_POSIX_RE || GTEST_USES_SIMPLE_RE

class GTEST_API_ RE {
public:
RE(const RE& other) { Init(other.pattern()); }

RE(const ::std::string& regex) { Init(regex.c_str()); }  

RE(const char* regex) { Init(regex); }  
~RE();

const char* pattern() const { return pattern_; }

static bool FullMatch(const ::std::string& str, const RE& re) {
return FullMatch(str.c_str(), re);
}
static bool PartialMatch(const ::std::string& str, const RE& re) {
return PartialMatch(str.c_str(), re);
}

static bool FullMatch(const char* str, const RE& re);
static bool PartialMatch(const char* str, const RE& re);

private:
void Init(const char* regex);
const char* pattern_;
bool is_valid_;

# if GTEST_USES_POSIX_RE

regex_t full_regex_;     
regex_t partial_regex_;  

# else  

const char* full_pattern_;  

# endif
};

#endif  

GTEST_API_ ::std::string FormatFileLocation(const char* file, int line);

GTEST_API_ ::std::string FormatCompilerIndependentFileLocation(const char* file,
int line);


enum GTestLogSeverity {
GTEST_INFO,
GTEST_WARNING,
GTEST_ERROR,
GTEST_FATAL
};

class GTEST_API_ GTestLog {
public:
GTestLog(GTestLogSeverity severity, const char* file, int line);

~GTestLog();

::std::ostream& GetStream() { return ::std::cerr; }

private:
const GTestLogSeverity severity_;

GTEST_DISALLOW_COPY_AND_ASSIGN_(GTestLog);
};

#if !defined(GTEST_LOG_)

# define GTEST_LOG_(severity) \
::testing::internal::GTestLog(::testing::internal::GTEST_##severity, \
__FILE__, __LINE__).GetStream()

inline void LogToStderr() {}
inline void FlushInfoLog() { fflush(nullptr); }

#endif  

#if !defined(GTEST_CHECK_)
# define GTEST_CHECK_(condition) \
GTEST_AMBIGUOUS_ELSE_BLOCKER_ \
if (::testing::internal::IsTrue(condition)) \
; \
else \
GTEST_LOG_(FATAL) << "Condition " #condition " failed. "
#endif  

#define GTEST_CHECK_POSIX_SUCCESS_(posix_call) \
if (const int gtest_error = (posix_call)) \
GTEST_LOG_(FATAL) << #posix_call << "failed with error " \
<< gtest_error

template <typename T>
struct ConstRef { typedef const T& type; };
template <typename T>
struct ConstRef<T&> { typedef T& type; };

#define GTEST_REFERENCE_TO_CONST_(T) \
typename ::testing::internal::ConstRef<T>::type

template<typename To>
inline To ImplicitCast_(To x) { return x; }

template<typename To, typename From>  
inline To DownCast_(From* f) {  
GTEST_INTENTIONAL_CONST_COND_PUSH_()
if (false) {
GTEST_INTENTIONAL_CONST_COND_POP_()
const To to = nullptr;
::testing::internal::ImplicitCast_<From*>(to);
}

#if GTEST_HAS_RTTI
GTEST_CHECK_(f == nullptr || dynamic_cast<To>(f) != nullptr);
#endif
return static_cast<To>(f);
}

template <class Derived, class Base>
Derived* CheckedDowncastToActualType(Base* base) {
#if GTEST_HAS_RTTI
GTEST_CHECK_(typeid(*base) == typeid(Derived));
#endif

#if GTEST_HAS_DOWNCAST_
return ::down_cast<Derived*>(base);
#elif GTEST_HAS_RTTI
return dynamic_cast<Derived*>(base);  
#else
return static_cast<Derived*>(base);  
#endif
}

#if GTEST_HAS_STREAM_REDIRECTION

GTEST_API_ void CaptureStdout();
GTEST_API_ std::string GetCapturedStdout();
GTEST_API_ void CaptureStderr();
GTEST_API_ std::string GetCapturedStderr();

#endif  
GTEST_API_ size_t GetFileSize(FILE* file);

GTEST_API_ std::string ReadEntireFile(FILE* file);

GTEST_API_ std::vector<std::string> GetArgvs();

#if GTEST_HAS_DEATH_TEST

std::vector<std::string> GetInjectableArgvs();
void SetInjectableArgvs(const std::vector<std::string>* new_argvs);
void SetInjectableArgvs(const std::vector<std::string>& new_argvs);
void ClearInjectableArgvs();

#endif  

#if GTEST_IS_THREADSAFE
# if GTEST_HAS_PTHREAD
inline void SleepMilliseconds(int n) {
const timespec time = {
0,                  
n * 1000L * 1000L,  
};
nanosleep(&time, nullptr);
}
# endif  

# if GTEST_HAS_NOTIFICATION_

# elif GTEST_HAS_PTHREAD
class Notification {
public:
Notification() : notified_(false) {
GTEST_CHECK_POSIX_SUCCESS_(pthread_mutex_init(&mutex_, nullptr));
}
~Notification() {
pthread_mutex_destroy(&mutex_);
}

void Notify() {
pthread_mutex_lock(&mutex_);
notified_ = true;
pthread_mutex_unlock(&mutex_);
}

void WaitForNotification() {
for (;;) {
pthread_mutex_lock(&mutex_);
const bool notified = notified_;
pthread_mutex_unlock(&mutex_);
if (notified)
break;
SleepMilliseconds(10);
}
}

private:
pthread_mutex_t mutex_;
bool notified_;

GTEST_DISALLOW_COPY_AND_ASSIGN_(Notification);
};

# elif GTEST_OS_WINDOWS && !GTEST_OS_WINDOWS_PHONE && !GTEST_OS_WINDOWS_RT

GTEST_API_ void SleepMilliseconds(int n);

class GTEST_API_ AutoHandle {
public:
typedef void* Handle;
AutoHandle();
explicit AutoHandle(Handle handle);

~AutoHandle();

Handle Get() const;
void Reset();
void Reset(Handle handle);

private:
bool IsCloseable() const;

Handle handle_;

GTEST_DISALLOW_COPY_AND_ASSIGN_(AutoHandle);
};

class GTEST_API_ Notification {
public:
Notification();
void Notify();
void WaitForNotification();

private:
AutoHandle event_;

GTEST_DISALLOW_COPY_AND_ASSIGN_(Notification);
};
# endif  

# if GTEST_HAS_PTHREAD && !GTEST_OS_WINDOWS_MINGW

class ThreadWithParamBase {
public:
virtual ~ThreadWithParamBase() {}
virtual void Run() = 0;
};

extern "C" inline void* ThreadFuncWithCLinkage(void* thread) {
static_cast<ThreadWithParamBase*>(thread)->Run();
return nullptr;
}

template <typename T>
class ThreadWithParam : public ThreadWithParamBase {
public:
typedef void UserThreadFunc(T);

ThreadWithParam(UserThreadFunc* func, T param, Notification* thread_can_start)
: func_(func),
param_(param),
thread_can_start_(thread_can_start),
finished_(false) {
ThreadWithParamBase* const base = this;
GTEST_CHECK_POSIX_SUCCESS_(
pthread_create(&thread_, nullptr, &ThreadFuncWithCLinkage, base));
}
~ThreadWithParam() override { Join(); }

void Join() {
if (!finished_) {
GTEST_CHECK_POSIX_SUCCESS_(pthread_join(thread_, nullptr));
finished_ = true;
}
}

void Run() override {
if (thread_can_start_ != nullptr) thread_can_start_->WaitForNotification();
func_(param_);
}

private:
UserThreadFunc* const func_;  
const T param_;  
Notification* const thread_can_start_;
bool finished_;  
pthread_t thread_;  

GTEST_DISALLOW_COPY_AND_ASSIGN_(ThreadWithParam);
};
# endif  

# if GTEST_HAS_MUTEX_AND_THREAD_LOCAL_

# elif GTEST_OS_WINDOWS && !GTEST_OS_WINDOWS_PHONE && !GTEST_OS_WINDOWS_RT

class GTEST_API_ Mutex {
public:
enum MutexType { kStatic = 0, kDynamic = 1 };
enum StaticConstructorSelector { kStaticMutex = 0 };

explicit Mutex(StaticConstructorSelector ) {}

Mutex();
~Mutex();

void Lock();

void Unlock();

void AssertHeld();

private:
void ThreadSafeLazyInit();

unsigned int owner_thread_id_;

MutexType type_;
long critical_section_init_phase_;  
GTEST_CRITICAL_SECTION* critical_section_;

GTEST_DISALLOW_COPY_AND_ASSIGN_(Mutex);
};

# define GTEST_DECLARE_STATIC_MUTEX_(mutex) \
extern ::testing::internal::Mutex mutex

# define GTEST_DEFINE_STATIC_MUTEX_(mutex) \
::testing::internal::Mutex mutex(::testing::internal::Mutex::kStaticMutex)

class GTestMutexLock {
public:
explicit GTestMutexLock(Mutex* mutex)
: mutex_(mutex) { mutex_->Lock(); }

~GTestMutexLock() { mutex_->Unlock(); }

private:
Mutex* const mutex_;

GTEST_DISALLOW_COPY_AND_ASSIGN_(GTestMutexLock);
};

typedef GTestMutexLock MutexLock;

class ThreadLocalValueHolderBase {
public:
virtual ~ThreadLocalValueHolderBase() {}
};

class ThreadLocalBase {
public:
virtual ThreadLocalValueHolderBase* NewValueForCurrentThread() const = 0;

protected:
ThreadLocalBase() {}
virtual ~ThreadLocalBase() {}

private:
GTEST_DISALLOW_COPY_AND_ASSIGN_(ThreadLocalBase);
};

class GTEST_API_ ThreadLocalRegistry {
public:
static ThreadLocalValueHolderBase* GetValueOnCurrentThread(
const ThreadLocalBase* thread_local_instance);

static void OnThreadLocalDestroyed(
const ThreadLocalBase* thread_local_instance);
};

class GTEST_API_ ThreadWithParamBase {
public:
void Join();

protected:
class Runnable {
public:
virtual ~Runnable() {}
virtual void Run() = 0;
};

ThreadWithParamBase(Runnable *runnable, Notification* thread_can_start);
virtual ~ThreadWithParamBase();

private:
AutoHandle thread_;
};

template <typename T>
class ThreadWithParam : public ThreadWithParamBase {
public:
typedef void UserThreadFunc(T);

ThreadWithParam(UserThreadFunc* func, T param, Notification* thread_can_start)
: ThreadWithParamBase(new RunnableImpl(func, param), thread_can_start) {
}
virtual ~ThreadWithParam() {}

private:
class RunnableImpl : public Runnable {
public:
RunnableImpl(UserThreadFunc* func, T param)
: func_(func),
param_(param) {
}
virtual ~RunnableImpl() {}
virtual void Run() {
func_(param_);
}

private:
UserThreadFunc* const func_;
const T param_;

GTEST_DISALLOW_COPY_AND_ASSIGN_(RunnableImpl);
};

GTEST_DISALLOW_COPY_AND_ASSIGN_(ThreadWithParam);
};

template <typename T>
class ThreadLocal : public ThreadLocalBase {
public:
ThreadLocal() : default_factory_(new DefaultValueHolderFactory()) {}
explicit ThreadLocal(const T& value)
: default_factory_(new InstanceValueHolderFactory(value)) {}

~ThreadLocal() { ThreadLocalRegistry::OnThreadLocalDestroyed(this); }

T* pointer() { return GetOrCreateValue(); }
const T* pointer() const { return GetOrCreateValue(); }
const T& get() const { return *pointer(); }
void set(const T& value) { *pointer() = value; }

private:
class ValueHolder : public ThreadLocalValueHolderBase {
public:
ValueHolder() : value_() {}
explicit ValueHolder(const T& value) : value_(value) {}

T* pointer() { return &value_; }

private:
T value_;
GTEST_DISALLOW_COPY_AND_ASSIGN_(ValueHolder);
};


T* GetOrCreateValue() const {
return static_cast<ValueHolder*>(
ThreadLocalRegistry::GetValueOnCurrentThread(this))->pointer();
}

virtual ThreadLocalValueHolderBase* NewValueForCurrentThread() const {
return default_factory_->MakeNewHolder();
}

class ValueHolderFactory {
public:
ValueHolderFactory() {}
virtual ~ValueHolderFactory() {}
virtual ValueHolder* MakeNewHolder() const = 0;

private:
GTEST_DISALLOW_COPY_AND_ASSIGN_(ValueHolderFactory);
};

class DefaultValueHolderFactory : public ValueHolderFactory {
public:
DefaultValueHolderFactory() {}
ValueHolder* MakeNewHolder() const override { return new ValueHolder(); }

private:
GTEST_DISALLOW_COPY_AND_ASSIGN_(DefaultValueHolderFactory);
};

class InstanceValueHolderFactory : public ValueHolderFactory {
public:
explicit InstanceValueHolderFactory(const T& value) : value_(value) {}
ValueHolder* MakeNewHolder() const override {
return new ValueHolder(value_);
}

private:
const T value_;  

GTEST_DISALLOW_COPY_AND_ASSIGN_(InstanceValueHolderFactory);
};

std::unique_ptr<ValueHolderFactory> default_factory_;

GTEST_DISALLOW_COPY_AND_ASSIGN_(ThreadLocal);
};

# elif GTEST_HAS_PTHREAD

class MutexBase {
public:
void Lock() {
GTEST_CHECK_POSIX_SUCCESS_(pthread_mutex_lock(&mutex_));
owner_ = pthread_self();
has_owner_ = true;
}

void Unlock() {
has_owner_ = false;
GTEST_CHECK_POSIX_SUCCESS_(pthread_mutex_unlock(&mutex_));
}

void AssertHeld() const {
GTEST_CHECK_(has_owner_ && pthread_equal(owner_, pthread_self()))
<< "The current thread is not holding the mutex @" << this;
}

public:
pthread_mutex_t mutex_;  
bool has_owner_;
pthread_t owner_;  
};

#  define GTEST_DECLARE_STATIC_MUTEX_(mutex) \
extern ::testing::internal::MutexBase mutex

#define GTEST_DEFINE_STATIC_MUTEX_(mutex) \
::testing::internal::MutexBase mutex = {PTHREAD_MUTEX_INITIALIZER, false, 0}

class Mutex : public MutexBase {
public:
Mutex() {
GTEST_CHECK_POSIX_SUCCESS_(pthread_mutex_init(&mutex_, nullptr));
has_owner_ = false;
}
~Mutex() {
GTEST_CHECK_POSIX_SUCCESS_(pthread_mutex_destroy(&mutex_));
}

private:
GTEST_DISALLOW_COPY_AND_ASSIGN_(Mutex);
};

class GTestMutexLock {
public:
explicit GTestMutexLock(MutexBase* mutex)
: mutex_(mutex) { mutex_->Lock(); }

~GTestMutexLock() { mutex_->Unlock(); }

private:
MutexBase* const mutex_;

GTEST_DISALLOW_COPY_AND_ASSIGN_(GTestMutexLock);
};

typedef GTestMutexLock MutexLock;


class ThreadLocalValueHolderBase {
public:
virtual ~ThreadLocalValueHolderBase() {}
};

extern "C" inline void DeleteThreadLocalValue(void* value_holder) {
delete static_cast<ThreadLocalValueHolderBase*>(value_holder);
}

template <typename T>
class GTEST_API_ ThreadLocal {
public:
ThreadLocal()
: key_(CreateKey()), default_factory_(new DefaultValueHolderFactory()) {}
explicit ThreadLocal(const T& value)
: key_(CreateKey()),
default_factory_(new InstanceValueHolderFactory(value)) {}

~ThreadLocal() {
DeleteThreadLocalValue(pthread_getspecific(key_));

GTEST_CHECK_POSIX_SUCCESS_(pthread_key_delete(key_));
}

T* pointer() { return GetOrCreateValue(); }
const T* pointer() const { return GetOrCreateValue(); }
const T& get() const { return *pointer(); }
void set(const T& value) { *pointer() = value; }

private:
class ValueHolder : public ThreadLocalValueHolderBase {
public:
ValueHolder() : value_() {}
explicit ValueHolder(const T& value) : value_(value) {}

T* pointer() { return &value_; }

private:
T value_;
GTEST_DISALLOW_COPY_AND_ASSIGN_(ValueHolder);
};

static pthread_key_t CreateKey() {
pthread_key_t key;
GTEST_CHECK_POSIX_SUCCESS_(
pthread_key_create(&key, &DeleteThreadLocalValue));
return key;
}

T* GetOrCreateValue() const {
ThreadLocalValueHolderBase* const holder =
static_cast<ThreadLocalValueHolderBase*>(pthread_getspecific(key_));
if (holder != nullptr) {
return CheckedDowncastToActualType<ValueHolder>(holder)->pointer();
}

ValueHolder* const new_holder = default_factory_->MakeNewHolder();
ThreadLocalValueHolderBase* const holder_base = new_holder;
GTEST_CHECK_POSIX_SUCCESS_(pthread_setspecific(key_, holder_base));
return new_holder->pointer();
}

class ValueHolderFactory {
public:
ValueHolderFactory() {}
virtual ~ValueHolderFactory() {}
virtual ValueHolder* MakeNewHolder() const = 0;

private:
GTEST_DISALLOW_COPY_AND_ASSIGN_(ValueHolderFactory);
};

class DefaultValueHolderFactory : public ValueHolderFactory {
public:
DefaultValueHolderFactory() {}
ValueHolder* MakeNewHolder() const override { return new ValueHolder(); }

private:
GTEST_DISALLOW_COPY_AND_ASSIGN_(DefaultValueHolderFactory);
};

class InstanceValueHolderFactory : public ValueHolderFactory {
public:
explicit InstanceValueHolderFactory(const T& value) : value_(value) {}
ValueHolder* MakeNewHolder() const override {
return new ValueHolder(value_);
}

private:
const T value_;  

GTEST_DISALLOW_COPY_AND_ASSIGN_(InstanceValueHolderFactory);
};

const pthread_key_t key_;
std::unique_ptr<ValueHolderFactory> default_factory_;

GTEST_DISALLOW_COPY_AND_ASSIGN_(ThreadLocal);
};

# endif  

#else  


class Mutex {
public:
Mutex() {}
void Lock() {}
void Unlock() {}
void AssertHeld() const {}
};

# define GTEST_DECLARE_STATIC_MUTEX_(mutex) \
extern ::testing::internal::Mutex mutex

# define GTEST_DEFINE_STATIC_MUTEX_(mutex) ::testing::internal::Mutex mutex

class GTestMutexLock {
public:
explicit GTestMutexLock(Mutex*) {}  
};

typedef GTestMutexLock MutexLock;

template <typename T>
class GTEST_API_ ThreadLocal {
public:
ThreadLocal() : value_() {}
explicit ThreadLocal(const T& value) : value_(value) {}
T* pointer() { return &value_; }
const T* pointer() const { return &value_; }
const T& get() const { return value_; }
void set(const T& value) { value_ = value; }
private:
T value_;
};

#endif  

GTEST_API_ size_t GetThreadCount();

#if GTEST_OS_WINDOWS
# define GTEST_PATH_SEP_ "\\"
# define GTEST_HAS_ALT_PATH_SEP_ 1
#else
# define GTEST_PATH_SEP_ "/"
# define GTEST_HAS_ALT_PATH_SEP_ 0
#endif  



inline bool IsAlpha(char ch) {
return isalpha(static_cast<unsigned char>(ch)) != 0;
}
inline bool IsAlNum(char ch) {
return isalnum(static_cast<unsigned char>(ch)) != 0;
}
inline bool IsDigit(char ch) {
return isdigit(static_cast<unsigned char>(ch)) != 0;
}
inline bool IsLower(char ch) {
return islower(static_cast<unsigned char>(ch)) != 0;
}
inline bool IsSpace(char ch) {
return isspace(static_cast<unsigned char>(ch)) != 0;
}
inline bool IsUpper(char ch) {
return isupper(static_cast<unsigned char>(ch)) != 0;
}
inline bool IsXDigit(char ch) {
return isxdigit(static_cast<unsigned char>(ch)) != 0;
}
inline bool IsXDigit(wchar_t ch) {
const unsigned char low_byte = static_cast<unsigned char>(ch);
return ch == low_byte && isxdigit(low_byte) != 0;
}

inline char ToLower(char ch) {
return static_cast<char>(tolower(static_cast<unsigned char>(ch)));
}
inline char ToUpper(char ch) {
return static_cast<char>(toupper(static_cast<unsigned char>(ch)));
}

inline std::string StripTrailingSpaces(std::string str) {
std::string::iterator it = str.end();
while (it != str.begin() && IsSpace(*--it))
it = str.erase(it);
return str;
}


namespace posix {


#if GTEST_OS_WINDOWS

typedef struct _stat StatStruct;

# ifdef __BORLANDC__
inline int DoIsATTY(int fd) { return isatty(fd); }
inline int StrCaseCmp(const char* s1, const char* s2) {
return stricmp(s1, s2);
}
inline char* StrDup(const char* src) { return strdup(src); }
# else  
#  if GTEST_OS_WINDOWS_MOBILE
inline int DoIsATTY(int ) { return 0; }
#  else
inline int DoIsATTY(int fd) { return _isatty(fd); }
#  endif  
inline int StrCaseCmp(const char* s1, const char* s2) {
return _stricmp(s1, s2);
}
inline char* StrDup(const char* src) { return _strdup(src); }
# endif  

# if GTEST_OS_WINDOWS_MOBILE
inline int FileNo(FILE* file) { return reinterpret_cast<int>(_fileno(file)); }
# else
inline int FileNo(FILE* file) { return _fileno(file); }
inline int Stat(const char* path, StatStruct* buf) { return _stat(path, buf); }
inline int RmDir(const char* dir) { return _rmdir(dir); }
inline bool IsDir(const StatStruct& st) {
return (_S_IFDIR & st.st_mode) != 0;
}
# endif  

#elif GTEST_OS_ESP8266
typedef struct stat StatStruct;

inline int FileNo(FILE* file) { return fileno(file); }
inline int DoIsATTY(int fd) { return isatty(fd); }
inline int Stat(const char* path, StatStruct* buf) {
return 0;
}
inline int StrCaseCmp(const char* s1, const char* s2) {
return strcasecmp(s1, s2);
}
inline char* StrDup(const char* src) { return strdup(src); }
inline int RmDir(const char* dir) { return rmdir(dir); }
inline bool IsDir(const StatStruct& st) { return S_ISDIR(st.st_mode); }

#else

typedef struct stat StatStruct;

inline int FileNo(FILE* file) { return fileno(file); }
inline int DoIsATTY(int fd) { return isatty(fd); }
inline int Stat(const char* path, StatStruct* buf) { return stat(path, buf); }
inline int StrCaseCmp(const char* s1, const char* s2) {
return strcasecmp(s1, s2);
}
inline char* StrDup(const char* src) { return strdup(src); }
inline int RmDir(const char* dir) { return rmdir(dir); }
inline bool IsDir(const StatStruct& st) { return S_ISDIR(st.st_mode); }

#endif  

inline int IsATTY(int fd) {
int savedErrno = errno;
int isAttyValue = DoIsATTY(fd);
errno = savedErrno;

return isAttyValue;
}


GTEST_DISABLE_MSC_DEPRECATED_PUSH_()


#if !GTEST_OS_WINDOWS_MOBILE && !GTEST_OS_WINDOWS_PHONE && !GTEST_OS_WINDOWS_RT
inline int ChDir(const char* dir) { return chdir(dir); }
#endif
inline FILE* FOpen(const char* path, const char* mode) {
#if GTEST_OS_WINDOWS && !GTEST_OS_WINDOWS_MINGW
struct wchar_codecvt : public std::codecvt<wchar_t, char, std::mbstate_t> {};
std::wstring_convert<wchar_codecvt> converter;
std::wstring wide_path = converter.from_bytes(path);
std::wstring wide_mode = converter.from_bytes(mode);
return _wfopen(wide_path.c_str(), wide_mode.c_str());
#else  
return fopen(path, mode);
#endif  
}
#if !GTEST_OS_WINDOWS_MOBILE
inline FILE *FReopen(const char* path, const char* mode, FILE* stream) {
return freopen(path, mode, stream);
}
inline FILE* FDOpen(int fd, const char* mode) { return fdopen(fd, mode); }
#endif
inline int FClose(FILE* fp) { return fclose(fp); }
#if !GTEST_OS_WINDOWS_MOBILE
inline int Read(int fd, void* buf, unsigned int count) {
return static_cast<int>(read(fd, buf, count));
}
inline int Write(int fd, const void* buf, unsigned int count) {
return static_cast<int>(write(fd, buf, count));
}
inline int Close(int fd) { return close(fd); }
inline const char* StrError(int errnum) { return strerror(errnum); }
#endif
inline const char* GetEnv(const char* name) {
#if GTEST_OS_WINDOWS_MOBILE || GTEST_OS_WINDOWS_PHONE || \
GTEST_OS_WINDOWS_RT || GTEST_OS_ESP8266
static_cast<void>(name);  
return nullptr;
#elif defined(__BORLANDC__) || defined(__SunOS_5_8) || defined(__SunOS_5_9)
const char* const env = getenv(name);
return (env != nullptr && env[0] != '\0') ? env : nullptr;
#else
return getenv(name);
#endif
}

GTEST_DISABLE_MSC_DEPRECATED_POP_()

#if GTEST_OS_WINDOWS_MOBILE
[[noreturn]] void Abort();
#else
[[noreturn]] inline void Abort() { abort(); }
#endif  

}  

#if _MSC_VER && !GTEST_OS_WINDOWS_MOBILE
# define GTEST_SNPRINTF_(buffer, size, format, ...) \
_snprintf_s(buffer, size, size, format, __VA_ARGS__)
#elif defined(_MSC_VER)
# define GTEST_SNPRINTF_ _snprintf
#else
# define GTEST_SNPRINTF_ snprintf
#endif

using BiggestInt = long long;  

constexpr BiggestInt kMaxBiggestInt = (std::numeric_limits<BiggestInt>::max)();

template <size_t size>
class TypeWithSize {
public:
using UInt = void;
};

template <>
class TypeWithSize<4> {
public:
using Int = std::int32_t;
using UInt = std::uint32_t;
};

template <>
class TypeWithSize<8> {
public:
using Int = std::int64_t;
using UInt = std::uint64_t;
};

using TimeInMillis = int64_t;  


#if !defined(GTEST_FLAG)
# define GTEST_FLAG(name) FLAGS_gtest_##name
#endif  

#if !defined(GTEST_USE_OWN_FLAGFILE_FLAG_)
# define GTEST_USE_OWN_FLAGFILE_FLAG_ 1
#endif  

#if !defined(GTEST_DECLARE_bool_)
# define GTEST_FLAG_SAVER_ ::testing::internal::GTestFlagSaver

# define GTEST_DECLARE_bool_(name) GTEST_API_ extern bool GTEST_FLAG(name)
# define GTEST_DECLARE_int32_(name) \
GTEST_API_ extern std::int32_t GTEST_FLAG(name)
# define GTEST_DECLARE_string_(name) \
GTEST_API_ extern ::std::string GTEST_FLAG(name)

# define GTEST_DEFINE_bool_(name, default_val, doc) \
GTEST_API_ bool GTEST_FLAG(name) = (default_val)
# define GTEST_DEFINE_int32_(name, default_val, doc) \
GTEST_API_ std::int32_t GTEST_FLAG(name) = (default_val)
# define GTEST_DEFINE_string_(name, default_val, doc) \
GTEST_API_ ::std::string GTEST_FLAG(name) = (default_val)

#endif  

#if !defined(GTEST_EXCLUSIVE_LOCK_REQUIRED_)
# define GTEST_EXCLUSIVE_LOCK_REQUIRED_(locks)
# define GTEST_LOCK_EXCLUDED_(locks)
#endif  

GTEST_API_ bool ParseInt32(const Message& src_text, const char* str,
int32_t* value);

bool BoolFromGTestEnv(const char* flag, bool default_val);
GTEST_API_ int32_t Int32FromGTestEnv(const char* flag, int32_t default_val);
std::string OutputFlagAlsoCheckEnvVar();
const char* StringFromGTestEnv(const char* flag, const char* default_val);

}  
}  

#if !defined(GTEST_INTERNAL_DEPRECATED)

#if defined(_MSC_VER)
#define GTEST_INTERNAL_DEPRECATED(message) __declspec(deprecated(message))
#elif defined(__GNUC__)
#define GTEST_INTERNAL_DEPRECATED(message) __attribute__((deprecated(message)))
#else
#define GTEST_INTERNAL_DEPRECATED(message)
#endif

#endif  

#if GTEST_HAS_ABSL
#define GTEST_INTERNAL_HAS_ANY 1
#include "absl/types/any.h"
namespace testing {
namespace internal {
using Any = ::absl::any;
}  
}  
#else
#ifdef __has_include
#if __has_include(<any>) && __cplusplus >= 201703L
#define GTEST_INTERNAL_HAS_ANY 1
#include <any>
namespace testing {
namespace internal {
using Any = ::std::any;
}  
}  
#endif  
#endif  
#endif  

#if GTEST_HAS_ABSL
#define GTEST_INTERNAL_HAS_OPTIONAL 1
#include "absl/types/optional.h"
namespace testing {
namespace internal {
template <typename T>
using Optional = ::absl::optional<T>;
}  
}  
#else
#ifdef __has_include
#if __has_include(<optional>) && __cplusplus >= 201703L
#define GTEST_INTERNAL_HAS_OPTIONAL 1
#include <optional>
namespace testing {
namespace internal {
template <typename T>
using Optional = ::std::optional<T>;
}  
}  
#endif  
#endif  
#endif  

#if GTEST_HAS_ABSL
# define GTEST_INTERNAL_HAS_STRING_VIEW 1
#include "absl/strings/string_view.h"
namespace testing {
namespace internal {
using StringView = ::absl::string_view;
}  
}  
#else
# ifdef __has_include
#   if __has_include(<string_view>) && __cplusplus >= 201703L
#   define GTEST_INTERNAL_HAS_STRING_VIEW 1
#include <string_view>
namespace testing {
namespace internal {
using StringView = ::std::string_view;
}  
}  
#  endif  
# endif  
#endif  

#if GTEST_HAS_ABSL
#define GTEST_INTERNAL_HAS_VARIANT 1
#include "absl/types/variant.h"
namespace testing {
namespace internal {
template <typename... T>
using Variant = ::absl::variant<T...>;
}  
}  
#else
#ifdef __has_include
#if __has_include(<variant>) && __cplusplus >= 201703L
#define GTEST_INTERNAL_HAS_VARIANT 1
#include <variant>
namespace testing {
namespace internal {
template <typename... T>
using Variant = ::std::variant<T...>;
}  
}  
#endif  
#endif  
#endif  

#endif  

#if GTEST_OS_LINUX
# include <stdlib.h>
# include <sys/types.h>
# include <sys/wait.h>
# include <unistd.h>
#endif  

#if GTEST_HAS_EXCEPTIONS
# include <stdexcept>
#endif

#include <ctype.h>
#include <float.h>
#include <string.h>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <type_traits>
#include <vector>




#ifndef GTEST_INCLUDE_GTEST_GTEST_MESSAGE_H_
#define GTEST_INCLUDE_GTEST_GTEST_MESSAGE_H_

#include <limits>
#include <memory>
#include <sstream>


GTEST_DISABLE_MSC_WARNINGS_PUSH_(4251 \
)

void operator<<(const testing::internal::Secret&, int);

namespace testing {

class GTEST_API_ Message {
private:
typedef std::ostream& (*BasicNarrowIoManip)(std::ostream&);

public:
Message();

Message(const Message& msg) : ss_(new ::std::stringstream) {  
*ss_ << msg.GetString();
}

explicit Message(const char* str) : ss_(new ::std::stringstream) {
*ss_ << str;
}

template <typename T>
inline Message& operator <<(const T& val) {
using ::operator <<;
*ss_ << val;
return *this;
}

template <typename T>
inline Message& operator <<(T* const& pointer) {  
if (pointer == nullptr) {
*ss_ << "(null)";
} else {
*ss_ << pointer;
}
return *this;
}

Message& operator <<(BasicNarrowIoManip val) {
*ss_ << val;
return *this;
}

Message& operator <<(bool b) {
return *this << (b ? "true" : "false");
}

Message& operator <<(const wchar_t* wide_c_str);
Message& operator <<(wchar_t* wide_c_str);

#if GTEST_HAS_STD_WSTRING
Message& operator <<(const ::std::wstring& wstr);
#endif  

std::string GetString() const;

private:
const std::unique_ptr< ::std::stringstream> ss_;

void operator=(const Message&);
};

inline std::ostream& operator <<(std::ostream& os, const Message& sb) {
return os << sb.GetString();
}

namespace internal {

template <typename T>
std::string StreamableToString(const T& streamable) {
return (Message() << streamable).GetString();
}

}  
}  

GTEST_DISABLE_MSC_WARNINGS_POP_()  

#endif  


#ifndef GTEST_INCLUDE_GTEST_INTERNAL_GTEST_FILEPATH_H_
#define GTEST_INCLUDE_GTEST_INTERNAL_GTEST_FILEPATH_H_



#ifndef GTEST_INCLUDE_GTEST_INTERNAL_GTEST_STRING_H_
#define GTEST_INCLUDE_GTEST_INTERNAL_GTEST_STRING_H_

#ifdef __BORLANDC__
# include <mem.h>
#endif

#include <string.h>
#include <cstdint>
#include <string>


namespace testing {
namespace internal {

class GTEST_API_ String {
public:

static const char* CloneCString(const char* c_str);

#if GTEST_OS_WINDOWS_MOBILE

static LPCWSTR AnsiToUtf16(const char* c_str);

static const char* Utf16ToAnsi(LPCWSTR utf16_str);
#endif

static bool CStringEquals(const char* lhs, const char* rhs);

static std::string ShowWideCString(const wchar_t* wide_c_str);

static bool WideCStringEquals(const wchar_t* lhs, const wchar_t* rhs);

static bool CaseInsensitiveCStringEquals(const char* lhs,
const char* rhs);

static bool CaseInsensitiveWideCStringEquals(const wchar_t* lhs,
const wchar_t* rhs);

static bool EndsWithCaseInsensitive(
const std::string& str, const std::string& suffix);

static std::string FormatIntWidth2(int value);  

static std::string FormatHexInt(int value);

static std::string FormatHexUInt32(uint32_t value);

static std::string FormatByte(unsigned char value);

private:
String();  
};  

GTEST_API_ std::string StringStreamToString(::std::stringstream* stream);

}  
}  

#endif  

GTEST_DISABLE_MSC_WARNINGS_PUSH_(4251 \
)

namespace testing {
namespace internal {


class GTEST_API_ FilePath {
public:
FilePath() : pathname_("") { }
FilePath(const FilePath& rhs) : pathname_(rhs.pathname_) { }

explicit FilePath(const std::string& pathname) : pathname_(pathname) {
Normalize();
}

FilePath& operator=(const FilePath& rhs) {
Set(rhs);
return *this;
}

void Set(const FilePath& rhs) {
pathname_ = rhs.pathname_;
}

const std::string& string() const { return pathname_; }
const char* c_str() const { return pathname_.c_str(); }

static FilePath GetCurrentDir();

static FilePath MakeFileName(const FilePath& directory,
const FilePath& base_name,
int number,
const char* extension);

static FilePath ConcatPaths(const FilePath& directory,
const FilePath& relative_path);

static FilePath GenerateUniqueFileName(const FilePath& directory,
const FilePath& base_name,
const char* extension);

bool IsEmpty() const { return pathname_.empty(); }

FilePath RemoveTrailingPathSeparator() const;

FilePath RemoveDirectoryName() const;

FilePath RemoveFileName() const;

FilePath RemoveExtension(const char* extension) const;

bool CreateDirectoriesRecursively() const;

bool CreateFolder() const;

bool FileOrDirectoryExists() const;

bool DirectoryExists() const;

bool IsDirectory() const;

bool IsRootDirectory() const;

bool IsAbsolutePath() const;

private:

void Normalize();

const char* FindLastPathSeparator() const;

std::string pathname_;
};  

}  
}  

GTEST_DISABLE_MSC_WARNINGS_POP_()  

#endif  



#ifndef GTEST_INCLUDE_GTEST_INTERNAL_GTEST_TYPE_UTIL_H_
#define GTEST_INCLUDE_GTEST_INTERNAL_GTEST_TYPE_UTIL_H_


# if GTEST_HAS_CXXABI_H_
#  include <cxxabi.h>
# elif defined(__HP_aCC)
#  include <acxx_demangle.h>
# endif  

namespace testing {
namespace internal {

inline std::string CanonicalizeForStdLibVersioning(std::string s) {
static const char prefix[] = "std::__";
if (s.compare(0, strlen(prefix), prefix) == 0) {
std::string::size_type end = s.find("::", strlen(prefix));
if (end != s.npos) {
s.erase(strlen("std"), end - strlen("std"));
}
}
return s;
}

#if GTEST_HAS_RTTI
inline std::string GetTypeName(const std::type_info& type) {
const char* const name = type.name();
#if GTEST_HAS_CXXABI_H_ || defined(__HP_aCC)
int status = 0;
#if GTEST_HAS_CXXABI_H_
using abi::__cxa_demangle;
#endif  
char* const readable_name = __cxa_demangle(name, nullptr, nullptr, &status);
const std::string name_str(status == 0 ? readable_name : name);
free(readable_name);
return CanonicalizeForStdLibVersioning(name_str);
#else
return name;
#endif  
}
#endif  

template <typename T>
std::string GetTypeName() {
#if GTEST_HAS_RTTI
return GetTypeName(typeid(T));
#else
return "<type>";
#endif  
}

#if GTEST_HAS_TYPED_TEST || GTEST_HAS_TYPED_TEST_P

struct None {};

# define GTEST_TEMPLATE_ template <typename T> class

template <GTEST_TEMPLATE_ Tmpl>
struct TemplateSel {
template <typename T>
struct Bind {
typedef Tmpl<T> type;
};
};

# define GTEST_BIND_(TmplSel, T) \
TmplSel::template Bind<T>::type

template <GTEST_TEMPLATE_ Head_, GTEST_TEMPLATE_... Tail_>
struct Templates {
using Head = TemplateSel<Head_>;
using Tail = Templates<Tail_...>;
};

template <GTEST_TEMPLATE_ Head_>
struct Templates<Head_> {
using Head = TemplateSel<Head_>;
using Tail = None;
};

template <typename Head_, typename... Tail_>
struct Types {
using Head = Head_;
using Tail = Types<Tail_...>;
};

template <typename Head_>
struct Types<Head_> {
using Head = Head_;
using Tail = None;
};

template <typename... Ts>
struct ProxyTypeList {
using type = Types<Ts...>;
};

template <typename>
struct is_proxy_type_list : std::false_type {};

template <typename... Ts>
struct is_proxy_type_list<ProxyTypeList<Ts...>> : std::true_type {};

template <typename T>
struct GenerateTypeList {
private:
using proxy = typename std::conditional<is_proxy_type_list<T>::value, T,
ProxyTypeList<T>>::type;

public:
using type = typename proxy::type;
};

#endif  

}  

template <typename... Ts>
using Types = internal::ProxyTypeList<Ts...>;

}  

#endif  

#define GTEST_CONCAT_TOKEN_(foo, bar) GTEST_CONCAT_TOKEN_IMPL_(foo, bar)
#define GTEST_CONCAT_TOKEN_IMPL_(foo, bar) foo ## bar

#define GTEST_STRINGIFY_HELPER_(name, ...) #name
#define GTEST_STRINGIFY_(...) GTEST_STRINGIFY_HELPER_(__VA_ARGS__, )

namespace proto2 {
class MessageLite;
}

namespace testing {


class AssertionResult;                 
class Message;                         
class Test;                            
class TestInfo;                        
class TestPartResult;                  
class UnitTest;                        

template <typename T>
::std::string PrintToString(const T& value);

namespace internal {

struct TraceInfo;                      
class TestInfoImpl;                    
class UnitTestImpl;                    

GTEST_API_ extern const char kStackTraceMarker[];

class IgnoredValue {
struct Sink {};
public:
template <typename T,
typename std::enable_if<!std::is_convertible<T, Sink>::value,
int>::type = 0>
IgnoredValue(const T& ) {}  
};

GTEST_API_ std::string AppendUserMessage(
const std::string& gtest_msg, const Message& user_msg);

#if GTEST_HAS_EXCEPTIONS

GTEST_DISABLE_MSC_WARNINGS_PUSH_(4275 \
)

class GTEST_API_ GoogleTestFailureException : public ::std::runtime_error {
public:
explicit GoogleTestFailureException(const TestPartResult& failure);
};

GTEST_DISABLE_MSC_WARNINGS_POP_()  

#endif  

namespace edit_distance {
enum EditType { kMatch, kAdd, kRemove, kReplace };
GTEST_API_ std::vector<EditType> CalculateOptimalEdits(
const std::vector<size_t>& left, const std::vector<size_t>& right);

GTEST_API_ std::vector<EditType> CalculateOptimalEdits(
const std::vector<std::string>& left,
const std::vector<std::string>& right);

GTEST_API_ std::string CreateUnifiedDiff(const std::vector<std::string>& left,
const std::vector<std::string>& right,
size_t context = 2);

}  

GTEST_API_ std::string DiffStrings(const std::string& left,
const std::string& right,
size_t* total_line_count);

GTEST_API_ AssertionResult EqFailure(const char* expected_expression,
const char* actual_expression,
const std::string& expected_value,
const std::string& actual_value,
bool ignoring_case);

GTEST_API_ std::string GetBoolAssertionFailureMessage(
const AssertionResult& assertion_result,
const char* expression_text,
const char* actual_predicate_value,
const char* expected_predicate_value);

template <typename RawType>
class FloatingPoint {
public:
typedef typename TypeWithSize<sizeof(RawType)>::UInt Bits;


static const size_t kBitCount = 8*sizeof(RawType);

static const size_t kFractionBitCount =
std::numeric_limits<RawType>::digits - 1;

static const size_t kExponentBitCount = kBitCount - 1 - kFractionBitCount;

static const Bits kSignBitMask = static_cast<Bits>(1) << (kBitCount - 1);

static const Bits kFractionBitMask =
~static_cast<Bits>(0) >> (kExponentBitCount + 1);

static const Bits kExponentBitMask = ~(kSignBitMask | kFractionBitMask);

static const size_t kMaxUlps = 4;

explicit FloatingPoint(const RawType& x) { u_.value_ = x; }


static RawType ReinterpretBits(const Bits bits) {
FloatingPoint fp(0);
fp.u_.bits_ = bits;
return fp.u_.value_;
}

static RawType Infinity() {
return ReinterpretBits(kExponentBitMask);
}

static RawType Max();


const Bits &bits() const { return u_.bits_; }

Bits exponent_bits() const { return kExponentBitMask & u_.bits_; }

Bits fraction_bits() const { return kFractionBitMask & u_.bits_; }

Bits sign_bit() const { return kSignBitMask & u_.bits_; }

bool is_nan() const {
return (exponent_bits() == kExponentBitMask) && (fraction_bits() != 0);
}

bool AlmostEquals(const FloatingPoint& rhs) const {
if (is_nan() || rhs.is_nan()) return false;

return DistanceBetweenSignAndMagnitudeNumbers(u_.bits_, rhs.u_.bits_)
<= kMaxUlps;
}

private:
union FloatingPointUnion {
RawType value_;  
Bits bits_;      
};

static Bits SignAndMagnitudeToBiased(const Bits &sam) {
if (kSignBitMask & sam) {
return ~sam + 1;
} else {
return kSignBitMask | sam;
}
}

static Bits DistanceBetweenSignAndMagnitudeNumbers(const Bits &sam1,
const Bits &sam2) {
const Bits biased1 = SignAndMagnitudeToBiased(sam1);
const Bits biased2 = SignAndMagnitudeToBiased(sam2);
return (biased1 >= biased2) ? (biased1 - biased2) : (biased2 - biased1);
}

FloatingPointUnion u_;
};

template <>
inline float FloatingPoint<float>::Max() { return FLT_MAX; }
template <>
inline double FloatingPoint<double>::Max() { return DBL_MAX; }

typedef FloatingPoint<float> Float;
typedef FloatingPoint<double> Double;

typedef const void* TypeId;

template <typename T>
class TypeIdHelper {
public:
static bool dummy_;
};

template <typename T>
bool TypeIdHelper<T>::dummy_ = false;

template <typename T>
TypeId GetTypeId() {
return &(TypeIdHelper<T>::dummy_);
}

GTEST_API_ TypeId GetTestTypeId();

class TestFactoryBase {
public:
virtual ~TestFactoryBase() {}

virtual Test* CreateTest() = 0;

protected:
TestFactoryBase() {}

private:
GTEST_DISALLOW_COPY_AND_ASSIGN_(TestFactoryBase);
};

template <class TestClass>
class TestFactoryImpl : public TestFactoryBase {
public:
Test* CreateTest() override { return new TestClass; }
};

#if GTEST_OS_WINDOWS

GTEST_API_ AssertionResult IsHRESULTSuccess(const char* expr,
long hr);  
GTEST_API_ AssertionResult IsHRESULTFailure(const char* expr,
long hr);  

#endif  

using SetUpTestSuiteFunc = void (*)();
using TearDownTestSuiteFunc = void (*)();

struct CodeLocation {
CodeLocation(const std::string& a_file, int a_line)
: file(a_file), line(a_line) {}

std::string file;
int line;
};


using SetUpTearDownSuiteFuncType = void (*)();

inline SetUpTearDownSuiteFuncType GetNotDefaultOrNull(
SetUpTearDownSuiteFuncType a, SetUpTearDownSuiteFuncType def) {
return a == def ? nullptr : a;
}

template <typename T>
struct SuiteApiResolver : T {
using Test =
typename std::conditional<sizeof(T) != 0, ::testing::Test, void>::type;

static SetUpTearDownSuiteFuncType GetSetUpCaseOrSuite(const char* filename,
int line_num) {
#ifndef GTEST_REMOVE_LEGACY_TEST_CASEAPI_
SetUpTearDownSuiteFuncType test_case_fp =
GetNotDefaultOrNull(&T::SetUpTestCase, &Test::SetUpTestCase);
SetUpTearDownSuiteFuncType test_suite_fp =
GetNotDefaultOrNull(&T::SetUpTestSuite, &Test::SetUpTestSuite);

GTEST_CHECK_(!test_case_fp || !test_suite_fp)
<< "Test can not provide both SetUpTestSuite and SetUpTestCase, please "
"make sure there is only one present at "
<< filename << ":" << line_num;

return test_case_fp != nullptr ? test_case_fp : test_suite_fp;
#else
(void)(filename);
(void)(line_num);
return &T::SetUpTestSuite;
#endif
}

static SetUpTearDownSuiteFuncType GetTearDownCaseOrSuite(const char* filename,
int line_num) {
#ifndef GTEST_REMOVE_LEGACY_TEST_CASEAPI_
SetUpTearDownSuiteFuncType test_case_fp =
GetNotDefaultOrNull(&T::TearDownTestCase, &Test::TearDownTestCase);
SetUpTearDownSuiteFuncType test_suite_fp =
GetNotDefaultOrNull(&T::TearDownTestSuite, &Test::TearDownTestSuite);

GTEST_CHECK_(!test_case_fp || !test_suite_fp)
<< "Test can not provide both TearDownTestSuite and TearDownTestCase,"
" please make sure there is only one present at"
<< filename << ":" << line_num;

return test_case_fp != nullptr ? test_case_fp : test_suite_fp;
#else
(void)(filename);
(void)(line_num);
return &T::TearDownTestSuite;
#endif
}
};

GTEST_API_ TestInfo* MakeAndRegisterTestInfo(
const char* test_suite_name, const char* name, const char* type_param,
const char* value_param, CodeLocation code_location,
TypeId fixture_class_id, SetUpTestSuiteFunc set_up_tc,
TearDownTestSuiteFunc tear_down_tc, TestFactoryBase* factory);

GTEST_API_ bool SkipPrefix(const char* prefix, const char** pstr);

#if GTEST_HAS_TYPED_TEST || GTEST_HAS_TYPED_TEST_P

GTEST_DISABLE_MSC_WARNINGS_PUSH_(4251 \
)

class GTEST_API_ TypedTestSuitePState {
public:
TypedTestSuitePState() : registered_(false) {}

bool AddTestName(const char* file, int line, const char* case_name,
const char* test_name) {
if (registered_) {
fprintf(stderr,
"%s Test %s must be defined before "
"REGISTER_TYPED_TEST_SUITE_P(%s, ...).\n",
FormatFileLocation(file, line).c_str(), test_name, case_name);
fflush(stderr);
posix::Abort();
}
registered_tests_.insert(
::std::make_pair(test_name, CodeLocation(file, line)));
return true;
}

bool TestExists(const std::string& test_name) const {
return registered_tests_.count(test_name) > 0;
}

const CodeLocation& GetCodeLocation(const std::string& test_name) const {
RegisteredTestsMap::const_iterator it = registered_tests_.find(test_name);
GTEST_CHECK_(it != registered_tests_.end());
return it->second;
}

const char* VerifyRegisteredTestNames(const char* test_suite_name,
const char* file, int line,
const char* registered_tests);

private:
typedef ::std::map<std::string, CodeLocation> RegisteredTestsMap;

bool registered_;
RegisteredTestsMap registered_tests_;
};

#ifndef GTEST_REMOVE_LEGACY_TEST_CASEAPI_
using TypedTestCasePState = TypedTestSuitePState;
#endif  

GTEST_DISABLE_MSC_WARNINGS_POP_()  

inline const char* SkipComma(const char* str) {
const char* comma = strchr(str, ',');
if (comma == nullptr) {
return nullptr;
}
while (IsSpace(*(++comma))) {}
return comma;
}

inline std::string GetPrefixUntilComma(const char* str) {
const char* comma = strchr(str, ',');
return comma == nullptr ? str : std::string(str, comma);
}

void SplitString(const ::std::string& str, char delimiter,
::std::vector< ::std::string>* dest);

struct DefaultNameGenerator {
template <typename T>
static std::string GetName(int i) {
return StreamableToString(i);
}
};

template <typename Provided = DefaultNameGenerator>
struct NameGeneratorSelector {
typedef Provided type;
};

template <typename NameGenerator>
void GenerateNamesRecursively(internal::None, std::vector<std::string>*, int) {}

template <typename NameGenerator, typename Types>
void GenerateNamesRecursively(Types, std::vector<std::string>* result, int i) {
result->push_back(NameGenerator::template GetName<typename Types::Head>(i));
GenerateNamesRecursively<NameGenerator>(typename Types::Tail(), result,
i + 1);
}

template <typename NameGenerator, typename Types>
std::vector<std::string> GenerateNames() {
std::vector<std::string> result;
GenerateNamesRecursively<NameGenerator>(Types(), &result, 0);
return result;
}

template <GTEST_TEMPLATE_ Fixture, class TestSel, typename Types>
class TypeParameterizedTest {
public:
static bool Register(const char* prefix, const CodeLocation& code_location,
const char* case_name, const char* test_names, int index,
const std::vector<std::string>& type_names =
GenerateNames<DefaultNameGenerator, Types>()) {
typedef typename Types::Head Type;
typedef Fixture<Type> FixtureClass;
typedef typename GTEST_BIND_(TestSel, Type) TestClass;

MakeAndRegisterTestInfo(
(std::string(prefix) + (prefix[0] == '\0' ? "" : "/") + case_name +
"/" + type_names[static_cast<size_t>(index)])
.c_str(),
StripTrailingSpaces(GetPrefixUntilComma(test_names)).c_str(),
GetTypeName<Type>().c_str(),
nullptr,  
code_location, GetTypeId<FixtureClass>(),
SuiteApiResolver<TestClass>::GetSetUpCaseOrSuite(
code_location.file.c_str(), code_location.line),
SuiteApiResolver<TestClass>::GetTearDownCaseOrSuite(
code_location.file.c_str(), code_location.line),
new TestFactoryImpl<TestClass>);

return TypeParameterizedTest<Fixture, TestSel,
typename Types::Tail>::Register(prefix,
code_location,
case_name,
test_names,
index + 1,
type_names);
}
};

template <GTEST_TEMPLATE_ Fixture, class TestSel>
class TypeParameterizedTest<Fixture, TestSel, internal::None> {
public:
static bool Register(const char* , const CodeLocation&,
const char* , const char* ,
int ,
const std::vector<std::string>& =
std::vector<std::string>() ) {
return true;
}
};

GTEST_API_ void RegisterTypeParameterizedTestSuite(const char* test_suite_name,
CodeLocation code_location);
GTEST_API_ void RegisterTypeParameterizedTestSuiteInstantiation(
const char* case_name);

template <GTEST_TEMPLATE_ Fixture, typename Tests, typename Types>
class TypeParameterizedTestSuite {
public:
static bool Register(const char* prefix, CodeLocation code_location,
const TypedTestSuitePState* state, const char* case_name,
const char* test_names,
const std::vector<std::string>& type_names =
GenerateNames<DefaultNameGenerator, Types>()) {
RegisterTypeParameterizedTestSuiteInstantiation(case_name);
std::string test_name = StripTrailingSpaces(
GetPrefixUntilComma(test_names));
if (!state->TestExists(test_name)) {
fprintf(stderr, "Failed to get code location for test %s.%s at %s.",
case_name, test_name.c_str(),
FormatFileLocation(code_location.file.c_str(),
code_location.line).c_str());
fflush(stderr);
posix::Abort();
}
const CodeLocation& test_location = state->GetCodeLocation(test_name);

typedef typename Tests::Head Head;

TypeParameterizedTest<Fixture, Head, Types>::Register(
prefix, test_location, case_name, test_names, 0, type_names);

return TypeParameterizedTestSuite<Fixture, typename Tests::Tail,
Types>::Register(prefix, code_location,
state, case_name,
SkipComma(test_names),
type_names);
}
};

template <GTEST_TEMPLATE_ Fixture, typename Types>
class TypeParameterizedTestSuite<Fixture, internal::None, Types> {
public:
static bool Register(const char* , const CodeLocation&,
const TypedTestSuitePState* ,
const char* , const char* ,
const std::vector<std::string>& =
std::vector<std::string>() ) {
return true;
}
};

#endif  

GTEST_API_ std::string GetCurrentOsStackTraceExceptTop(
UnitTest* unit_test, int skip_count);


GTEST_API_ bool AlwaysTrue();

inline bool AlwaysFalse() { return !AlwaysTrue(); }

struct GTEST_API_ ConstCharPtr {
ConstCharPtr(const char* str) : value(str) {}
operator bool() const { return true; }
const char* value;
};

struct TrueWithString {
TrueWithString() = default;
explicit TrueWithString(const char* str) : value(str) {}
explicit TrueWithString(const std::string& str) : value(str) {}
explicit operator bool() const { return true; }
std::string value;
};

class GTEST_API_ Random {
public:
static const uint32_t kMaxRange = 1u << 31;

explicit Random(uint32_t seed) : state_(seed) {}

void Reseed(uint32_t seed) { state_ = seed; }

uint32_t Generate(uint32_t range);

private:
uint32_t state_;
GTEST_DISALLOW_COPY_AND_ASSIGN_(Random);
};

#define GTEST_REMOVE_REFERENCE_AND_CONST_(T) \
typename std::remove_const<typename std::remove_reference<T>::type>::type

template <typename T>
struct IsAProtocolMessage
: public std::is_convertible<const T*, const ::proto2::MessageLite*> {};

typedef int IsContainer;
template <class C,
class Iterator = decltype(::std::declval<const C&>().begin()),
class = decltype(::std::declval<const C&>().end()),
class = decltype(++::std::declval<Iterator&>()),
class = decltype(*::std::declval<Iterator>()),
class = typename C::const_iterator>
IsContainer IsContainerTest(int ) {
return 0;
}

typedef char IsNotContainer;
template <class C>
IsNotContainer IsContainerTest(long ) { return '\0'; }

template <typename T>
struct IsHashTable {
private:
template <typename U>
static char test(typename U::hasher*, typename U::reverse_iterator*);
template <typename U>
static int test(typename U::hasher*, ...);
template <typename U>
static char test(...);

public:
static const bool value = sizeof(test<T>(nullptr, nullptr)) == sizeof(int);
};

template <typename T>
const bool IsHashTable<T>::value;

template <typename C,
bool = sizeof(IsContainerTest<C>(0)) == sizeof(IsContainer)>
struct IsRecursiveContainerImpl;

template <typename C>
struct IsRecursiveContainerImpl<C, false> : public std::false_type {};

template <typename C>
struct IsRecursiveContainerImpl<C, true> {
using value_type = decltype(*std::declval<typename C::const_iterator>());
using type =
std::is_same<typename std::remove_const<
typename std::remove_reference<value_type>::type>::type,
C>;
};

template <typename C>
struct IsRecursiveContainer : public IsRecursiveContainerImpl<C>::type {};



template <typename T, typename U>
bool ArrayEq(const T* lhs, size_t size, const U* rhs);

template <typename T, typename U>
inline bool ArrayEq(const T& lhs, const U& rhs) { return lhs == rhs; }

template <typename T, typename U, size_t N>
inline bool ArrayEq(const T(&lhs)[N], const U(&rhs)[N]) {
return internal::ArrayEq(lhs, N, rhs);
}

template <typename T, typename U>
bool ArrayEq(const T* lhs, size_t size, const U* rhs) {
for (size_t i = 0; i != size; i++) {
if (!internal::ArrayEq(lhs[i], rhs[i]))
return false;
}
return true;
}

template <typename Iter, typename Element>
Iter ArrayAwareFind(Iter begin, Iter end, const Element& elem) {
for (Iter it = begin; it != end; ++it) {
if (internal::ArrayEq(*it, elem))
return it;
}
return end;
}


template <typename T, typename U>
void CopyArray(const T* from, size_t size, U* to);

template <typename T, typename U>
inline void CopyArray(const T& from, U* to) { *to = from; }

template <typename T, typename U, size_t N>
inline void CopyArray(const T(&from)[N], U(*to)[N]) {
internal::CopyArray(from, N, *to);
}

template <typename T, typename U>
void CopyArray(const T* from, size_t size, U* to) {
for (size_t i = 0; i != size; i++) {
internal::CopyArray(from[i], to + i);
}
}

struct RelationToSourceReference {};
struct RelationToSourceCopy {};

template <typename Element>
class NativeArray {
public:
typedef Element value_type;
typedef Element* iterator;
typedef const Element* const_iterator;

NativeArray(const Element* array, size_t count, RelationToSourceReference) {
InitRef(array, count);
}

NativeArray(const Element* array, size_t count, RelationToSourceCopy) {
InitCopy(array, count);
}

NativeArray(const NativeArray& rhs) {
(this->*rhs.clone_)(rhs.array_, rhs.size_);
}

~NativeArray() {
if (clone_ != &NativeArray::InitRef)
delete[] array_;
}

size_t size() const { return size_; }
const_iterator begin() const { return array_; }
const_iterator end() const { return array_ + size_; }
bool operator==(const NativeArray& rhs) const {
return size() == rhs.size() &&
ArrayEq(begin(), size(), rhs.begin());
}

private:
static_assert(!std::is_const<Element>::value, "Type must not be const");
static_assert(!std::is_reference<Element>::value,
"Type must not be a reference");

void InitCopy(const Element* array, size_t a_size) {
Element* const copy = new Element[a_size];
CopyArray(array, a_size, copy);
array_ = copy;
size_ = a_size;
clone_ = &NativeArray::InitCopy;
}

void InitRef(const Element* array, size_t a_size) {
array_ = array;
size_ = a_size;
clone_ = &NativeArray::InitRef;
}

const Element* array_;
size_t size_;
void (NativeArray::*clone_)(const Element*, size_t);
};

template <size_t... Is>
struct IndexSequence {
using type = IndexSequence;
};

template <bool plus_one, typename T, size_t sizeofT>
struct DoubleSequence;
template <size_t... I, size_t sizeofT>
struct DoubleSequence<true, IndexSequence<I...>, sizeofT> {
using type = IndexSequence<I..., (sizeofT + I)..., 2 * sizeofT>;
};
template <size_t... I, size_t sizeofT>
struct DoubleSequence<false, IndexSequence<I...>, sizeofT> {
using type = IndexSequence<I..., (sizeofT + I)...>;
};

template <size_t N>
struct MakeIndexSequence
: DoubleSequence<N % 2 == 1, typename MakeIndexSequence<N / 2>::type,
N / 2>::type {};

template <>
struct MakeIndexSequence<0> : IndexSequence<> {};

template <size_t>
struct Ignore {
Ignore(...);  
};

template <typename>
struct ElemFromListImpl;
template <size_t... I>
struct ElemFromListImpl<IndexSequence<I...>> {
template <typename R>
static R Apply(Ignore<0 * I>..., R (*)(), ...);
};

template <size_t N, typename... T>
struct ElemFromList {
using type =
decltype(ElemFromListImpl<typename MakeIndexSequence<N>::type>::Apply(
static_cast<T (*)()>(nullptr)...));
};

template <typename... T>
class FlatTuple;

template <typename Derived, size_t I>
struct FlatTupleElemBase;

template <typename... T, size_t I>
struct FlatTupleElemBase<FlatTuple<T...>, I> {
using value_type = typename ElemFromList<I, T...>::type;
FlatTupleElemBase() = default;
explicit FlatTupleElemBase(value_type t) : value(std::move(t)) {}
value_type value;
};

template <typename Derived, typename Idx>
struct FlatTupleBase;

template <size_t... Idx, typename... T>
struct FlatTupleBase<FlatTuple<T...>, IndexSequence<Idx...>>
: FlatTupleElemBase<FlatTuple<T...>, Idx>... {
using Indices = IndexSequence<Idx...>;
FlatTupleBase() = default;
explicit FlatTupleBase(T... t)
: FlatTupleElemBase<FlatTuple<T...>, Idx>(std::move(t))... {}
};

template <typename... T>
class FlatTuple
: private FlatTupleBase<FlatTuple<T...>,
typename MakeIndexSequence<sizeof...(T)>::type> {
using Indices = typename FlatTupleBase<
FlatTuple<T...>, typename MakeIndexSequence<sizeof...(T)>::type>::Indices;

public:
FlatTuple() = default;
explicit FlatTuple(T... t) : FlatTuple::FlatTupleBase(std::move(t)...) {}

template <size_t I>
const typename ElemFromList<I, T...>::type& Get() const {
return static_cast<const FlatTupleElemBase<FlatTuple, I>*>(this)->value;
}

template <size_t I>
typename ElemFromList<I, T...>::type& Get() {
return static_cast<FlatTupleElemBase<FlatTuple, I>*>(this)->value;
}
};

GTEST_INTERNAL_DEPRECATED(
"INSTANTIATE_TEST_CASE_P is deprecated, please use "
"INSTANTIATE_TEST_SUITE_P")
constexpr bool InstantiateTestCase_P_IsDeprecated() { return true; }

GTEST_INTERNAL_DEPRECATED(
"TYPED_TEST_CASE_P is deprecated, please use "
"TYPED_TEST_SUITE_P")
constexpr bool TypedTestCase_P_IsDeprecated() { return true; }

GTEST_INTERNAL_DEPRECATED(
"TYPED_TEST_CASE is deprecated, please use "
"TYPED_TEST_SUITE")
constexpr bool TypedTestCaseIsDeprecated() { return true; }

GTEST_INTERNAL_DEPRECATED(
"REGISTER_TYPED_TEST_CASE_P is deprecated, please use "
"REGISTER_TYPED_TEST_SUITE_P")
constexpr bool RegisterTypedTestCase_P_IsDeprecated() { return true; }

GTEST_INTERNAL_DEPRECATED(
"INSTANTIATE_TYPED_TEST_CASE_P is deprecated, please use "
"INSTANTIATE_TYPED_TEST_SUITE_P")
constexpr bool InstantiateTypedTestCase_P_IsDeprecated() { return true; }

}  
}  

#define GTEST_MESSAGE_AT_(file, line, message, result_type) \
::testing::internal::AssertHelper(result_type, file, line, message) \
= ::testing::Message()

#define GTEST_MESSAGE_(message, result_type) \
GTEST_MESSAGE_AT_(__FILE__, __LINE__, message, result_type)

#define GTEST_FATAL_FAILURE_(message) \
return GTEST_MESSAGE_(message, ::testing::TestPartResult::kFatalFailure)

#define GTEST_NONFATAL_FAILURE_(message) \
GTEST_MESSAGE_(message, ::testing::TestPartResult::kNonFatalFailure)

#define GTEST_SUCCESS_(message) \
GTEST_MESSAGE_(message, ::testing::TestPartResult::kSuccess)

#define GTEST_SKIP_(message) \
return GTEST_MESSAGE_(message, ::testing::TestPartResult::kSkip)

#define GTEST_SUPPRESS_UNREACHABLE_CODE_WARNING_BELOW_(statement) \
if (::testing::internal::AlwaysTrue()) {                        \
statement;                                                    \
} else                                              \
static_assert(true, "")  

#if GTEST_HAS_EXCEPTIONS

namespace testing {
namespace internal {

class NeverThrown {
public:
const char* what() const noexcept {
return "this exception should never be thrown";
}
};

}  
}  

#if GTEST_HAS_RTTI

#define GTEST_EXCEPTION_TYPE_(e) ::testing::internal::GetTypeName(typeid(e))

#else  

#define GTEST_EXCEPTION_TYPE_(e) \
std::string { "an std::exception-derived error" }

#endif  

#define GTEST_TEST_THROW_CATCH_STD_EXCEPTION_(statement, expected_exception)   \
catch (typename std::conditional<                                            \
std::is_same<typename std::remove_cv<typename std::remove_reference<  \
expected_exception>::type>::type,                    \
std::exception>::value,                                  \
const ::testing::internal::NeverThrown&, const std::exception&>::type \
e) {                                                              \
gtest_msg.value = "Expected: " #statement                                  \
" throws an exception of type " #expected_exception      \
".\n  Actual: it throws ";                               \
gtest_msg.value += GTEST_EXCEPTION_TYPE_(e);                               \
gtest_msg.value += " with description \"";                                 \
gtest_msg.value += e.what();                                               \
gtest_msg.value += "\".";                                                  \
goto GTEST_CONCAT_TOKEN_(gtest_label_testthrow_, __LINE__);                \
}

#else  

#define GTEST_TEST_THROW_CATCH_STD_EXCEPTION_(statement, expected_exception)

#endif  

#define GTEST_TEST_THROW_(statement, expected_exception, fail)              \
GTEST_AMBIGUOUS_ELSE_BLOCKER_                                             \
if (::testing::internal::TrueWithString gtest_msg{}) {                    \
bool gtest_caught_expected = false;                                     \
try {                                                                   \
GTEST_SUPPRESS_UNREACHABLE_CODE_WARNING_BELOW_(statement);            \
} catch (expected_exception const&) {                                   \
gtest_caught_expected = true;                                         \
}                                                                       \
GTEST_TEST_THROW_CATCH_STD_EXCEPTION_(statement, expected_exception)    \
catch (...) {                                                           \
gtest_msg.value = "Expected: " #statement                             \
" throws an exception of type " #expected_exception \
".\n  Actual: it throws a different type.";         \
goto GTEST_CONCAT_TOKEN_(gtest_label_testthrow_, __LINE__);           \
}                                                                       \
if (!gtest_caught_expected) {                                           \
gtest_msg.value = "Expected: " #statement                             \
" throws an exception of type " #expected_exception \
".\n  Actual: it throws nothing.";                  \
goto GTEST_CONCAT_TOKEN_(gtest_label_testthrow_, __LINE__);           \
}                                                                       \
} else                                                          \
GTEST_CONCAT_TOKEN_(gtest_label_testthrow_, __LINE__)                   \
: fail(gtest_msg.value.c_str())

#if GTEST_HAS_EXCEPTIONS

#define GTEST_TEST_NO_THROW_CATCH_STD_EXCEPTION_()                \
catch (std::exception const& e) {                               \
gtest_msg.value = "it throws ";                               \
gtest_msg.value += GTEST_EXCEPTION_TYPE_(e);                  \
gtest_msg.value += " with description \"";                    \
gtest_msg.value += e.what();                                  \
gtest_msg.value += "\".";                                     \
goto GTEST_CONCAT_TOKEN_(gtest_label_testnothrow_, __LINE__); \
}

#else  

#define GTEST_TEST_NO_THROW_CATCH_STD_EXCEPTION_()

#endif  

#define GTEST_TEST_NO_THROW_(statement, fail) \
GTEST_AMBIGUOUS_ELSE_BLOCKER_ \
if (::testing::internal::TrueWithString gtest_msg{}) { \
try { \
GTEST_SUPPRESS_UNREACHABLE_CODE_WARNING_BELOW_(statement); \
} \
GTEST_TEST_NO_THROW_CATCH_STD_EXCEPTION_() \
catch (...) { \
gtest_msg.value = "it throws."; \
goto GTEST_CONCAT_TOKEN_(gtest_label_testnothrow_, __LINE__); \
} \
} else \
GTEST_CONCAT_TOKEN_(gtest_label_testnothrow_, __LINE__): \
fail(("Expected: " #statement " doesn't throw an exception.\n" \
"  Actual: " + gtest_msg.value).c_str())

#define GTEST_TEST_ANY_THROW_(statement, fail) \
GTEST_AMBIGUOUS_ELSE_BLOCKER_ \
if (::testing::internal::AlwaysTrue()) { \
bool gtest_caught_any = false; \
try { \
GTEST_SUPPRESS_UNREACHABLE_CODE_WARNING_BELOW_(statement); \
} \
catch (...) { \
gtest_caught_any = true; \
} \
if (!gtest_caught_any) { \
goto GTEST_CONCAT_TOKEN_(gtest_label_testanythrow_, __LINE__); \
} \
} else \
GTEST_CONCAT_TOKEN_(gtest_label_testanythrow_, __LINE__): \
fail("Expected: " #statement " throws an exception.\n" \
"  Actual: it doesn't.")


#define GTEST_TEST_BOOLEAN_(expression, text, actual, expected, fail) \
GTEST_AMBIGUOUS_ELSE_BLOCKER_ \
if (const ::testing::AssertionResult gtest_ar_ = \
::testing::AssertionResult(expression)) \
; \
else \
fail(::testing::internal::GetBoolAssertionFailureMessage(\
gtest_ar_, text, #actual, #expected).c_str())

#define GTEST_TEST_NO_FATAL_FAILURE_(statement, fail) \
GTEST_AMBIGUOUS_ELSE_BLOCKER_ \
if (::testing::internal::AlwaysTrue()) { \
::testing::internal::HasNewFatalFailureHelper gtest_fatal_failure_checker; \
GTEST_SUPPRESS_UNREACHABLE_CODE_WARNING_BELOW_(statement); \
if (gtest_fatal_failure_checker.has_new_fatal_failure()) { \
goto GTEST_CONCAT_TOKEN_(gtest_label_testnofatal_, __LINE__); \
} \
} else \
GTEST_CONCAT_TOKEN_(gtest_label_testnofatal_, __LINE__): \
fail("Expected: " #statement " doesn't generate new fatal " \
"failures in the current thread.\n" \
"  Actual: it does.")

#define GTEST_TEST_CLASS_NAME_(test_suite_name, test_name) \
test_suite_name##_##test_name##_Test

#define GTEST_TEST_(test_suite_name, test_name, parent_class, parent_id)      \
static_assert(sizeof(GTEST_STRINGIFY_(test_suite_name)) > 1,                \
"test_suite_name must not be empty");                         \
static_assert(sizeof(GTEST_STRINGIFY_(test_name)) > 1,                      \
"test_name must not be empty");                               \
class GTEST_TEST_CLASS_NAME_(test_suite_name, test_name)                    \
: public parent_class {                                                 \
public:                                                                    \
GTEST_TEST_CLASS_NAME_(test_suite_name, test_name)() = default;           \
~GTEST_TEST_CLASS_NAME_(test_suite_name, test_name)() override = default; \
GTEST_DISALLOW_COPY_AND_ASSIGN_(GTEST_TEST_CLASS_NAME_(test_suite_name,   \
test_name));       \
GTEST_DISALLOW_MOVE_AND_ASSIGN_(GTEST_TEST_CLASS_NAME_(test_suite_name,   \
test_name));       \
\
private:                                                                   \
void TestBody() override;                                                 \
static ::testing::TestInfo* const test_info_ GTEST_ATTRIBUTE_UNUSED_;     \
};                                                                          \
\
::testing::TestInfo* const GTEST_TEST_CLASS_NAME_(test_suite_name,          \
test_name)::test_info_ =  \
::testing::internal::MakeAndRegisterTestInfo(                           \
#test_suite_name, #test_name, nullptr, nullptr,                     \
::testing::internal::CodeLocation(__FILE__, __LINE__), (parent_id), \
::testing::internal::SuiteApiResolver<                              \
parent_class>::GetSetUpCaseOrSuite(__FILE__, __LINE__),         \
::testing::internal::SuiteApiResolver<                              \
parent_class>::GetTearDownCaseOrSuite(__FILE__, __LINE__),      \
new ::testing::internal::TestFactoryImpl<GTEST_TEST_CLASS_NAME_(    \
test_suite_name, test_name)>);                                  \
void GTEST_TEST_CLASS_NAME_(test_suite_name, test_name)::TestBody()

#endif  


#ifndef GTEST_INCLUDE_GTEST_GTEST_DEATH_TEST_H_
#define GTEST_INCLUDE_GTEST_GTEST_DEATH_TEST_H_


#ifndef GTEST_INCLUDE_GTEST_INTERNAL_GTEST_DEATH_TEST_INTERNAL_H_
#define GTEST_INCLUDE_GTEST_INTERNAL_GTEST_DEATH_TEST_INTERNAL_H_




#ifndef GTEST_INCLUDE_GTEST_GTEST_MATCHERS_H_
#define GTEST_INCLUDE_GTEST_GTEST_MATCHERS_H_

#include <memory>
#include <ostream>
#include <string>
#include <type_traits>





#ifndef GTEST_INCLUDE_GTEST_GTEST_PRINTERS_H_
#define GTEST_INCLUDE_GTEST_GTEST_PRINTERS_H_

#include <functional>
#include <ostream>  
#include <sstream>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#if GTEST_HAS_ABSL
#include "absl/strings/string_view.h"
#endif  

namespace testing {

namespace internal {

template <typename T>
void UniversalPrint(const T& value, ::std::ostream* os);

struct ContainerPrinter {
template <typename T,
typename = typename std::enable_if<
(sizeof(IsContainerTest<T>(0)) == sizeof(IsContainer)) &&
!IsRecursiveContainer<T>::value>::type>
static void PrintValue(const T& container, std::ostream* os) {
const size_t kMaxCount = 32;  
*os << '{';
size_t count = 0;
for (auto&& elem : container) {
if (count > 0) {
*os << ',';
if (count == kMaxCount) {  
*os << " ...";
break;
}
}
*os << ' ';
internal::UniversalPrint(elem, os);
++count;
}

if (count > 0) {
*os << ' ';
}
*os << '}';
}
};

struct FunctionPointerPrinter {
template <typename T, typename = typename std::enable_if<
std::is_function<T>::value>::type>
static void PrintValue(T* p, ::std::ostream* os) {
if (p == nullptr) {
*os << "NULL";
} else {
*os << reinterpret_cast<const void*>(p);
}
}
};

struct PointerPrinter {
template <typename T>
static void PrintValue(T* p, ::std::ostream* os) {
if (p == nullptr) {
*os << "NULL";
} else {
*os << p;
}
}
};

namespace internal_stream {

struct Sentinel;
template <typename Char, typename CharTraits, typename T>
Sentinel* operator<<(::std::basic_ostream<Char, CharTraits>& os, const T& x);

template <typename T>
constexpr bool UseStreamOperator() {
return !std::is_same<decltype(std::declval<std::ostream&>()
<< std::declval<const T&>()),
Sentinel*>::value;
}

}  

struct StreamPrinter {
template <typename T, typename = typename std::enable_if<
internal_stream::UseStreamOperator<T>()>::type>
static void PrintValue(const T& value, ::std::ostream* os) {
*os << value;
}
};

struct ProtobufPrinter {
static const size_t kProtobufOneLinerMaxLength = 50;

template <typename T, typename = typename std::enable_if<
internal::IsAProtocolMessage<T>::value>::type>
static void PrintValue(const T& value, ::std::ostream* os) {
std::string pretty_str = value.ShortDebugString();
if (pretty_str.length() > kProtobufOneLinerMaxLength) {
pretty_str = "\n" + value.DebugString();
}
*os << ("<" + pretty_str + ">");
}
};

struct ConvertibleToIntegerPrinter {
static void PrintValue(internal::BiggestInt value, ::std::ostream* os) {
*os << value;
}
};

struct ConvertibleToStringViewPrinter {
#if GTEST_INTERNAL_HAS_STRING_VIEW
static void PrintValue(internal::StringView value, ::std::ostream* os) {
internal::UniversalPrint(value, os);
}
#endif
};


GTEST_API_ void PrintBytesInObjectTo(const unsigned char* obj_bytes,
size_t count,
::std::ostream* os);
struct FallbackPrinter {
template <typename T>
static void PrintValue(const T& value, ::std::ostream* os) {
PrintBytesInObjectTo(
static_cast<const unsigned char*>(
reinterpret_cast<const void*>(std::addressof(value))),
sizeof(value), os);
}
};

template <typename T, typename E, typename Printer, typename... Printers>
struct FindFirstPrinter : FindFirstPrinter<T, E, Printers...> {};

template <typename T, typename Printer, typename... Printers>
struct FindFirstPrinter<
T, decltype(Printer::PrintValue(std::declval<const T&>(), nullptr)),
Printer, Printers...> {
using type = Printer;
};

template <typename T>
void PrintWithFallback(const T& value, ::std::ostream* os) {
using Printer = typename FindFirstPrinter<
T, void, ContainerPrinter, FunctionPointerPrinter, PointerPrinter,
StreamPrinter, ProtobufPrinter, ConvertibleToIntegerPrinter,
ConvertibleToStringViewPrinter, FallbackPrinter>::type;
Printer::PrintValue(value, os);
}


template <typename ToPrint, typename OtherOperand>
class FormatForComparison {
public:
static ::std::string Format(const ToPrint& value) {
return ::testing::PrintToString(value);
}
};

template <typename ToPrint, size_t N, typename OtherOperand>
class FormatForComparison<ToPrint[N], OtherOperand> {
public:
static ::std::string Format(const ToPrint* value) {
return FormatForComparison<const ToPrint*, OtherOperand>::Format(value);
}
};


#define GTEST_IMPL_FORMAT_C_STRING_AS_POINTER_(CharType)                \
template <typename OtherOperand>                                      \
class FormatForComparison<CharType*, OtherOperand> {                  \
public:                                                              \
static ::std::string Format(CharType* value) {                      \
return ::testing::PrintToString(static_cast<const void*>(value)); \
}                                                                   \
}

GTEST_IMPL_FORMAT_C_STRING_AS_POINTER_(char);
GTEST_IMPL_FORMAT_C_STRING_AS_POINTER_(const char);
GTEST_IMPL_FORMAT_C_STRING_AS_POINTER_(wchar_t);
GTEST_IMPL_FORMAT_C_STRING_AS_POINTER_(const wchar_t);
#ifdef __cpp_char8_t
GTEST_IMPL_FORMAT_C_STRING_AS_POINTER_(char8_t);
GTEST_IMPL_FORMAT_C_STRING_AS_POINTER_(const char8_t);
#endif
GTEST_IMPL_FORMAT_C_STRING_AS_POINTER_(char16_t);
GTEST_IMPL_FORMAT_C_STRING_AS_POINTER_(const char16_t);
GTEST_IMPL_FORMAT_C_STRING_AS_POINTER_(char32_t);
GTEST_IMPL_FORMAT_C_STRING_AS_POINTER_(const char32_t);

#undef GTEST_IMPL_FORMAT_C_STRING_AS_POINTER_


#define GTEST_IMPL_FORMAT_C_STRING_AS_STRING_(CharType, OtherStringType) \
template <>                                                           \
class FormatForComparison<CharType*, OtherStringType> {               \
public:                                                              \
static ::std::string Format(CharType* value) {                      \
return ::testing::PrintToString(value);                           \
}                                                                   \
}

GTEST_IMPL_FORMAT_C_STRING_AS_STRING_(char, ::std::string);
GTEST_IMPL_FORMAT_C_STRING_AS_STRING_(const char, ::std::string);
#ifdef __cpp_char8_t
GTEST_IMPL_FORMAT_C_STRING_AS_STRING_(char8_t, ::std::u8string);
GTEST_IMPL_FORMAT_C_STRING_AS_STRING_(const char8_t, ::std::u8string);
#endif
GTEST_IMPL_FORMAT_C_STRING_AS_STRING_(char16_t, ::std::u16string);
GTEST_IMPL_FORMAT_C_STRING_AS_STRING_(const char16_t, ::std::u16string);
GTEST_IMPL_FORMAT_C_STRING_AS_STRING_(char32_t, ::std::u32string);
GTEST_IMPL_FORMAT_C_STRING_AS_STRING_(const char32_t, ::std::u32string);

#if GTEST_HAS_STD_WSTRING
GTEST_IMPL_FORMAT_C_STRING_AS_STRING_(wchar_t, ::std::wstring);
GTEST_IMPL_FORMAT_C_STRING_AS_STRING_(const wchar_t, ::std::wstring);
#endif

#undef GTEST_IMPL_FORMAT_C_STRING_AS_STRING_

template <typename T1, typename T2>
std::string FormatForComparisonFailureMessage(
const T1& value, const T2& ) {
return FormatForComparison<T1, T2>::Format(value);
}

template <typename T>
class UniversalPrinter;

template <typename T>
void PrintTo(const T& value, ::std::ostream* os) {
internal::PrintWithFallback(value, os);
}


GTEST_API_ void PrintTo(unsigned char c, ::std::ostream* os);
GTEST_API_ void PrintTo(signed char c, ::std::ostream* os);
inline void PrintTo(char c, ::std::ostream* os) {
PrintTo(static_cast<unsigned char>(c), os);
}

inline void PrintTo(bool x, ::std::ostream* os) {
*os << (x ? "true" : "false");
}

GTEST_API_ void PrintTo(wchar_t wc, ::std::ostream* os);

GTEST_API_ void PrintTo(char32_t c, ::std::ostream* os);
inline void PrintTo(char16_t c, ::std::ostream* os) {
PrintTo(ImplicitCast_<char32_t>(c), os);
}
#ifdef __cpp_char8_t
inline void PrintTo(char8_t c, ::std::ostream* os) {
PrintTo(ImplicitCast_<char32_t>(c), os);
}
#endif

GTEST_API_ void PrintTo(const char* s, ::std::ostream* os);
inline void PrintTo(char* s, ::std::ostream* os) {
PrintTo(ImplicitCast_<const char*>(s), os);
}

inline void PrintTo(const signed char* s, ::std::ostream* os) {
PrintTo(ImplicitCast_<const void*>(s), os);
}
inline void PrintTo(signed char* s, ::std::ostream* os) {
PrintTo(ImplicitCast_<const void*>(s), os);
}
inline void PrintTo(const unsigned char* s, ::std::ostream* os) {
PrintTo(ImplicitCast_<const void*>(s), os);
}
inline void PrintTo(unsigned char* s, ::std::ostream* os) {
PrintTo(ImplicitCast_<const void*>(s), os);
}
#ifdef __cpp_char8_t
inline void PrintTo(const char8_t* s, ::std::ostream* os) {
PrintTo(ImplicitCast_<const void*>(s), os);
}
inline void PrintTo(char8_t* s, ::std::ostream* os) {
PrintTo(ImplicitCast_<const void*>(s), os);
}
#endif
inline void PrintTo(const char16_t* s, ::std::ostream* os) {
PrintTo(ImplicitCast_<const void*>(s), os);
}
inline void PrintTo(char16_t* s, ::std::ostream* os) {
PrintTo(ImplicitCast_<const void*>(s), os);
}
inline void PrintTo(const char32_t* s, ::std::ostream* os) {
PrintTo(ImplicitCast_<const void*>(s), os);
}
inline void PrintTo(char32_t* s, ::std::ostream* os) {
PrintTo(ImplicitCast_<const void*>(s), os);
}

#if !defined(_MSC_VER) || defined(_NATIVE_WCHAR_T_DEFINED)
GTEST_API_ void PrintTo(const wchar_t* s, ::std::ostream* os);
inline void PrintTo(wchar_t* s, ::std::ostream* os) {
PrintTo(ImplicitCast_<const wchar_t*>(s), os);
}
#endif


template <typename T>
void PrintRawArrayTo(const T a[], size_t count, ::std::ostream* os) {
UniversalPrint(a[0], os);
for (size_t i = 1; i != count; i++) {
*os << ", ";
UniversalPrint(a[i], os);
}
}

GTEST_API_ void PrintStringTo(const ::std::string&s, ::std::ostream* os);
inline void PrintTo(const ::std::string& s, ::std::ostream* os) {
PrintStringTo(s, os);
}

#if GTEST_HAS_STD_WSTRING
GTEST_API_ void PrintWideStringTo(const ::std::wstring&s, ::std::ostream* os);
inline void PrintTo(const ::std::wstring& s, ::std::ostream* os) {
PrintWideStringTo(s, os);
}
#endif  

#if GTEST_INTERNAL_HAS_STRING_VIEW
inline void PrintTo(internal::StringView sp, ::std::ostream* os) {
PrintTo(::std::string(sp), os);
}
#endif  

inline void PrintTo(std::nullptr_t, ::std::ostream* os) { *os << "(nullptr)"; }

template <typename T>
void PrintTo(std::reference_wrapper<T> ref, ::std::ostream* os) {
UniversalPrinter<T&>::Print(ref.get(), os);
}

template <typename T>
void PrintTupleTo(const T&, std::integral_constant<size_t, 0>,
::std::ostream*) {}

template <typename T, size_t I>
void PrintTupleTo(const T& t, std::integral_constant<size_t, I>,
::std::ostream* os) {
PrintTupleTo(t, std::integral_constant<size_t, I - 1>(), os);
GTEST_INTENTIONAL_CONST_COND_PUSH_()
if (I > 1) {
GTEST_INTENTIONAL_CONST_COND_POP_()
*os << ", ";
}
UniversalPrinter<typename std::tuple_element<I - 1, T>::type>::Print(
std::get<I - 1>(t), os);
}

template <typename... Types>
void PrintTo(const ::std::tuple<Types...>& t, ::std::ostream* os) {
*os << "(";
PrintTupleTo(t, std::integral_constant<size_t, sizeof...(Types)>(), os);
*os << ")";
}

template <typename T1, typename T2>
void PrintTo(const ::std::pair<T1, T2>& value, ::std::ostream* os) {
*os << '(';
UniversalPrinter<T1>::Print(value.first, os);
*os << ", ";
UniversalPrinter<T2>::Print(value.second, os);
*os << ')';
}

template <typename T>
class UniversalPrinter {
public:
GTEST_DISABLE_MSC_WARNINGS_PUSH_(4180)

static void Print(const T& value, ::std::ostream* os) {
PrintTo(value, os);
}

GTEST_DISABLE_MSC_WARNINGS_POP_()
};

#if GTEST_INTERNAL_HAS_ANY


template <>
class UniversalPrinter<Any> {
public:
static void Print(const Any& value, ::std::ostream* os) {
if (value.has_value()) {
*os << "value of type " << GetTypeName(value);
} else {
*os << "no value";
}
}

private:
static std::string GetTypeName(const Any& value) {
#if GTEST_HAS_RTTI
return internal::GetTypeName(value.type());
#else
static_cast<void>(value);  
return "<unknown_type>";
#endif  
}
};

#endif  

#if GTEST_INTERNAL_HAS_OPTIONAL


template <typename T>
class UniversalPrinter<Optional<T>> {
public:
static void Print(const Optional<T>& value, ::std::ostream* os) {
*os << '(';
if (!value) {
*os << "nullopt";
} else {
UniversalPrint(*value, os);
}
*os << ')';
}
};

#endif  

#if GTEST_INTERNAL_HAS_VARIANT


template <typename... T>
class UniversalPrinter<Variant<T...>> {
public:
static void Print(const Variant<T...>& value, ::std::ostream* os) {
*os << '(';
#if GTEST_HAS_ABSL
absl::visit(Visitor{os, value.index()}, value);
#else
std::visit(Visitor{os, value.index()}, value);
#endif  
*os << ')';
}

private:
struct Visitor {
template <typename U>
void operator()(const U& u) const {
*os << "'" << GetTypeName<U>() << "(index = " << index
<< ")' with value ";
UniversalPrint(u, os);
}
::std::ostream* os;
std::size_t index;
};
};

#endif  

template <typename T>
void UniversalPrintArray(const T* begin, size_t len, ::std::ostream* os) {
if (len == 0) {
*os << "{}";
} else {
*os << "{ ";
const size_t kThreshold = 18;
const size_t kChunkSize = 8;
if (len <= kThreshold) {
PrintRawArrayTo(begin, len, os);
} else {
PrintRawArrayTo(begin, kChunkSize, os);
*os << ", ..., ";
PrintRawArrayTo(begin + len - kChunkSize, kChunkSize, os);
}
*os << " }";
}
}
GTEST_API_ void UniversalPrintArray(
const char* begin, size_t len, ::std::ostream* os);

GTEST_API_ void UniversalPrintArray(
const wchar_t* begin, size_t len, ::std::ostream* os);

template <typename T, size_t N>
class UniversalPrinter<T[N]> {
public:
static void Print(const T (&a)[N], ::std::ostream* os) {
UniversalPrintArray(a, N, os);
}
};

template <typename T>
class UniversalPrinter<T&> {
public:
GTEST_DISABLE_MSC_WARNINGS_PUSH_(4180)

static void Print(const T& value, ::std::ostream* os) {
*os << "@" << reinterpret_cast<const void*>(&value) << " ";

UniversalPrint(value, os);
}

GTEST_DISABLE_MSC_WARNINGS_POP_()
};


template <typename T>
class UniversalTersePrinter {
public:
static void Print(const T& value, ::std::ostream* os) {
UniversalPrint(value, os);
}
};
template <typename T>
class UniversalTersePrinter<T&> {
public:
static void Print(const T& value, ::std::ostream* os) {
UniversalPrint(value, os);
}
};
template <typename T, size_t N>
class UniversalTersePrinter<T[N]> {
public:
static void Print(const T (&value)[N], ::std::ostream* os) {
UniversalPrinter<T[N]>::Print(value, os);
}
};
template <>
class UniversalTersePrinter<const char*> {
public:
static void Print(const char* str, ::std::ostream* os) {
if (str == nullptr) {
*os << "NULL";
} else {
UniversalPrint(std::string(str), os);
}
}
};
template <>
class UniversalTersePrinter<char*> {
public:
static void Print(char* str, ::std::ostream* os) {
UniversalTersePrinter<const char*>::Print(str, os);
}
};

#if GTEST_HAS_STD_WSTRING
template <>
class UniversalTersePrinter<const wchar_t*> {
public:
static void Print(const wchar_t* str, ::std::ostream* os) {
if (str == nullptr) {
*os << "NULL";
} else {
UniversalPrint(::std::wstring(str), os);
}
}
};
#endif

template <>
class UniversalTersePrinter<wchar_t*> {
public:
static void Print(wchar_t* str, ::std::ostream* os) {
UniversalTersePrinter<const wchar_t*>::Print(str, os);
}
};

template <typename T>
void UniversalTersePrint(const T& value, ::std::ostream* os) {
UniversalTersePrinter<T>::Print(value, os);
}

template <typename T>
void UniversalPrint(const T& value, ::std::ostream* os) {
typedef T T1;
UniversalPrinter<T1>::Print(value, os);
}

typedef ::std::vector< ::std::string> Strings;

template <typename Tuple>
void TersePrintPrefixToStrings(const Tuple&, std::integral_constant<size_t, 0>,
Strings*) {}
template <typename Tuple, size_t I>
void TersePrintPrefixToStrings(const Tuple& t,
std::integral_constant<size_t, I>,
Strings* strings) {
TersePrintPrefixToStrings(t, std::integral_constant<size_t, I - 1>(),
strings);
::std::stringstream ss;
UniversalTersePrint(std::get<I - 1>(t), &ss);
strings->push_back(ss.str());
}

template <typename Tuple>
Strings UniversalTersePrintTupleFieldsToStrings(const Tuple& value) {
Strings result;
TersePrintPrefixToStrings(
value, std::integral_constant<size_t, std::tuple_size<Tuple>::value>(),
&result);
return result;
}

}  

template <typename T>
::std::string PrintToString(const T& value) {
::std::stringstream ss;
internal::UniversalTersePrinter<T>::Print(value, &ss);
return ss.str();
}

}  


#ifndef GTEST_INCLUDE_GTEST_INTERNAL_CUSTOM_GTEST_PRINTERS_H_
#define GTEST_INCLUDE_GTEST_INTERNAL_CUSTOM_GTEST_PRINTERS_H_

#endif  

#endif  

#if defined(_MSC_VER) && _MSC_VER >= 1915
#define GTEST_MAYBE_5046_ 5046
#else
#define GTEST_MAYBE_5046_
#endif

GTEST_DISABLE_MSC_WARNINGS_PUSH_(
4251 GTEST_MAYBE_5046_ 
)

namespace testing {


class MatchResultListener {
public:
explicit MatchResultListener(::std::ostream* os) : stream_(os) {}
virtual ~MatchResultListener() = 0;  

template <typename T>
MatchResultListener& operator<<(const T& x) {
if (stream_ != nullptr) *stream_ << x;
return *this;
}

::std::ostream* stream() { return stream_; }

bool IsInterested() const { return stream_ != nullptr; }

private:
::std::ostream* const stream_;

GTEST_DISALLOW_COPY_AND_ASSIGN_(MatchResultListener);
};

inline MatchResultListener::~MatchResultListener() {
}

class MatcherDescriberInterface {
public:
virtual ~MatcherDescriberInterface() {}

virtual void DescribeTo(::std::ostream* os) const = 0;

virtual void DescribeNegationTo(::std::ostream* os) const {
*os << "not (";
DescribeTo(os);
*os << ")";
}
};

template <typename T>
class MatcherInterface : public MatcherDescriberInterface {
public:
virtual bool MatchAndExplain(T x, MatchResultListener* listener) const = 0;

};

namespace internal {

template <typename T>
class MatcherInterfaceAdapter : public MatcherInterface<const T&> {
public:
explicit MatcherInterfaceAdapter(const MatcherInterface<T>* impl)
: impl_(impl) {}
~MatcherInterfaceAdapter() override { delete impl_; }

void DescribeTo(::std::ostream* os) const override { impl_->DescribeTo(os); }

void DescribeNegationTo(::std::ostream* os) const override {
impl_->DescribeNegationTo(os);
}

bool MatchAndExplain(const T& x,
MatchResultListener* listener) const override {
return impl_->MatchAndExplain(x, listener);
}

private:
const MatcherInterface<T>* const impl_;

GTEST_DISALLOW_COPY_AND_ASSIGN_(MatcherInterfaceAdapter);
};

struct AnyEq {
template <typename A, typename B>
bool operator()(const A& a, const B& b) const { return a == b; }
};
struct AnyNe {
template <typename A, typename B>
bool operator()(const A& a, const B& b) const { return a != b; }
};
struct AnyLt {
template <typename A, typename B>
bool operator()(const A& a, const B& b) const { return a < b; }
};
struct AnyGt {
template <typename A, typename B>
bool operator()(const A& a, const B& b) const { return a > b; }
};
struct AnyLe {
template <typename A, typename B>
bool operator()(const A& a, const B& b) const { return a <= b; }
};
struct AnyGe {
template <typename A, typename B>
bool operator()(const A& a, const B& b) const { return a >= b; }
};

class DummyMatchResultListener : public MatchResultListener {
public:
DummyMatchResultListener() : MatchResultListener(nullptr) {}

private:
GTEST_DISALLOW_COPY_AND_ASSIGN_(DummyMatchResultListener);
};

class StreamMatchResultListener : public MatchResultListener {
public:
explicit StreamMatchResultListener(::std::ostream* os)
: MatchResultListener(os) {}

private:
GTEST_DISALLOW_COPY_AND_ASSIGN_(StreamMatchResultListener);
};

template <typename T>
class MatcherBase {
public:
bool MatchAndExplain(const T& x, MatchResultListener* listener) const {
return impl_->MatchAndExplain(x, listener);
}

bool Matches(const T& x) const {
DummyMatchResultListener dummy;
return MatchAndExplain(x, &dummy);
}

void DescribeTo(::std::ostream* os) const { impl_->DescribeTo(os); }

void DescribeNegationTo(::std::ostream* os) const {
impl_->DescribeNegationTo(os);
}

void ExplainMatchResultTo(const T& x, ::std::ostream* os) const {
StreamMatchResultListener listener(os);
MatchAndExplain(x, &listener);
}

const MatcherDescriberInterface* GetDescriber() const {
return impl_.get();
}

protected:
MatcherBase() {}

explicit MatcherBase(const MatcherInterface<const T&>* impl) : impl_(impl) {}

template <typename U>
explicit MatcherBase(
const MatcherInterface<U>* impl,
typename std::enable_if<!std::is_same<U, const U&>::value>::type* =
nullptr)
: impl_(new internal::MatcherInterfaceAdapter<U>(impl)) {}

MatcherBase(const MatcherBase&) = default;
MatcherBase& operator=(const MatcherBase&) = default;
MatcherBase(MatcherBase&&) = default;
MatcherBase& operator=(MatcherBase&&) = default;

virtual ~MatcherBase() {}

private:
std::shared_ptr<const MatcherInterface<const T&>> impl_;
};

}  

template <typename T>
class Matcher : public internal::MatcherBase<T> {
public:
explicit Matcher() {}  

explicit Matcher(const MatcherInterface<const T&>* impl)
: internal::MatcherBase<T>(impl) {}

template <typename U>
explicit Matcher(
const MatcherInterface<U>* impl,
typename std::enable_if<!std::is_same<U, const U&>::value>::type* =
nullptr)
: internal::MatcherBase<T>(impl) {}

Matcher(T value);  
};

template <>
class GTEST_API_ Matcher<const std::string&>
: public internal::MatcherBase<const std::string&> {
public:
Matcher() {}

explicit Matcher(const MatcherInterface<const std::string&>* impl)
: internal::MatcherBase<const std::string&>(impl) {}

Matcher(const std::string& s);  

Matcher(const char* s);  
};

template <>
class GTEST_API_ Matcher<std::string>
: public internal::MatcherBase<std::string> {
public:
Matcher() {}

explicit Matcher(const MatcherInterface<const std::string&>* impl)
: internal::MatcherBase<std::string>(impl) {}
explicit Matcher(const MatcherInterface<std::string>* impl)
: internal::MatcherBase<std::string>(impl) {}

Matcher(const std::string& s);  

Matcher(const char* s);  
};

#if GTEST_INTERNAL_HAS_STRING_VIEW
template <>
class GTEST_API_ Matcher<const internal::StringView&>
: public internal::MatcherBase<const internal::StringView&> {
public:
Matcher() {}

explicit Matcher(const MatcherInterface<const internal::StringView&>* impl)
: internal::MatcherBase<const internal::StringView&>(impl) {}

Matcher(const std::string& s);  

Matcher(const char* s);  

Matcher(internal::StringView s);  
};

template <>
class GTEST_API_ Matcher<internal::StringView>
: public internal::MatcherBase<internal::StringView> {
public:
Matcher() {}

explicit Matcher(const MatcherInterface<const internal::StringView&>* impl)
: internal::MatcherBase<internal::StringView>(impl) {}
explicit Matcher(const MatcherInterface<internal::StringView>* impl)
: internal::MatcherBase<internal::StringView>(impl) {}

Matcher(const std::string& s);  

Matcher(const char* s);  

Matcher(internal::StringView s);  
};
#endif  

template <typename T>
std::ostream& operator<<(std::ostream& os, const Matcher<T>& matcher) {
matcher.DescribeTo(&os);
return os;
}

template <class Impl>
class PolymorphicMatcher {
public:
explicit PolymorphicMatcher(const Impl& an_impl) : impl_(an_impl) {}

Impl& mutable_impl() { return impl_; }

const Impl& impl() const { return impl_; }

template <typename T>
operator Matcher<T>() const {
return Matcher<T>(new MonomorphicImpl<const T&>(impl_));
}

private:
template <typename T>
class MonomorphicImpl : public MatcherInterface<T> {
public:
explicit MonomorphicImpl(const Impl& impl) : impl_(impl) {}

void DescribeTo(::std::ostream* os) const override { impl_.DescribeTo(os); }

void DescribeNegationTo(::std::ostream* os) const override {
impl_.DescribeNegationTo(os);
}

bool MatchAndExplain(T x, MatchResultListener* listener) const override {
return impl_.MatchAndExplain(x, listener);
}

private:
const Impl impl_;
};

Impl impl_;
};

template <typename T>
inline Matcher<T> MakeMatcher(const MatcherInterface<T>* impl) {
return Matcher<T>(impl);
}

template <class Impl>
inline PolymorphicMatcher<Impl> MakePolymorphicMatcher(const Impl& impl) {
return PolymorphicMatcher<Impl>(impl);
}

namespace internal {
template <typename D, typename Rhs, typename Op>
class ComparisonBase {
public:
explicit ComparisonBase(const Rhs& rhs) : rhs_(rhs) {}
template <typename Lhs>
operator Matcher<Lhs>() const {
return Matcher<Lhs>(new Impl<const Lhs&>(rhs_));
}

private:
template <typename T>
static const T& Unwrap(const T& v) { return v; }
template <typename T>
static const T& Unwrap(std::reference_wrapper<T> v) { return v; }

template <typename Lhs, typename = Rhs>
class Impl : public MatcherInterface<Lhs> {
public:
explicit Impl(const Rhs& rhs) : rhs_(rhs) {}
bool MatchAndExplain(Lhs lhs,
MatchResultListener* ) const override {
return Op()(lhs, Unwrap(rhs_));
}
void DescribeTo(::std::ostream* os) const override {
*os << D::Desc() << " ";
UniversalPrint(Unwrap(rhs_), os);
}
void DescribeNegationTo(::std::ostream* os) const override {
*os << D::NegatedDesc() <<  " ";
UniversalPrint(Unwrap(rhs_), os);
}

private:
Rhs rhs_;
};
Rhs rhs_;
};

template <typename Rhs>
class EqMatcher : public ComparisonBase<EqMatcher<Rhs>, Rhs, AnyEq> {
public:
explicit EqMatcher(const Rhs& rhs)
: ComparisonBase<EqMatcher<Rhs>, Rhs, AnyEq>(rhs) { }
static const char* Desc() { return "is equal to"; }
static const char* NegatedDesc() { return "isn't equal to"; }
};
template <typename Rhs>
class NeMatcher : public ComparisonBase<NeMatcher<Rhs>, Rhs, AnyNe> {
public:
explicit NeMatcher(const Rhs& rhs)
: ComparisonBase<NeMatcher<Rhs>, Rhs, AnyNe>(rhs) { }
static const char* Desc() { return "isn't equal to"; }
static const char* NegatedDesc() { return "is equal to"; }
};
template <typename Rhs>
class LtMatcher : public ComparisonBase<LtMatcher<Rhs>, Rhs, AnyLt> {
public:
explicit LtMatcher(const Rhs& rhs)
: ComparisonBase<LtMatcher<Rhs>, Rhs, AnyLt>(rhs) { }
static const char* Desc() { return "is <"; }
static const char* NegatedDesc() { return "isn't <"; }
};
template <typename Rhs>
class GtMatcher : public ComparisonBase<GtMatcher<Rhs>, Rhs, AnyGt> {
public:
explicit GtMatcher(const Rhs& rhs)
: ComparisonBase<GtMatcher<Rhs>, Rhs, AnyGt>(rhs) { }
static const char* Desc() { return "is >"; }
static const char* NegatedDesc() { return "isn't >"; }
};
template <typename Rhs>
class LeMatcher : public ComparisonBase<LeMatcher<Rhs>, Rhs, AnyLe> {
public:
explicit LeMatcher(const Rhs& rhs)
: ComparisonBase<LeMatcher<Rhs>, Rhs, AnyLe>(rhs) { }
static const char* Desc() { return "is <="; }
static const char* NegatedDesc() { return "isn't <="; }
};
template <typename Rhs>
class GeMatcher : public ComparisonBase<GeMatcher<Rhs>, Rhs, AnyGe> {
public:
explicit GeMatcher(const Rhs& rhs)
: ComparisonBase<GeMatcher<Rhs>, Rhs, AnyGe>(rhs) { }
static const char* Desc() { return "is >="; }
static const char* NegatedDesc() { return "isn't >="; }
};

template <typename T, typename = typename std::enable_if<
std::is_constructible<std::string, T>::value>::type>
using StringLike = T;

class MatchesRegexMatcher {
public:
MatchesRegexMatcher(const RE* regex, bool full_match)
: regex_(regex), full_match_(full_match) {}

#if GTEST_INTERNAL_HAS_STRING_VIEW
bool MatchAndExplain(const internal::StringView& s,
MatchResultListener* listener) const {
return MatchAndExplain(std::string(s), listener);
}
#endif  

template <typename CharType>
bool MatchAndExplain(CharType* s, MatchResultListener* listener) const {
return s != nullptr && MatchAndExplain(std::string(s), listener);
}

template <class MatcheeStringType>
bool MatchAndExplain(const MatcheeStringType& s,
MatchResultListener* ) const {
const std::string& s2(s);
return full_match_ ? RE::FullMatch(s2, *regex_)
: RE::PartialMatch(s2, *regex_);
}

void DescribeTo(::std::ostream* os) const {
*os << (full_match_ ? "matches" : "contains") << " regular expression ";
UniversalPrinter<std::string>::Print(regex_->pattern(), os);
}

void DescribeNegationTo(::std::ostream* os) const {
*os << "doesn't " << (full_match_ ? "match" : "contain")
<< " regular expression ";
UniversalPrinter<std::string>::Print(regex_->pattern(), os);
}

private:
const std::shared_ptr<const RE> regex_;
const bool full_match_;
};
}  

inline PolymorphicMatcher<internal::MatchesRegexMatcher> MatchesRegex(
const internal::RE* regex) {
return MakePolymorphicMatcher(internal::MatchesRegexMatcher(regex, true));
}
template <typename T = std::string>
PolymorphicMatcher<internal::MatchesRegexMatcher> MatchesRegex(
const internal::StringLike<T>& regex) {
return MatchesRegex(new internal::RE(std::string(regex)));
}

inline PolymorphicMatcher<internal::MatchesRegexMatcher> ContainsRegex(
const internal::RE* regex) {
return MakePolymorphicMatcher(internal::MatchesRegexMatcher(regex, false));
}
template <typename T = std::string>
PolymorphicMatcher<internal::MatchesRegexMatcher> ContainsRegex(
const internal::StringLike<T>& regex) {
return ContainsRegex(new internal::RE(std::string(regex)));
}

template <typename T>
inline internal::EqMatcher<T> Eq(T x) { return internal::EqMatcher<T>(x); }

template <typename T>
Matcher<T>::Matcher(T value) { *this = Eq(value); }

template <typename Lhs, typename Rhs>
inline Matcher<Lhs> TypedEq(const Rhs& rhs) { return Eq(rhs); }

template <typename Rhs>
inline internal::GeMatcher<Rhs> Ge(Rhs x) {
return internal::GeMatcher<Rhs>(x);
}

template <typename Rhs>
inline internal::GtMatcher<Rhs> Gt(Rhs x) {
return internal::GtMatcher<Rhs>(x);
}

template <typename Rhs>
inline internal::LeMatcher<Rhs> Le(Rhs x) {
return internal::LeMatcher<Rhs>(x);
}

template <typename Rhs>
inline internal::LtMatcher<Rhs> Lt(Rhs x) {
return internal::LtMatcher<Rhs>(x);
}

template <typename Rhs>
inline internal::NeMatcher<Rhs> Ne(Rhs x) {
return internal::NeMatcher<Rhs>(x);
}
}  

GTEST_DISABLE_MSC_WARNINGS_POP_()  

#endif  

#include <stdio.h>
#include <memory>

namespace testing {
namespace internal {

GTEST_DECLARE_string_(internal_run_death_test);

const char kDeathTestStyleFlag[] = "death_test_style";
const char kDeathTestUseFork[] = "death_test_use_fork";
const char kInternalRunDeathTestFlag[] = "internal_run_death_test";

#if GTEST_HAS_DEATH_TEST

GTEST_DISABLE_MSC_WARNINGS_PUSH_(4251 \
)


class GTEST_API_ DeathTest {
public:
static bool Create(const char* statement, Matcher<const std::string&> matcher,
const char* file, int line, DeathTest** test);
DeathTest();
virtual ~DeathTest() { }

class ReturnSentinel {
public:
explicit ReturnSentinel(DeathTest* test) : test_(test) { }
~ReturnSentinel() { test_->Abort(TEST_ENCOUNTERED_RETURN_STATEMENT); }
private:
DeathTest* const test_;
GTEST_DISALLOW_COPY_AND_ASSIGN_(ReturnSentinel);
} GTEST_ATTRIBUTE_UNUSED_;

enum TestRole { OVERSEE_TEST, EXECUTE_TEST };

enum AbortReason {
TEST_ENCOUNTERED_RETURN_STATEMENT,
TEST_THREW_EXCEPTION,
TEST_DID_NOT_DIE
};

virtual TestRole AssumeRole() = 0;

virtual int Wait() = 0;

virtual bool Passed(bool exit_status_ok) = 0;

virtual void Abort(AbortReason reason) = 0;

static const char* LastMessage();

static void set_last_death_test_message(const std::string& message);

private:
static std::string last_death_test_message_;

GTEST_DISALLOW_COPY_AND_ASSIGN_(DeathTest);
};

GTEST_DISABLE_MSC_WARNINGS_POP_()  

class DeathTestFactory {
public:
virtual ~DeathTestFactory() { }
virtual bool Create(const char* statement,
Matcher<const std::string&> matcher, const char* file,
int line, DeathTest** test) = 0;
};

class DefaultDeathTestFactory : public DeathTestFactory {
public:
bool Create(const char* statement, Matcher<const std::string&> matcher,
const char* file, int line, DeathTest** test) override;
};

GTEST_API_ bool ExitedUnsuccessfully(int exit_status);

inline Matcher<const ::std::string&> MakeDeathTestMatcher(
::testing::internal::RE regex) {
return ContainsRegex(regex.pattern());
}
inline Matcher<const ::std::string&> MakeDeathTestMatcher(const char* regex) {
return ContainsRegex(regex);
}
inline Matcher<const ::std::string&> MakeDeathTestMatcher(
const ::std::string& regex) {
return ContainsRegex(regex);
}

inline Matcher<const ::std::string&> MakeDeathTestMatcher(
Matcher<const ::std::string&> matcher) {
return matcher;
}

# if GTEST_HAS_EXCEPTIONS
#  define GTEST_EXECUTE_DEATH_TEST_STATEMENT_(statement, death_test) \
try { \
GTEST_SUPPRESS_UNREACHABLE_CODE_WARNING_BELOW_(statement); \
} catch (const ::std::exception& gtest_exception) { \
fprintf(\
stderr, \
"\n%s: Caught std::exception-derived exception escaping the " \
"death test statement. Exception message: %s\n", \
::testing::internal::FormatFileLocation(__FILE__, __LINE__).c_str(), \
gtest_exception.what()); \
fflush(stderr); \
death_test->Abort(::testing::internal::DeathTest::TEST_THREW_EXCEPTION); \
} catch (...) { \
death_test->Abort(::testing::internal::DeathTest::TEST_THREW_EXCEPTION); \
}

# else
#  define GTEST_EXECUTE_DEATH_TEST_STATEMENT_(statement, death_test) \
GTEST_SUPPRESS_UNREACHABLE_CODE_WARNING_BELOW_(statement)

# endif

#define GTEST_DEATH_TEST_(statement, predicate, regex_or_matcher, fail)        \
GTEST_AMBIGUOUS_ELSE_BLOCKER_                                                \
if (::testing::internal::AlwaysTrue()) {                                     \
::testing::internal::DeathTest* gtest_dt;                                  \
if (!::testing::internal::DeathTest::Create(                               \
#statement,                                                        \
::testing::internal::MakeDeathTestMatcher(regex_or_matcher),       \
__FILE__, __LINE__, &gtest_dt)) {                                  \
goto GTEST_CONCAT_TOKEN_(gtest_label_, __LINE__);                        \
}                                                                          \
if (gtest_dt != nullptr) {                                                 \
std::unique_ptr< ::testing::internal::DeathTest> gtest_dt_ptr(gtest_dt); \
switch (gtest_dt->AssumeRole()) {                                        \
case ::testing::internal::DeathTest::OVERSEE_TEST:                     \
if (!gtest_dt->Passed(predicate(gtest_dt->Wait()))) {                \
goto GTEST_CONCAT_TOKEN_(gtest_label_, __LINE__);                  \
}                                                                    \
break;                                                               \
case ::testing::internal::DeathTest::EXECUTE_TEST: {                   \
::testing::internal::DeathTest::ReturnSentinel gtest_sentinel(       \
gtest_dt);                                                       \
GTEST_EXECUTE_DEATH_TEST_STATEMENT_(statement, gtest_dt);            \
gtest_dt->Abort(::testing::internal::DeathTest::TEST_DID_NOT_DIE);   \
break;                                                               \
}                                                                      \
default:                                                               \
break;                                                               \
}                                                                        \
}                                                                          \
} else                                                                       \
GTEST_CONCAT_TOKEN_(gtest_label_, __LINE__)                                \
: fail(::testing::internal::DeathTest::LastMessage())

#define GTEST_EXECUTE_STATEMENT_(statement, regex_or_matcher)    \
GTEST_AMBIGUOUS_ELSE_BLOCKER_                                  \
if (::testing::internal::AlwaysTrue()) {                       \
GTEST_SUPPRESS_UNREACHABLE_CODE_WARNING_BELOW_(statement);   \
} else if (!::testing::internal::AlwaysTrue()) {               \
::testing::internal::MakeDeathTestMatcher(regex_or_matcher); \
} else                                                         \
::testing::Message()

class InternalRunDeathTestFlag {
public:
InternalRunDeathTestFlag(const std::string& a_file,
int a_line,
int an_index,
int a_write_fd)
: file_(a_file), line_(a_line), index_(an_index),
write_fd_(a_write_fd) {}

~InternalRunDeathTestFlag() {
if (write_fd_ >= 0)
posix::Close(write_fd_);
}

const std::string& file() const { return file_; }
int line() const { return line_; }
int index() const { return index_; }
int write_fd() const { return write_fd_; }

private:
std::string file_;
int line_;
int index_;
int write_fd_;

GTEST_DISALLOW_COPY_AND_ASSIGN_(InternalRunDeathTestFlag);
};

InternalRunDeathTestFlag* ParseInternalRunDeathTestFlag();

#endif  

}  
}  

#endif  

namespace testing {

GTEST_DECLARE_string_(death_test_style);

#if GTEST_HAS_DEATH_TEST

namespace internal {

GTEST_API_ bool InDeathTestChild();

}  



# define ASSERT_EXIT(statement, predicate, regex) \
GTEST_DEATH_TEST_(statement, predicate, regex, GTEST_FATAL_FAILURE_)

# define EXPECT_EXIT(statement, predicate, regex) \
GTEST_DEATH_TEST_(statement, predicate, regex, GTEST_NONFATAL_FAILURE_)

# define ASSERT_DEATH(statement, regex) \
ASSERT_EXIT(statement, ::testing::internal::ExitedUnsuccessfully, regex)

# define EXPECT_DEATH(statement, regex) \
EXPECT_EXIT(statement, ::testing::internal::ExitedUnsuccessfully, regex)


class GTEST_API_ ExitedWithCode {
public:
explicit ExitedWithCode(int exit_code);
ExitedWithCode(const ExitedWithCode&) = default;
void operator=(const ExitedWithCode& other) = delete;
bool operator()(int exit_status) const;
private:
const int exit_code_;
};

# if !GTEST_OS_WINDOWS && !GTEST_OS_FUCHSIA
class GTEST_API_ KilledBySignal {
public:
explicit KilledBySignal(int signum);
bool operator()(int exit_status) const;
private:
const int signum_;
};
# endif  

# ifdef NDEBUG

#  define EXPECT_DEBUG_DEATH(statement, regex) \
GTEST_EXECUTE_STATEMENT_(statement, regex)

#  define ASSERT_DEBUG_DEATH(statement, regex) \
GTEST_EXECUTE_STATEMENT_(statement, regex)

# else

#  define EXPECT_DEBUG_DEATH(statement, regex) \
EXPECT_DEATH(statement, regex)

#  define ASSERT_DEBUG_DEATH(statement, regex) \
ASSERT_DEATH(statement, regex)

# endif  
#endif  

# define GTEST_UNSUPPORTED_DEATH_TEST(statement, regex, terminator) \
GTEST_AMBIGUOUS_ELSE_BLOCKER_ \
if (::testing::internal::AlwaysTrue()) { \
GTEST_LOG_(WARNING) \
<< "Death tests are not supported on this platform.\n" \
<< "Statement '" #statement "' cannot be verified."; \
} else if (::testing::internal::AlwaysFalse()) { \
::testing::internal::RE::PartialMatch(".*", (regex)); \
GTEST_SUPPRESS_UNREACHABLE_CODE_WARNING_BELOW_(statement); \
terminator; \
} else \
::testing::Message()

#if GTEST_HAS_DEATH_TEST
# define EXPECT_DEATH_IF_SUPPORTED(statement, regex) \
EXPECT_DEATH(statement, regex)
# define ASSERT_DEATH_IF_SUPPORTED(statement, regex) \
ASSERT_DEATH(statement, regex)
#else
# define EXPECT_DEATH_IF_SUPPORTED(statement, regex) \
GTEST_UNSUPPORTED_DEATH_TEST(statement, regex, )
# define ASSERT_DEATH_IF_SUPPORTED(statement, regex) \
GTEST_UNSUPPORTED_DEATH_TEST(statement, regex, return)
#endif

}  

#endif  
#ifndef GTEST_INCLUDE_GTEST_GTEST_PARAM_TEST_H_
#define GTEST_INCLUDE_GTEST_GTEST_PARAM_TEST_H_



#if 0


class FooTest : public ::testing::TestWithParam<const char*> {
};


TEST_P(FooTest, DoesBlah) {
EXPECT_TRUE(foo.Blah(GetParam()));
...
}

TEST_P(FooTest, HasBlahBlah) {
...
}


INSTANTIATE_TEST_SUITE_P(InstantiationName,
FooTest,
Values("meeny", "miny", "moe"));


const char* pets[] = {"cat", "dog"};
INSTANTIATE_TEST_SUITE_P(AnotherInstantiationName, FooTest, ValuesIn(pets));


class BaseTest : public ::testing::Test {
};

class DerivedTest : public BaseTest, public ::testing::WithParamInterface<int> {
};

TEST_F(BaseTest, HasFoo) {
}

TEST_P(DerivedTest, DoesBlah) {
EXPECT_TRUE(foo.Blah(GetParam()));
}

#endif  

#include <iterator>
#include <utility>





#ifndef GTEST_INCLUDE_GTEST_INTERNAL_GTEST_PARAM_UTIL_H_
#define GTEST_INCLUDE_GTEST_INTERNAL_GTEST_PARAM_UTIL_H_

#include <ctype.h>

#include <cassert>
#include <iterator>
#include <memory>
#include <set>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>


#ifndef GTEST_INCLUDE_GTEST_GTEST_TEST_PART_H_
#define GTEST_INCLUDE_GTEST_GTEST_TEST_PART_H_

#include <iosfwd>
#include <vector>

GTEST_DISABLE_MSC_WARNINGS_PUSH_(4251 \
)

namespace testing {

class GTEST_API_ TestPartResult {
public:
enum Type {
kSuccess,          
kNonFatalFailure,  
kFatalFailure,     
kSkip              
};

TestPartResult(Type a_type, const char* a_file_name, int a_line_number,
const char* a_message)
: type_(a_type),
file_name_(a_file_name == nullptr ? "" : a_file_name),
line_number_(a_line_number),
summary_(ExtractSummary(a_message)),
message_(a_message) {}

Type type() const { return type_; }

const char* file_name() const {
return file_name_.empty() ? nullptr : file_name_.c_str();
}

int line_number() const { return line_number_; }

const char* summary() const { return summary_.c_str(); }

const char* message() const { return message_.c_str(); }

bool skipped() const { return type_ == kSkip; }

bool passed() const { return type_ == kSuccess; }

bool nonfatally_failed() const { return type_ == kNonFatalFailure; }

bool fatally_failed() const { return type_ == kFatalFailure; }

bool failed() const { return fatally_failed() || nonfatally_failed(); }

private:
Type type_;

static std::string ExtractSummary(const char* message);

std::string file_name_;
int line_number_;
std::string summary_;  
std::string message_;  
};

std::ostream& operator<<(std::ostream& os, const TestPartResult& result);

class GTEST_API_ TestPartResultArray {
public:
TestPartResultArray() {}

void Append(const TestPartResult& result);

const TestPartResult& GetTestPartResult(int index) const;

int size() const;

private:
std::vector<TestPartResult> array_;

GTEST_DISALLOW_COPY_AND_ASSIGN_(TestPartResultArray);
};

class GTEST_API_ TestPartResultReporterInterface {
public:
virtual ~TestPartResultReporterInterface() {}

virtual void ReportTestPartResult(const TestPartResult& result) = 0;
};

namespace internal {

class GTEST_API_ HasNewFatalFailureHelper
: public TestPartResultReporterInterface {
public:
HasNewFatalFailureHelper();
~HasNewFatalFailureHelper() override;
void ReportTestPartResult(const TestPartResult& result) override;
bool has_new_fatal_failure() const { return has_new_fatal_failure_; }
private:
bool has_new_fatal_failure_;
TestPartResultReporterInterface* original_reporter_;

GTEST_DISALLOW_COPY_AND_ASSIGN_(HasNewFatalFailureHelper);
};

}  

}  

GTEST_DISABLE_MSC_WARNINGS_POP_()  

#endif  

namespace testing {
template <class ParamType>
struct TestParamInfo {
TestParamInfo(const ParamType& a_param, size_t an_index) :
param(a_param),
index(an_index) {}
ParamType param;
size_t index;
};

struct PrintToStringParamName {
template <class ParamType>
std::string operator()(const TestParamInfo<ParamType>& info) const {
return PrintToString(info.param);
}
};

namespace internal {


GTEST_API_ void ReportInvalidTestSuiteType(const char* test_suite_name,
CodeLocation code_location);

template <typename> class ParamGeneratorInterface;
template <typename> class ParamGenerator;

template <typename T>
class ParamIteratorInterface {
public:
virtual ~ParamIteratorInterface() {}
virtual const ParamGeneratorInterface<T>* BaseGenerator() const = 0;
virtual void Advance() = 0;
virtual ParamIteratorInterface* Clone() const = 0;
virtual const T* Current() const = 0;
virtual bool Equals(const ParamIteratorInterface& other) const = 0;
};

template <typename T>
class ParamIterator {
public:
typedef T value_type;
typedef const T& reference;
typedef ptrdiff_t difference_type;

ParamIterator(const ParamIterator& other) : impl_(other.impl_->Clone()) {}
ParamIterator& operator=(const ParamIterator& other) {
if (this != &other)
impl_.reset(other.impl_->Clone());
return *this;
}

const T& operator*() const { return *impl_->Current(); }
const T* operator->() const { return impl_->Current(); }
ParamIterator& operator++() {
impl_->Advance();
return *this;
}
ParamIterator operator++(int ) {
ParamIteratorInterface<T>* clone = impl_->Clone();
impl_->Advance();
return ParamIterator(clone);
}
bool operator==(const ParamIterator& other) const {
return impl_.get() == other.impl_.get() || impl_->Equals(*other.impl_);
}
bool operator!=(const ParamIterator& other) const {
return !(*this == other);
}

private:
friend class ParamGenerator<T>;
explicit ParamIterator(ParamIteratorInterface<T>* impl) : impl_(impl) {}
std::unique_ptr<ParamIteratorInterface<T> > impl_;
};

template <typename T>
class ParamGeneratorInterface {
public:
typedef T ParamType;

virtual ~ParamGeneratorInterface() {}

virtual ParamIteratorInterface<T>* Begin() const = 0;
virtual ParamIteratorInterface<T>* End() const = 0;
};

template<typename T>
class ParamGenerator {
public:
typedef ParamIterator<T> iterator;

explicit ParamGenerator(ParamGeneratorInterface<T>* impl) : impl_(impl) {}
ParamGenerator(const ParamGenerator& other) : impl_(other.impl_) {}

ParamGenerator& operator=(const ParamGenerator& other) {
impl_ = other.impl_;
return *this;
}

iterator begin() const { return iterator(impl_->Begin()); }
iterator end() const { return iterator(impl_->End()); }

private:
std::shared_ptr<const ParamGeneratorInterface<T> > impl_;
};

template <typename T, typename IncrementT>
class RangeGenerator : public ParamGeneratorInterface<T> {
public:
RangeGenerator(T begin, T end, IncrementT step)
: begin_(begin), end_(end),
step_(step), end_index_(CalculateEndIndex(begin, end, step)) {}
~RangeGenerator() override {}

ParamIteratorInterface<T>* Begin() const override {
return new Iterator(this, begin_, 0, step_);
}
ParamIteratorInterface<T>* End() const override {
return new Iterator(this, end_, end_index_, step_);
}

private:
class Iterator : public ParamIteratorInterface<T> {
public:
Iterator(const ParamGeneratorInterface<T>* base, T value, int index,
IncrementT step)
: base_(base), value_(value), index_(index), step_(step) {}
~Iterator() override {}

const ParamGeneratorInterface<T>* BaseGenerator() const override {
return base_;
}
void Advance() override {
value_ = static_cast<T>(value_ + step_);
index_++;
}
ParamIteratorInterface<T>* Clone() const override {
return new Iterator(*this);
}
const T* Current() const override { return &value_; }
bool Equals(const ParamIteratorInterface<T>& other) const override {
GTEST_CHECK_(BaseGenerator() == other.BaseGenerator())
<< "The program attempted to compare iterators "
<< "from different generators." << std::endl;
const int other_index =
CheckedDowncastToActualType<const Iterator>(&other)->index_;
return index_ == other_index;
}

private:
Iterator(const Iterator& other)
: ParamIteratorInterface<T>(),
base_(other.base_), value_(other.value_), index_(other.index_),
step_(other.step_) {}

void operator=(const Iterator& other);

const ParamGeneratorInterface<T>* const base_;
T value_;
int index_;
const IncrementT step_;
};  

static int CalculateEndIndex(const T& begin,
const T& end,
const IncrementT& step) {
int end_index = 0;
for (T i = begin; i < end; i = static_cast<T>(i + step))
end_index++;
return end_index;
}

void operator=(const RangeGenerator& other);

const T begin_;
const T end_;
const IncrementT step_;
const int end_index_;
};  


template <typename T>
class ValuesInIteratorRangeGenerator : public ParamGeneratorInterface<T> {
public:
template <typename ForwardIterator>
ValuesInIteratorRangeGenerator(ForwardIterator begin, ForwardIterator end)
: container_(begin, end) {}
~ValuesInIteratorRangeGenerator() override {}

ParamIteratorInterface<T>* Begin() const override {
return new Iterator(this, container_.begin());
}
ParamIteratorInterface<T>* End() const override {
return new Iterator(this, container_.end());
}

private:
typedef typename ::std::vector<T> ContainerType;

class Iterator : public ParamIteratorInterface<T> {
public:
Iterator(const ParamGeneratorInterface<T>* base,
typename ContainerType::const_iterator iterator)
: base_(base), iterator_(iterator) {}
~Iterator() override {}

const ParamGeneratorInterface<T>* BaseGenerator() const override {
return base_;
}
void Advance() override {
++iterator_;
value_.reset();
}
ParamIteratorInterface<T>* Clone() const override {
return new Iterator(*this);
}
const T* Current() const override {
if (value_.get() == nullptr) value_.reset(new T(*iterator_));
return value_.get();
}
bool Equals(const ParamIteratorInterface<T>& other) const override {
GTEST_CHECK_(BaseGenerator() == other.BaseGenerator())
<< "The program attempted to compare iterators "
<< "from different generators." << std::endl;
return iterator_ ==
CheckedDowncastToActualType<const Iterator>(&other)->iterator_;
}

private:
Iterator(const Iterator& other)
: ParamIteratorInterface<T>(),
base_(other.base_),
iterator_(other.iterator_) {}

const ParamGeneratorInterface<T>* const base_;
typename ContainerType::const_iterator iterator_;
mutable std::unique_ptr<const T> value_;
};  

void operator=(const ValuesInIteratorRangeGenerator& other);

const ContainerType container_;
};  

template <class ParamType>
std::string DefaultParamName(const TestParamInfo<ParamType>& info) {
Message name_stream;
name_stream << info.index;
return name_stream.GetString();
}

template <typename T = int>
void TestNotEmpty() {
static_assert(sizeof(T) == 0, "Empty arguments are not allowed.");
}
template <typename T = int>
void TestNotEmpty(const T&) {}

template <class TestClass>
class ParameterizedTestFactory : public TestFactoryBase {
public:
typedef typename TestClass::ParamType ParamType;
explicit ParameterizedTestFactory(ParamType parameter) :
parameter_(parameter) {}
Test* CreateTest() override {
TestClass::SetParam(&parameter_);
return new TestClass();
}

private:
const ParamType parameter_;

GTEST_DISALLOW_COPY_AND_ASSIGN_(ParameterizedTestFactory);
};

template <class ParamType>
class TestMetaFactoryBase {
public:
virtual ~TestMetaFactoryBase() {}

virtual TestFactoryBase* CreateTestFactory(ParamType parameter) = 0;
};

template <class TestSuite>
class TestMetaFactory
: public TestMetaFactoryBase<typename TestSuite::ParamType> {
public:
using ParamType = typename TestSuite::ParamType;

TestMetaFactory() {}

TestFactoryBase* CreateTestFactory(ParamType parameter) override {
return new ParameterizedTestFactory<TestSuite>(parameter);
}

private:
GTEST_DISALLOW_COPY_AND_ASSIGN_(TestMetaFactory);
};

class ParameterizedTestSuiteInfoBase {
public:
virtual ~ParameterizedTestSuiteInfoBase() {}

virtual const std::string& GetTestSuiteName() const = 0;
virtual TypeId GetTestSuiteTypeId() const = 0;
virtual void RegisterTests() = 0;

protected:
ParameterizedTestSuiteInfoBase() {}

private:
GTEST_DISALLOW_COPY_AND_ASSIGN_(ParameterizedTestSuiteInfoBase);
};

struct MarkAsIgnored {
explicit MarkAsIgnored(const char* test_suite);
};

GTEST_API_ void InsertSyntheticTestCase(const std::string& name,
CodeLocation location, bool has_test_p);

template <class TestSuite>
class ParameterizedTestSuiteInfo : public ParameterizedTestSuiteInfoBase {
public:
using ParamType = typename TestSuite::ParamType;
typedef ParamGenerator<ParamType>(GeneratorCreationFunc)();
using ParamNameGeneratorFunc = std::string(const TestParamInfo<ParamType>&);

explicit ParameterizedTestSuiteInfo(const char* name,
CodeLocation code_location)
: test_suite_name_(name), code_location_(code_location) {}

const std::string& GetTestSuiteName() const override {
return test_suite_name_;
}
TypeId GetTestSuiteTypeId() const override { return GetTypeId<TestSuite>(); }
void AddTestPattern(const char* test_suite_name, const char* test_base_name,
TestMetaFactoryBase<ParamType>* meta_factory,
CodeLocation code_location) {
tests_.push_back(std::shared_ptr<TestInfo>(new TestInfo(
test_suite_name, test_base_name, meta_factory, code_location)));
}
int AddTestSuiteInstantiation(const std::string& instantiation_name,
GeneratorCreationFunc* func,
ParamNameGeneratorFunc* name_func,
const char* file, int line) {
instantiations_.push_back(
InstantiationInfo(instantiation_name, func, name_func, file, line));
return 0;  
}
void RegisterTests() override {
bool generated_instantiations = false;

for (typename TestInfoContainer::iterator test_it = tests_.begin();
test_it != tests_.end(); ++test_it) {
std::shared_ptr<TestInfo> test_info = *test_it;
for (typename InstantiationContainer::iterator gen_it =
instantiations_.begin(); gen_it != instantiations_.end();
++gen_it) {
const std::string& instantiation_name = gen_it->name;
ParamGenerator<ParamType> generator((*gen_it->generator)());
ParamNameGeneratorFunc* name_func = gen_it->name_func;
const char* file = gen_it->file;
int line = gen_it->line;

std::string test_suite_name;
if ( !instantiation_name.empty() )
test_suite_name = instantiation_name + "/";
test_suite_name += test_info->test_suite_base_name;

size_t i = 0;
std::set<std::string> test_param_names;
for (typename ParamGenerator<ParamType>::iterator param_it =
generator.begin();
param_it != generator.end(); ++param_it, ++i) {
generated_instantiations = true;

Message test_name_stream;

std::string param_name = name_func(
TestParamInfo<ParamType>(*param_it, i));

GTEST_CHECK_(IsValidParamName(param_name))
<< "Parameterized test name '" << param_name
<< "' is invalid, in " << file
<< " line " << line << std::endl;

GTEST_CHECK_(test_param_names.count(param_name) == 0)
<< "Duplicate parameterized test name '" << param_name
<< "', in " << file << " line " << line << std::endl;

test_param_names.insert(param_name);

if (!test_info->test_base_name.empty()) {
test_name_stream << test_info->test_base_name << "/";
}
test_name_stream << param_name;
MakeAndRegisterTestInfo(
test_suite_name.c_str(), test_name_stream.GetString().c_str(),
nullptr,  
PrintToString(*param_it).c_str(), test_info->code_location,
GetTestSuiteTypeId(),
SuiteApiResolver<TestSuite>::GetSetUpCaseOrSuite(file, line),
SuiteApiResolver<TestSuite>::GetTearDownCaseOrSuite(file, line),
test_info->test_meta_factory->CreateTestFactory(*param_it));
}  
}  
}  

if (!generated_instantiations) {
InsertSyntheticTestCase(GetTestSuiteName(), code_location_,
!tests_.empty());
}
}    

private:
struct TestInfo {
TestInfo(const char* a_test_suite_base_name, const char* a_test_base_name,
TestMetaFactoryBase<ParamType>* a_test_meta_factory,
CodeLocation a_code_location)
: test_suite_base_name(a_test_suite_base_name),
test_base_name(a_test_base_name),
test_meta_factory(a_test_meta_factory),
code_location(a_code_location) {}

const std::string test_suite_base_name;
const std::string test_base_name;
const std::unique_ptr<TestMetaFactoryBase<ParamType> > test_meta_factory;
const CodeLocation code_location;
};
using TestInfoContainer = ::std::vector<std::shared_ptr<TestInfo> >;
struct InstantiationInfo {
InstantiationInfo(const std::string &name_in,
GeneratorCreationFunc* generator_in,
ParamNameGeneratorFunc* name_func_in,
const char* file_in,
int line_in)
: name(name_in),
generator(generator_in),
name_func(name_func_in),
file(file_in),
line(line_in) {}

std::string name;
GeneratorCreationFunc* generator;
ParamNameGeneratorFunc* name_func;
const char* file;
int line;
};
typedef ::std::vector<InstantiationInfo> InstantiationContainer;

static bool IsValidParamName(const std::string& name) {
if (name.empty())
return false;

for (std::string::size_type index = 0; index < name.size(); ++index) {
if (!isalnum(name[index]) && name[index] != '_')
return false;
}

return true;
}

const std::string test_suite_name_;
CodeLocation code_location_;
TestInfoContainer tests_;
InstantiationContainer instantiations_;

GTEST_DISALLOW_COPY_AND_ASSIGN_(ParameterizedTestSuiteInfo);
};  

#ifndef GTEST_REMOVE_LEGACY_TEST_CASEAPI_
template <class TestCase>
using ParameterizedTestCaseInfo = ParameterizedTestSuiteInfo<TestCase>;
#endif  

class ParameterizedTestSuiteRegistry {
public:
ParameterizedTestSuiteRegistry() {}
~ParameterizedTestSuiteRegistry() {
for (auto& test_suite_info : test_suite_infos_) {
delete test_suite_info;
}
}

template <class TestSuite>
ParameterizedTestSuiteInfo<TestSuite>* GetTestSuitePatternHolder(
const char* test_suite_name, CodeLocation code_location) {
ParameterizedTestSuiteInfo<TestSuite>* typed_test_info = nullptr;
for (auto& test_suite_info : test_suite_infos_) {
if (test_suite_info->GetTestSuiteName() == test_suite_name) {
if (test_suite_info->GetTestSuiteTypeId() != GetTypeId<TestSuite>()) {
ReportInvalidTestSuiteType(test_suite_name, code_location);
posix::Abort();
} else {
typed_test_info = CheckedDowncastToActualType<
ParameterizedTestSuiteInfo<TestSuite> >(test_suite_info);
}
break;
}
}
if (typed_test_info == nullptr) {
typed_test_info = new ParameterizedTestSuiteInfo<TestSuite>(
test_suite_name, code_location);
test_suite_infos_.push_back(typed_test_info);
}
return typed_test_info;
}
void RegisterTests() {
for (auto& test_suite_info : test_suite_infos_) {
test_suite_info->RegisterTests();
}
}
#ifndef GTEST_REMOVE_LEGACY_TEST_CASEAPI_
template <class TestCase>
ParameterizedTestCaseInfo<TestCase>* GetTestCasePatternHolder(
const char* test_case_name, CodeLocation code_location) {
return GetTestSuitePatternHolder<TestCase>(test_case_name, code_location);
}

#endif  

private:
using TestSuiteInfoContainer = ::std::vector<ParameterizedTestSuiteInfoBase*>;

TestSuiteInfoContainer test_suite_infos_;

GTEST_DISALLOW_COPY_AND_ASSIGN_(ParameterizedTestSuiteRegistry);
};

class TypeParameterizedTestSuiteRegistry {
public:
void RegisterTestSuite(const char* test_suite_name,
CodeLocation code_location);

void RegisterInstantiation(const char* test_suite_name);

void CheckForInstantiations();

private:
struct TypeParameterizedTestSuiteInfo {
explicit TypeParameterizedTestSuiteInfo(CodeLocation c)
: code_location(c), instantiated(false) {}

CodeLocation code_location;
bool instantiated;
};

std::map<std::string, TypeParameterizedTestSuiteInfo> suites_;
};

}  

template <class Container>
internal::ParamGenerator<typename Container::value_type> ValuesIn(
const Container& container);

namespace internal {

template <typename... Ts>
class ValueArray {
public:
ValueArray(Ts... v) : v_{std::move(v)...} {}

template <typename T>
operator ParamGenerator<T>() const {  
return ValuesIn(MakeVector<T>(MakeIndexSequence<sizeof...(Ts)>()));
}

private:
template <typename T, size_t... I>
std::vector<T> MakeVector(IndexSequence<I...>) const {
return std::vector<T>{static_cast<T>(v_.template Get<I>())...};
}

FlatTuple<Ts...> v_;
};

template <typename... T>
class CartesianProductGenerator
: public ParamGeneratorInterface<::std::tuple<T...>> {
public:
typedef ::std::tuple<T...> ParamType;

CartesianProductGenerator(const std::tuple<ParamGenerator<T>...>& g)
: generators_(g) {}
~CartesianProductGenerator() override {}

ParamIteratorInterface<ParamType>* Begin() const override {
return new Iterator(this, generators_, false);
}
ParamIteratorInterface<ParamType>* End() const override {
return new Iterator(this, generators_, true);
}

private:
template <class I>
class IteratorImpl;
template <size_t... I>
class IteratorImpl<IndexSequence<I...>>
: public ParamIteratorInterface<ParamType> {
public:
IteratorImpl(const ParamGeneratorInterface<ParamType>* base,
const std::tuple<ParamGenerator<T>...>& generators, bool is_end)
: base_(base),
begin_(std::get<I>(generators).begin()...),
end_(std::get<I>(generators).end()...),
current_(is_end ? end_ : begin_) {
ComputeCurrentValue();
}
~IteratorImpl() override {}

const ParamGeneratorInterface<ParamType>* BaseGenerator() const override {
return base_;
}
void Advance() override {
assert(!AtEnd());
++std::get<sizeof...(T) - 1>(current_);
AdvanceIfEnd<sizeof...(T) - 1>();
ComputeCurrentValue();
}
ParamIteratorInterface<ParamType>* Clone() const override {
return new IteratorImpl(*this);
}

const ParamType* Current() const override { return current_value_.get(); }

bool Equals(const ParamIteratorInterface<ParamType>& other) const override {
GTEST_CHECK_(BaseGenerator() == other.BaseGenerator())
<< "The program attempted to compare iterators "
<< "from different generators." << std::endl;
const IteratorImpl* typed_other =
CheckedDowncastToActualType<const IteratorImpl>(&other);

if (AtEnd() && typed_other->AtEnd()) return true;

bool same = true;
bool dummy[] = {
(same = same && std::get<I>(current_) ==
std::get<I>(typed_other->current_))...};
(void)dummy;
return same;
}

private:
template <size_t ThisI>
void AdvanceIfEnd() {
if (std::get<ThisI>(current_) != std::get<ThisI>(end_)) return;

bool last = ThisI == 0;
if (last) {
return;
}

constexpr size_t NextI = ThisI - (ThisI != 0);
std::get<ThisI>(current_) = std::get<ThisI>(begin_);
++std::get<NextI>(current_);
AdvanceIfEnd<NextI>();
}

void ComputeCurrentValue() {
if (!AtEnd())
current_value_ = std::make_shared<ParamType>(*std::get<I>(current_)...);
}
bool AtEnd() const {
bool at_end = false;
bool dummy[] = {
(at_end = at_end || std::get<I>(current_) == std::get<I>(end_))...};
(void)dummy;
return at_end;
}

const ParamGeneratorInterface<ParamType>* const base_;
std::tuple<typename ParamGenerator<T>::iterator...> begin_;
std::tuple<typename ParamGenerator<T>::iterator...> end_;
std::tuple<typename ParamGenerator<T>::iterator...> current_;
std::shared_ptr<ParamType> current_value_;
};

using Iterator = IteratorImpl<typename MakeIndexSequence<sizeof...(T)>::type>;

std::tuple<ParamGenerator<T>...> generators_;
};

template <class... Gen>
class CartesianProductHolder {
public:
CartesianProductHolder(const Gen&... g) : generators_(g...) {}
template <typename... T>
operator ParamGenerator<::std::tuple<T...>>() const {
return ParamGenerator<::std::tuple<T...>>(
new CartesianProductGenerator<T...>(generators_));
}

private:
std::tuple<Gen...> generators_;
};

}  
}  

#endif  

namespace testing {


template <typename T, typename IncrementT>
internal::ParamGenerator<T> Range(T start, T end, IncrementT step) {
return internal::ParamGenerator<T>(
new internal::RangeGenerator<T, IncrementT>(start, end, step));
}

template <typename T>
internal::ParamGenerator<T> Range(T start, T end) {
return Range(start, end, 1);
}

template <typename ForwardIterator>
internal::ParamGenerator<
typename std::iterator_traits<ForwardIterator>::value_type>
ValuesIn(ForwardIterator begin, ForwardIterator end) {
typedef typename std::iterator_traits<ForwardIterator>::value_type ParamType;
return internal::ParamGenerator<ParamType>(
new internal::ValuesInIteratorRangeGenerator<ParamType>(begin, end));
}

template <typename T, size_t N>
internal::ParamGenerator<T> ValuesIn(const T (&array)[N]) {
return ValuesIn(array, array + N);
}

template <class Container>
internal::ParamGenerator<typename Container::value_type> ValuesIn(
const Container& container) {
return ValuesIn(container.begin(), container.end());
}

template <typename... T>
internal::ValueArray<T...> Values(T... v) {
return internal::ValueArray<T...>(std::move(v)...);
}

inline internal::ParamGenerator<bool> Bool() {
return Values(false, true);
}

template <typename... Generator>
internal::CartesianProductHolder<Generator...> Combine(const Generator&... g) {
return internal::CartesianProductHolder<Generator...>(g...);
}

#define TEST_P(test_suite_name, test_name)                                     \
class GTEST_TEST_CLASS_NAME_(test_suite_name, test_name)                     \
: public test_suite_name {                                               \
public:                                                                     \
GTEST_TEST_CLASS_NAME_(test_suite_name, test_name)() {}                    \
void TestBody() override;                                                  \
\
private:                                                                    \
static int AddToRegistry() {                                               \
::testing::UnitTest::GetInstance()                                       \
->parameterized_test_registry()                                      \
.GetTestSuitePatternHolder<test_suite_name>(                         \
GTEST_STRINGIFY_(test_suite_name),                               \
::testing::internal::CodeLocation(__FILE__, __LINE__))           \
->AddTestPattern(                                                    \
GTEST_STRINGIFY_(test_suite_name), GTEST_STRINGIFY_(test_name),  \
new ::testing::internal::TestMetaFactory<GTEST_TEST_CLASS_NAME_( \
test_suite_name, test_name)>(),                              \
::testing::internal::CodeLocation(__FILE__, __LINE__));          \
return 0;                                                                \
}                                                                          \
static int gtest_registering_dummy_ GTEST_ATTRIBUTE_UNUSED_;               \
GTEST_DISALLOW_COPY_AND_ASSIGN_(GTEST_TEST_CLASS_NAME_(test_suite_name,    \
test_name));        \
};                                                                           \
int GTEST_TEST_CLASS_NAME_(test_suite_name,                                  \
test_name)::gtest_registering_dummy_ =            \
GTEST_TEST_CLASS_NAME_(test_suite_name, test_name)::AddToRegistry();     \
void GTEST_TEST_CLASS_NAME_(test_suite_name, test_name)::TestBody()


#define GTEST_EXPAND_(arg) arg
#define GTEST_GET_FIRST_(first, ...) first
#define GTEST_GET_SECOND_(first, second, ...) second

#define INSTANTIATE_TEST_SUITE_P(prefix, test_suite_name, ...)                \
static ::testing::internal::ParamGenerator<test_suite_name::ParamType>      \
gtest_##prefix##test_suite_name##_EvalGenerator_() {                    \
return GTEST_EXPAND_(GTEST_GET_FIRST_(__VA_ARGS__, DUMMY_PARAM_));        \
}                                                                           \
static ::std::string gtest_##prefix##test_suite_name##_EvalGenerateName_(   \
const ::testing::TestParamInfo<test_suite_name::ParamType>& info) {     \
if (::testing::internal::AlwaysFalse()) {                                 \
::testing::internal::TestNotEmpty(GTEST_EXPAND_(GTEST_GET_SECOND_(      \
__VA_ARGS__,                                                        \
::testing::internal::DefaultParamName<test_suite_name::ParamType>,  \
DUMMY_PARAM_)));                                                    \
auto t = std::make_tuple(__VA_ARGS__);                                  \
static_assert(std::tuple_size<decltype(t)>::value <= 2,                 \
"Too Many Args!");                                        \
}                                                                         \
return ((GTEST_EXPAND_(GTEST_GET_SECOND_(                                 \
__VA_ARGS__,                                                          \
::testing::internal::DefaultParamName<test_suite_name::ParamType>,    \
DUMMY_PARAM_))))(info);                                               \
}                                                                           \
static int gtest_##prefix##test_suite_name##_dummy_                         \
GTEST_ATTRIBUTE_UNUSED_ =                                               \
::testing::UnitTest::GetInstance()                                  \
->parameterized_test_registry()                                 \
.GetTestSuitePatternHolder<test_suite_name>(                    \
GTEST_STRINGIFY_(test_suite_name),                          \
::testing::internal::CodeLocation(__FILE__, __LINE__))      \
->AddTestSuiteInstantiation(                                    \
GTEST_STRINGIFY_(prefix),                                   \
&gtest_##prefix##test_suite_name##_EvalGenerator_,          \
&gtest_##prefix##test_suite_name##_EvalGenerateName_,       \
__FILE__, __LINE__)


#define GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(T)                   \
namespace gtest_do_not_use_outside_namespace_scope {}                   \
static const ::testing::internal::MarkAsIgnored gtest_allow_ignore_##T( \
GTEST_STRINGIFY_(T))

#ifndef GTEST_REMOVE_LEGACY_TEST_CASEAPI_
#define INSTANTIATE_TEST_CASE_P                                            \
static_assert(::testing::internal::InstantiateTestCase_P_IsDeprecated(), \
"");                                                       \
INSTANTIATE_TEST_SUITE_P
#endif  

}  

#endif  


#ifndef GTEST_INCLUDE_GTEST_GTEST_PROD_H_
#define GTEST_INCLUDE_GTEST_GTEST_PROD_H_


#define FRIEND_TEST(test_case_name, test_name)\
friend class test_case_name##_##test_name##_Test

#endif  


#ifndef GTEST_INCLUDE_GTEST_GTEST_TYPED_TEST_H_
#define GTEST_INCLUDE_GTEST_GTEST_TYPED_TEST_H_



#if 0

template <typename T>
class FooTest : public testing::Test {
public:
...
typedef std::list<T> List;
static T shared_;
T value_;
};

typedef testing::Types<char, int, unsigned int> MyTypes;
TYPED_TEST_SUITE(FooTest, MyTypes);


TYPED_TEST(FooTest, DoesBlah) {
TypeParam n = this->value_;

n += TestFixture::shared_;

typename TestFixture::List values;
values.push_back(n);
...
}

TYPED_TEST(FooTest, HasPropertyA) { ... }


#endif  


#if 0

template <typename T>
class FooTest : public testing::Test {
...
};

TYPED_TEST_SUITE_P(FooTest);

TYPED_TEST_P(FooTest, DoesBlah) {
TypeParam n = 0;
...
}

TYPED_TEST_P(FooTest, HasPropertyA) { ... }

REGISTER_TYPED_TEST_SUITE_P(FooTest,
DoesBlah, HasPropertyA);

typedef testing::Types<char, int, unsigned int> MyTypes;
INSTANTIATE_TYPED_TEST_SUITE_P(My, FooTest, MyTypes);


#endif  



#if GTEST_HAS_TYPED_TEST

#define GTEST_TYPE_PARAMS_(TestSuiteName) gtest_type_params_##TestSuiteName##_

#define GTEST_NAME_GENERATOR_(TestSuiteName) \
gtest_type_params_##TestSuiteName##_NameGenerator

#define TYPED_TEST_SUITE(CaseName, Types, ...)                          \
typedef ::testing::internal::GenerateTypeList<Types>::type            \
GTEST_TYPE_PARAMS_(CaseName);                                     \
typedef ::testing::internal::NameGeneratorSelector<__VA_ARGS__>::type \
GTEST_NAME_GENERATOR_(CaseName)

#define TYPED_TEST(CaseName, TestName)                                        \
static_assert(sizeof(GTEST_STRINGIFY_(TestName)) > 1,                       \
"test-name must not be empty");                               \
template <typename gtest_TypeParam_>                                        \
class GTEST_TEST_CLASS_NAME_(CaseName, TestName)                            \
: public CaseName<gtest_TypeParam_> {                                   \
private:                                                                   \
typedef CaseName<gtest_TypeParam_> TestFixture;                           \
typedef gtest_TypeParam_ TypeParam;                                       \
void TestBody() override;                                                 \
};                                                                          \
static bool gtest_##CaseName##_##TestName##_registered_                     \
GTEST_ATTRIBUTE_UNUSED_ = ::testing::internal::TypeParameterizedTest<   \
CaseName,                                                           \
::testing::internal::TemplateSel<GTEST_TEST_CLASS_NAME_(CaseName,   \
TestName)>, \
GTEST_TYPE_PARAMS_(                                                 \
CaseName)>::Register("",                                        \
::testing::internal::CodeLocation(         \
__FILE__, __LINE__),                   \
GTEST_STRINGIFY_(CaseName),                \
GTEST_STRINGIFY_(TestName), 0,             \
::testing::internal::GenerateNames<        \
GTEST_NAME_GENERATOR_(CaseName),       \
GTEST_TYPE_PARAMS_(CaseName)>());      \
template <typename gtest_TypeParam_>                                        \
void GTEST_TEST_CLASS_NAME_(CaseName,                                       \
TestName)<gtest_TypeParam_>::TestBody()

#ifndef GTEST_REMOVE_LEGACY_TEST_CASEAPI_
#define TYPED_TEST_CASE                                                \
static_assert(::testing::internal::TypedTestCaseIsDeprecated(), ""); \
TYPED_TEST_SUITE
#endif  

#endif  


#if GTEST_HAS_TYPED_TEST_P

#define GTEST_SUITE_NAMESPACE_(TestSuiteName) gtest_suite_##TestSuiteName##_

#define GTEST_TYPED_TEST_SUITE_P_STATE_(TestSuiteName) \
gtest_typed_test_suite_p_state_##TestSuiteName##_

#define GTEST_REGISTERED_TEST_NAMES_(TestSuiteName) \
gtest_registered_test_names_##TestSuiteName##_

#define TYPED_TEST_SUITE_P(SuiteName)              \
static ::testing::internal::TypedTestSuitePState \
GTEST_TYPED_TEST_SUITE_P_STATE_(SuiteName)

#ifndef GTEST_REMOVE_LEGACY_TEST_CASEAPI_
#define TYPED_TEST_CASE_P                                                 \
static_assert(::testing::internal::TypedTestCase_P_IsDeprecated(), ""); \
TYPED_TEST_SUITE_P
#endif  

#define TYPED_TEST_P(SuiteName, TestName)                             \
namespace GTEST_SUITE_NAMESPACE_(SuiteName) {                       \
template <typename gtest_TypeParam_>                              \
class TestName : public SuiteName<gtest_TypeParam_> {             \
private:                                                         \
typedef SuiteName<gtest_TypeParam_> TestFixture;                \
typedef gtest_TypeParam_ TypeParam;                             \
void TestBody() override;                                       \
};                                                                \
static bool gtest_##TestName##_defined_ GTEST_ATTRIBUTE_UNUSED_ = \
GTEST_TYPED_TEST_SUITE_P_STATE_(SuiteName).AddTestName(       \
__FILE__, __LINE__, GTEST_STRINGIFY_(SuiteName),          \
GTEST_STRINGIFY_(TestName));                              \
}                                                                   \
template <typename gtest_TypeParam_>                                \
void GTEST_SUITE_NAMESPACE_(                                        \
SuiteName)::TestName<gtest_TypeParam_>::TestBody()

#define REGISTER_TYPED_TEST_SUITE_P(SuiteName, ...)                         \
namespace GTEST_SUITE_NAMESPACE_(SuiteName) {                             \
typedef ::testing::internal::Templates<__VA_ARGS__> gtest_AllTests_;    \
}                                                                         \
static const char* const GTEST_REGISTERED_TEST_NAMES_(                    \
SuiteName) GTEST_ATTRIBUTE_UNUSED_ =                                  \
GTEST_TYPED_TEST_SUITE_P_STATE_(SuiteName).VerifyRegisteredTestNames( \
GTEST_STRINGIFY_(SuiteName), __FILE__, __LINE__, #__VA_ARGS__)

#ifndef GTEST_REMOVE_LEGACY_TEST_CASEAPI_
#define REGISTER_TYPED_TEST_CASE_P                                           \
static_assert(::testing::internal::RegisterTypedTestCase_P_IsDeprecated(), \
"");                                                         \
REGISTER_TYPED_TEST_SUITE_P
#endif  

#define INSTANTIATE_TYPED_TEST_SUITE_P(Prefix, SuiteName, Types, ...)       \
static_assert(sizeof(GTEST_STRINGIFY_(Prefix)) > 1,                       \
"test-suit-prefix must not be empty");                      \
static bool gtest_##Prefix##_##SuiteName GTEST_ATTRIBUTE_UNUSED_ =        \
::testing::internal::TypeParameterizedTestSuite<                      \
SuiteName, GTEST_SUITE_NAMESPACE_(SuiteName)::gtest_AllTests_,    \
::testing::internal::GenerateTypeList<Types>::type>::             \
Register(GTEST_STRINGIFY_(Prefix),                                \
::testing::internal::CodeLocation(__FILE__, __LINE__),   \
&GTEST_TYPED_TEST_SUITE_P_STATE_(SuiteName),             \
GTEST_STRINGIFY_(SuiteName),                             \
GTEST_REGISTERED_TEST_NAMES_(SuiteName),                 \
::testing::internal::GenerateNames<                      \
::testing::internal::NameGeneratorSelector<          \
__VA_ARGS__>::type,                              \
::testing::internal::GenerateTypeList<Types>::type>())

#ifndef GTEST_REMOVE_LEGACY_TEST_CASEAPI_
#define INSTANTIATE_TYPED_TEST_CASE_P                                      \
static_assert(                                                           \
::testing::internal::InstantiateTypedTestCase_P_IsDeprecated(), ""); \
INSTANTIATE_TYPED_TEST_SUITE_P
#endif  

#endif  

#endif  

GTEST_DISABLE_MSC_WARNINGS_PUSH_(4251 \
)

namespace testing {

#ifdef _MSC_VER
# pragma warning(push)
# pragma warning(disable:4805)
# pragma warning(disable:4100)
#endif



GTEST_DECLARE_bool_(also_run_disabled_tests);

GTEST_DECLARE_bool_(break_on_failure);

GTEST_DECLARE_bool_(catch_exceptions);

GTEST_DECLARE_string_(color);

GTEST_DECLARE_bool_(fail_fast);

GTEST_DECLARE_string_(filter);

GTEST_DECLARE_bool_(install_failure_signal_handler);

GTEST_DECLARE_bool_(list_tests);

GTEST_DECLARE_string_(output);

GTEST_DECLARE_bool_(brief);

GTEST_DECLARE_bool_(print_time);

GTEST_DECLARE_bool_(print_utf8);

GTEST_DECLARE_int32_(random_seed);

GTEST_DECLARE_int32_(repeat);

GTEST_DECLARE_bool_(show_internal_stack_frames);

GTEST_DECLARE_bool_(shuffle);

GTEST_DECLARE_int32_(stack_trace_depth);

GTEST_DECLARE_bool_(throw_on_failure);

GTEST_DECLARE_string_(stream_result_to);

#if GTEST_USE_OWN_FLAGFILE_FLAG_
GTEST_DECLARE_string_(flagfile);
#endif  

const int kMaxStackTraceDepth = 100;

namespace internal {

class AssertHelper;
class DefaultGlobalTestPartResultReporter;
class ExecDeathTest;
class NoExecDeathTest;
class FinalSuccessChecker;
class GTestFlagSaver;
class StreamingListenerTest;
class TestResultAccessor;
class TestEventListenersAccessor;
class TestEventRepeater;
class UnitTestRecordPropertyTestHelper;
class WindowsDeathTest;
class FuchsiaDeathTest;
class UnitTestImpl* GetUnitTestImpl();
void ReportFailureInUnknownLocation(TestPartResult::Type result_type,
const std::string& message);
std::set<std::string>* GetIgnoredParameterizedTestSuites();

}  

class Test;
class TestSuite;

#ifndef GTEST_REMOVE_LEGACY_TEST_CASEAPI_
using TestCase = TestSuite;
#endif
class TestInfo;
class UnitTest;

class GTEST_API_ AssertionResult {
public:
AssertionResult(const AssertionResult& other);

#if defined(_MSC_VER) && (_MSC_VER < 1910 || _MSC_VER >= 1920)
GTEST_DISABLE_MSC_WARNINGS_PUSH_(4800 )
#endif

template <typename T>
explicit AssertionResult(
const T& success,
typename std::enable_if<
!std::is_convertible<T, AssertionResult>::value>::type*

= nullptr)
: success_(success) {}

#if defined(_MSC_VER) && (_MSC_VER < 1910 || _MSC_VER >= 1920)
GTEST_DISABLE_MSC_WARNINGS_POP_()
#endif

AssertionResult& operator=(AssertionResult other) {
swap(other);
return *this;
}

operator bool() const { return success_; }  

AssertionResult operator!() const;

const char* message() const {
return message_.get() != nullptr ? message_->c_str() : "";
}
const char* failure_message() const { return message(); }

template <typename T> AssertionResult& operator<<(const T& value) {
AppendMessage(Message() << value);
return *this;
}

AssertionResult& operator<<(
::std::ostream& (*basic_manipulator)(::std::ostream& stream)) {
AppendMessage(Message() << basic_manipulator);
return *this;
}

private:
void AppendMessage(const Message& a_message) {
if (message_.get() == nullptr) message_.reset(new ::std::string);
message_->append(a_message.GetString().c_str());
}

void swap(AssertionResult& other);

bool success_;
std::unique_ptr< ::std::string> message_;
};

GTEST_API_ AssertionResult AssertionSuccess();

GTEST_API_ AssertionResult AssertionFailure();

GTEST_API_ AssertionResult AssertionFailure(const Message& msg);

}  



#ifndef GTEST_INCLUDE_GTEST_GTEST_PRED_IMPL_H_
#define GTEST_INCLUDE_GTEST_GTEST_PRED_IMPL_H_


namespace testing {



#define GTEST_ASSERT_(expression, on_failure) \
GTEST_AMBIGUOUS_ELSE_BLOCKER_ \
if (const ::testing::AssertionResult gtest_ar = (expression)) \
; \
else \
on_failure(gtest_ar.failure_message())


template <typename Pred,
typename T1>
AssertionResult AssertPred1Helper(const char* pred_text,
const char* e1,
Pred pred,
const T1& v1) {
if (pred(v1)) return AssertionSuccess();

return AssertionFailure()
<< pred_text << "(" << e1 << ") evaluates to false, where"
<< "\n"
<< e1 << " evaluates to " << ::testing::PrintToString(v1);
}

#define GTEST_PRED_FORMAT1_(pred_format, v1, on_failure)\
GTEST_ASSERT_(pred_format(#v1, v1), \
on_failure)

#define GTEST_PRED1_(pred, v1, on_failure)\
GTEST_ASSERT_(::testing::AssertPred1Helper(#pred, \
#v1, \
pred, \
v1), on_failure)

#define EXPECT_PRED_FORMAT1(pred_format, v1) \
GTEST_PRED_FORMAT1_(pred_format, v1, GTEST_NONFATAL_FAILURE_)
#define EXPECT_PRED1(pred, v1) \
GTEST_PRED1_(pred, v1, GTEST_NONFATAL_FAILURE_)
#define ASSERT_PRED_FORMAT1(pred_format, v1) \
GTEST_PRED_FORMAT1_(pred_format, v1, GTEST_FATAL_FAILURE_)
#define ASSERT_PRED1(pred, v1) \
GTEST_PRED1_(pred, v1, GTEST_FATAL_FAILURE_)



template <typename Pred,
typename T1,
typename T2>
AssertionResult AssertPred2Helper(const char* pred_text,
const char* e1,
const char* e2,
Pred pred,
const T1& v1,
const T2& v2) {
if (pred(v1, v2)) return AssertionSuccess();

return AssertionFailure()
<< pred_text << "(" << e1 << ", " << e2
<< ") evaluates to false, where"
<< "\n"
<< e1 << " evaluates to " << ::testing::PrintToString(v1) << "\n"
<< e2 << " evaluates to " << ::testing::PrintToString(v2);
}

#define GTEST_PRED_FORMAT2_(pred_format, v1, v2, on_failure)\
GTEST_ASSERT_(pred_format(#v1, #v2, v1, v2), \
on_failure)

#define GTEST_PRED2_(pred, v1, v2, on_failure)\
GTEST_ASSERT_(::testing::AssertPred2Helper(#pred, \
#v1, \
#v2, \
pred, \
v1, \
v2), on_failure)

#define EXPECT_PRED_FORMAT2(pred_format, v1, v2) \
GTEST_PRED_FORMAT2_(pred_format, v1, v2, GTEST_NONFATAL_FAILURE_)
#define EXPECT_PRED2(pred, v1, v2) \
GTEST_PRED2_(pred, v1, v2, GTEST_NONFATAL_FAILURE_)
#define ASSERT_PRED_FORMAT2(pred_format, v1, v2) \
GTEST_PRED_FORMAT2_(pred_format, v1, v2, GTEST_FATAL_FAILURE_)
#define ASSERT_PRED2(pred, v1, v2) \
GTEST_PRED2_(pred, v1, v2, GTEST_FATAL_FAILURE_)



template <typename Pred,
typename T1,
typename T2,
typename T3>
AssertionResult AssertPred3Helper(const char* pred_text,
const char* e1,
const char* e2,
const char* e3,
Pred pred,
const T1& v1,
const T2& v2,
const T3& v3) {
if (pred(v1, v2, v3)) return AssertionSuccess();

return AssertionFailure()
<< pred_text << "(" << e1 << ", " << e2 << ", " << e3
<< ") evaluates to false, where"
<< "\n"
<< e1 << " evaluates to " << ::testing::PrintToString(v1) << "\n"
<< e2 << " evaluates to " << ::testing::PrintToString(v2) << "\n"
<< e3 << " evaluates to " << ::testing::PrintToString(v3);
}

#define GTEST_PRED_FORMAT3_(pred_format, v1, v2, v3, on_failure)\
GTEST_ASSERT_(pred_format(#v1, #v2, #v3, v1, v2, v3), \
on_failure)

#define GTEST_PRED3_(pred, v1, v2, v3, on_failure)\
GTEST_ASSERT_(::testing::AssertPred3Helper(#pred, \
#v1, \
#v2, \
#v3, \
pred, \
v1, \
v2, \
v3), on_failure)

#define EXPECT_PRED_FORMAT3(pred_format, v1, v2, v3) \
GTEST_PRED_FORMAT3_(pred_format, v1, v2, v3, GTEST_NONFATAL_FAILURE_)
#define EXPECT_PRED3(pred, v1, v2, v3) \
GTEST_PRED3_(pred, v1, v2, v3, GTEST_NONFATAL_FAILURE_)
#define ASSERT_PRED_FORMAT3(pred_format, v1, v2, v3) \
GTEST_PRED_FORMAT3_(pred_format, v1, v2, v3, GTEST_FATAL_FAILURE_)
#define ASSERT_PRED3(pred, v1, v2, v3) \
GTEST_PRED3_(pred, v1, v2, v3, GTEST_FATAL_FAILURE_)



template <typename Pred,
typename T1,
typename T2,
typename T3,
typename T4>
AssertionResult AssertPred4Helper(const char* pred_text,
const char* e1,
const char* e2,
const char* e3,
const char* e4,
Pred pred,
const T1& v1,
const T2& v2,
const T3& v3,
const T4& v4) {
if (pred(v1, v2, v3, v4)) return AssertionSuccess();

return AssertionFailure()
<< pred_text << "(" << e1 << ", " << e2 << ", " << e3 << ", " << e4
<< ") evaluates to false, where"
<< "\n"
<< e1 << " evaluates to " << ::testing::PrintToString(v1) << "\n"
<< e2 << " evaluates to " << ::testing::PrintToString(v2) << "\n"
<< e3 << " evaluates to " << ::testing::PrintToString(v3) << "\n"
<< e4 << " evaluates to " << ::testing::PrintToString(v4);
}

#define GTEST_PRED_FORMAT4_(pred_format, v1, v2, v3, v4, on_failure)\
GTEST_ASSERT_(pred_format(#v1, #v2, #v3, #v4, v1, v2, v3, v4), \
on_failure)

#define GTEST_PRED4_(pred, v1, v2, v3, v4, on_failure)\
GTEST_ASSERT_(::testing::AssertPred4Helper(#pred, \
#v1, \
#v2, \
#v3, \
#v4, \
pred, \
v1, \
v2, \
v3, \
v4), on_failure)

#define EXPECT_PRED_FORMAT4(pred_format, v1, v2, v3, v4) \
GTEST_PRED_FORMAT4_(pred_format, v1, v2, v3, v4, GTEST_NONFATAL_FAILURE_)
#define EXPECT_PRED4(pred, v1, v2, v3, v4) \
GTEST_PRED4_(pred, v1, v2, v3, v4, GTEST_NONFATAL_FAILURE_)
#define ASSERT_PRED_FORMAT4(pred_format, v1, v2, v3, v4) \
GTEST_PRED_FORMAT4_(pred_format, v1, v2, v3, v4, GTEST_FATAL_FAILURE_)
#define ASSERT_PRED4(pred, v1, v2, v3, v4) \
GTEST_PRED4_(pred, v1, v2, v3, v4, GTEST_FATAL_FAILURE_)



template <typename Pred,
typename T1,
typename T2,
typename T3,
typename T4,
typename T5>
AssertionResult AssertPred5Helper(const char* pred_text,
const char* e1,
const char* e2,
const char* e3,
const char* e4,
const char* e5,
Pred pred,
const T1& v1,
const T2& v2,
const T3& v3,
const T4& v4,
const T5& v5) {
if (pred(v1, v2, v3, v4, v5)) return AssertionSuccess();

return AssertionFailure()
<< pred_text << "(" << e1 << ", " << e2 << ", " << e3 << ", " << e4
<< ", " << e5 << ") evaluates to false, where"
<< "\n"
<< e1 << " evaluates to " << ::testing::PrintToString(v1) << "\n"
<< e2 << " evaluates to " << ::testing::PrintToString(v2) << "\n"
<< e3 << " evaluates to " << ::testing::PrintToString(v3) << "\n"
<< e4 << " evaluates to " << ::testing::PrintToString(v4) << "\n"
<< e5 << " evaluates to " << ::testing::PrintToString(v5);
}

#define GTEST_PRED_FORMAT5_(pred_format, v1, v2, v3, v4, v5, on_failure)\
GTEST_ASSERT_(pred_format(#v1, #v2, #v3, #v4, #v5, v1, v2, v3, v4, v5), \
on_failure)

#define GTEST_PRED5_(pred, v1, v2, v3, v4, v5, on_failure)\
GTEST_ASSERT_(::testing::AssertPred5Helper(#pred, \
#v1, \
#v2, \
#v3, \
#v4, \
#v5, \
pred, \
v1, \
v2, \
v3, \
v4, \
v5), on_failure)

#define EXPECT_PRED_FORMAT5(pred_format, v1, v2, v3, v4, v5) \
GTEST_PRED_FORMAT5_(pred_format, v1, v2, v3, v4, v5, GTEST_NONFATAL_FAILURE_)
#define EXPECT_PRED5(pred, v1, v2, v3, v4, v5) \
GTEST_PRED5_(pred, v1, v2, v3, v4, v5, GTEST_NONFATAL_FAILURE_)
#define ASSERT_PRED_FORMAT5(pred_format, v1, v2, v3, v4, v5) \
GTEST_PRED_FORMAT5_(pred_format, v1, v2, v3, v4, v5, GTEST_FATAL_FAILURE_)
#define ASSERT_PRED5(pred, v1, v2, v3, v4, v5) \
GTEST_PRED5_(pred, v1, v2, v3, v4, v5, GTEST_FATAL_FAILURE_)



}  

#endif  

namespace testing {

class GTEST_API_ Test {
public:
friend class TestInfo;

virtual ~Test();

static void SetUpTestSuite() {}

static void TearDownTestSuite() {}

#ifndef GTEST_REMOVE_LEGACY_TEST_CASEAPI_
static void TearDownTestCase() {}
static void SetUpTestCase() {}
#endif  

static bool HasFatalFailure();

static bool HasNonfatalFailure();

static bool IsSkipped();

static bool HasFailure() { return HasFatalFailure() || HasNonfatalFailure(); }

static void RecordProperty(const std::string& key, const std::string& value);
static void RecordProperty(const std::string& key, int value);

protected:
Test();

virtual void SetUp();

virtual void TearDown();

private:
static bool HasSameFixtureClass();

virtual void TestBody() = 0;

void Run();

void DeleteSelf_() { delete this; }

const std::unique_ptr<GTEST_FLAG_SAVER_> gtest_flag_saver_;

struct Setup_should_be_spelled_SetUp {};
virtual Setup_should_be_spelled_SetUp* Setup() { return nullptr; }

GTEST_DISALLOW_COPY_AND_ASSIGN_(Test);
};

typedef internal::TimeInMillis TimeInMillis;

class TestProperty {
public:
TestProperty(const std::string& a_key, const std::string& a_value) :
key_(a_key), value_(a_value) {
}

const char* key() const {
return key_.c_str();
}

const char* value() const {
return value_.c_str();
}

void SetValue(const std::string& new_value) {
value_ = new_value;
}

private:
std::string key_;
std::string value_;
};

class GTEST_API_ TestResult {
public:
TestResult();

~TestResult();

int total_part_count() const;

int test_property_count() const;

bool Passed() const { return !Skipped() && !Failed(); }

bool Skipped() const;

bool Failed() const;

bool HasFatalFailure() const;

bool HasNonfatalFailure() const;

TimeInMillis elapsed_time() const { return elapsed_time_; }

TimeInMillis start_timestamp() const { return start_timestamp_; }

const TestPartResult& GetTestPartResult(int i) const;

const TestProperty& GetTestProperty(int i) const;

private:
friend class TestInfo;
friend class TestSuite;
friend class UnitTest;
friend class internal::DefaultGlobalTestPartResultReporter;
friend class internal::ExecDeathTest;
friend class internal::TestResultAccessor;
friend class internal::UnitTestImpl;
friend class internal::WindowsDeathTest;
friend class internal::FuchsiaDeathTest;

const std::vector<TestPartResult>& test_part_results() const {
return test_part_results_;
}

const std::vector<TestProperty>& test_properties() const {
return test_properties_;
}

void set_start_timestamp(TimeInMillis start) { start_timestamp_ = start; }

void set_elapsed_time(TimeInMillis elapsed) { elapsed_time_ = elapsed; }

void RecordProperty(const std::string& xml_element,
const TestProperty& test_property);

static bool ValidateTestProperty(const std::string& xml_element,
const TestProperty& test_property);

void AddTestPartResult(const TestPartResult& test_part_result);

int death_test_count() const { return death_test_count_; }

int increment_death_test_count() { return ++death_test_count_; }

void ClearTestPartResults();

void Clear();

internal::Mutex test_properites_mutex_;

std::vector<TestPartResult> test_part_results_;
std::vector<TestProperty> test_properties_;
int death_test_count_;
TimeInMillis start_timestamp_;
TimeInMillis elapsed_time_;

GTEST_DISALLOW_COPY_AND_ASSIGN_(TestResult);
};  

class GTEST_API_ TestInfo {
public:
~TestInfo();

const char* test_suite_name() const { return test_suite_name_.c_str(); }

#ifndef GTEST_REMOVE_LEGACY_TEST_CASEAPI_
const char* test_case_name() const { return test_suite_name(); }
#endif  

const char* name() const { return name_.c_str(); }

const char* type_param() const {
if (type_param_.get() != nullptr) return type_param_->c_str();
return nullptr;
}

const char* value_param() const {
if (value_param_.get() != nullptr) return value_param_->c_str();
return nullptr;
}

const char* file() const { return location_.file.c_str(); }

int line() const { return location_.line; }

bool is_in_another_shard() const { return is_in_another_shard_; }

bool should_run() const { return should_run_; }

bool is_reportable() const {
return matches_filter_ && !is_in_another_shard_;
}

const TestResult* result() const { return &result_; }

private:
#if GTEST_HAS_DEATH_TEST
friend class internal::DefaultDeathTestFactory;
#endif  
friend class Test;
friend class TestSuite;
friend class internal::UnitTestImpl;
friend class internal::StreamingListenerTest;
friend TestInfo* internal::MakeAndRegisterTestInfo(
const char* test_suite_name, const char* name, const char* type_param,
const char* value_param, internal::CodeLocation code_location,
internal::TypeId fixture_class_id, internal::SetUpTestSuiteFunc set_up_tc,
internal::TearDownTestSuiteFunc tear_down_tc,
internal::TestFactoryBase* factory);

TestInfo(const std::string& test_suite_name, const std::string& name,
const char* a_type_param,   
const char* a_value_param,  
internal::CodeLocation a_code_location,
internal::TypeId fixture_class_id,
internal::TestFactoryBase* factory);

int increment_death_test_count() {
return result_.increment_death_test_count();
}

void Run();

void Skip();

static void ClearTestResult(TestInfo* test_info) {
test_info->result_.Clear();
}

const std::string test_suite_name_;    
const std::string name_;               
const std::unique_ptr<const ::std::string> type_param_;
const std::unique_ptr<const ::std::string> value_param_;
internal::CodeLocation location_;
const internal::TypeId fixture_class_id_;  
bool should_run_;           
bool is_disabled_;          
bool matches_filter_;       
bool is_in_another_shard_;  
internal::TestFactoryBase* const factory_;  

TestResult result_;

GTEST_DISALLOW_COPY_AND_ASSIGN_(TestInfo);
};

class GTEST_API_ TestSuite {
public:
TestSuite(const char* name, const char* a_type_param,
internal::SetUpTestSuiteFunc set_up_tc,
internal::TearDownTestSuiteFunc tear_down_tc);

virtual ~TestSuite();

const char* name() const { return name_.c_str(); }

const char* type_param() const {
if (type_param_.get() != nullptr) return type_param_->c_str();
return nullptr;
}

bool should_run() const { return should_run_; }

int successful_test_count() const;

int skipped_test_count() const;

int failed_test_count() const;

int reportable_disabled_test_count() const;

int disabled_test_count() const;

int reportable_test_count() const;

int test_to_run_count() const;

int total_test_count() const;

bool Passed() const { return !Failed(); }

bool Failed() const {
return failed_test_count() > 0 || ad_hoc_test_result().Failed();
}

TimeInMillis elapsed_time() const { return elapsed_time_; }

TimeInMillis start_timestamp() const { return start_timestamp_; }

const TestInfo* GetTestInfo(int i) const;

const TestResult& ad_hoc_test_result() const { return ad_hoc_test_result_; }

private:
friend class Test;
friend class internal::UnitTestImpl;

std::vector<TestInfo*>& test_info_list() { return test_info_list_; }

const std::vector<TestInfo*>& test_info_list() const {
return test_info_list_;
}

TestInfo* GetMutableTestInfo(int i);

void set_should_run(bool should) { should_run_ = should; }

void AddTestInfo(TestInfo * test_info);

void ClearResult();

static void ClearTestSuiteResult(TestSuite* test_suite) {
test_suite->ClearResult();
}

void Run();

void Skip();

void RunSetUpTestSuite() {
if (set_up_tc_ != nullptr) {
(*set_up_tc_)();
}
}

void RunTearDownTestSuite() {
if (tear_down_tc_ != nullptr) {
(*tear_down_tc_)();
}
}

static bool TestPassed(const TestInfo* test_info) {
return test_info->should_run() && test_info->result()->Passed();
}

static bool TestSkipped(const TestInfo* test_info) {
return test_info->should_run() && test_info->result()->Skipped();
}

static bool TestFailed(const TestInfo* test_info) {
return test_info->should_run() && test_info->result()->Failed();
}

static bool TestReportableDisabled(const TestInfo* test_info) {
return test_info->is_reportable() && test_info->is_disabled_;
}

static bool TestDisabled(const TestInfo* test_info) {
return test_info->is_disabled_;
}

static bool TestReportable(const TestInfo* test_info) {
return test_info->is_reportable();
}

static bool ShouldRunTest(const TestInfo* test_info) {
return test_info->should_run();
}

void ShuffleTests(internal::Random* random);

void UnshuffleTests();

std::string name_;
const std::unique_ptr<const ::std::string> type_param_;
std::vector<TestInfo*> test_info_list_;
std::vector<int> test_indices_;
internal::SetUpTestSuiteFunc set_up_tc_;
internal::TearDownTestSuiteFunc tear_down_tc_;
bool should_run_;
TimeInMillis start_timestamp_;
TimeInMillis elapsed_time_;
TestResult ad_hoc_test_result_;

GTEST_DISALLOW_COPY_AND_ASSIGN_(TestSuite);
};

class Environment {
public:
virtual ~Environment() {}

virtual void SetUp() {}

virtual void TearDown() {}
private:
struct Setup_should_be_spelled_SetUp {};
virtual Setup_should_be_spelled_SetUp* Setup() { return nullptr; }
};

#if GTEST_HAS_EXCEPTIONS

class GTEST_API_ AssertionException
: public internal::GoogleTestFailureException {
public:
explicit AssertionException(const TestPartResult& result)
: GoogleTestFailureException(result) {}
};

#endif  

class TestEventListener {
public:
virtual ~TestEventListener() {}

virtual void OnTestProgramStart(const UnitTest& unit_test) = 0;

virtual void OnTestIterationStart(const UnitTest& unit_test,
int iteration) = 0;

virtual void OnEnvironmentsSetUpStart(const UnitTest& unit_test) = 0;

virtual void OnEnvironmentsSetUpEnd(const UnitTest& unit_test) = 0;

virtual void OnTestSuiteStart(const TestSuite& ) {}

#ifndef GTEST_REMOVE_LEGACY_TEST_CASEAPI_
virtual void OnTestCaseStart(const TestCase& ) {}
#endif  

virtual void OnTestStart(const TestInfo& test_info) = 0;

virtual void OnTestPartResult(const TestPartResult& test_part_result) = 0;

virtual void OnTestEnd(const TestInfo& test_info) = 0;

virtual void OnTestSuiteEnd(const TestSuite& ) {}

#ifndef GTEST_REMOVE_LEGACY_TEST_CASEAPI_
virtual void OnTestCaseEnd(const TestCase& ) {}
#endif  

virtual void OnEnvironmentsTearDownStart(const UnitTest& unit_test) = 0;

virtual void OnEnvironmentsTearDownEnd(const UnitTest& unit_test) = 0;

virtual void OnTestIterationEnd(const UnitTest& unit_test,
int iteration) = 0;

virtual void OnTestProgramEnd(const UnitTest& unit_test) = 0;
};

class EmptyTestEventListener : public TestEventListener {
public:
void OnTestProgramStart(const UnitTest& ) override {}
void OnTestIterationStart(const UnitTest& ,
int ) override {}
void OnEnvironmentsSetUpStart(const UnitTest& ) override {}
void OnEnvironmentsSetUpEnd(const UnitTest& ) override {}
void OnTestSuiteStart(const TestSuite& ) override {}
#ifndef GTEST_REMOVE_LEGACY_TEST_CASEAPI_
void OnTestCaseStart(const TestCase& ) override {}
#endif  

void OnTestStart(const TestInfo& ) override {}
void OnTestPartResult(const TestPartResult& ) override {}
void OnTestEnd(const TestInfo& ) override {}
void OnTestSuiteEnd(const TestSuite& ) override {}
#ifndef GTEST_REMOVE_LEGACY_TEST_CASEAPI_
void OnTestCaseEnd(const TestCase& ) override {}
#endif  

void OnEnvironmentsTearDownStart(const UnitTest& ) override {}
void OnEnvironmentsTearDownEnd(const UnitTest& ) override {}
void OnTestIterationEnd(const UnitTest& ,
int ) override {}
void OnTestProgramEnd(const UnitTest& ) override {}
};

class GTEST_API_ TestEventListeners {
public:
TestEventListeners();
~TestEventListeners();

void Append(TestEventListener* listener);

TestEventListener* Release(TestEventListener* listener);

TestEventListener* default_result_printer() const {
return default_result_printer_;
}

TestEventListener* default_xml_generator() const {
return default_xml_generator_;
}

private:
friend class TestSuite;
friend class TestInfo;
friend class internal::DefaultGlobalTestPartResultReporter;
friend class internal::NoExecDeathTest;
friend class internal::TestEventListenersAccessor;
friend class internal::UnitTestImpl;

TestEventListener* repeater();

void SetDefaultResultPrinter(TestEventListener* listener);

void SetDefaultXmlGenerator(TestEventListener* listener);

bool EventForwardingEnabled() const;
void SuppressEventForwarding();

internal::TestEventRepeater* repeater_;
TestEventListener* default_result_printer_;
TestEventListener* default_xml_generator_;

GTEST_DISALLOW_COPY_AND_ASSIGN_(TestEventListeners);
};

class GTEST_API_ UnitTest {
public:
static UnitTest* GetInstance();

int Run() GTEST_MUST_USE_RESULT_;

const char* original_working_dir() const;

const TestSuite* current_test_suite() const GTEST_LOCK_EXCLUDED_(mutex_);

#ifndef GTEST_REMOVE_LEGACY_TEST_CASEAPI_
const TestCase* current_test_case() const GTEST_LOCK_EXCLUDED_(mutex_);
#endif

const TestInfo* current_test_info() const
GTEST_LOCK_EXCLUDED_(mutex_);

int random_seed() const;

internal::ParameterizedTestSuiteRegistry& parameterized_test_registry()
GTEST_LOCK_EXCLUDED_(mutex_);

int successful_test_suite_count() const;

int failed_test_suite_count() const;

int total_test_suite_count() const;

int test_suite_to_run_count() const;

#ifndef GTEST_REMOVE_LEGACY_TEST_CASEAPI_
int successful_test_case_count() const;
int failed_test_case_count() const;
int total_test_case_count() const;
int test_case_to_run_count() const;
#endif  

int successful_test_count() const;

int skipped_test_count() const;

int failed_test_count() const;

int reportable_disabled_test_count() const;

int disabled_test_count() const;

int reportable_test_count() const;

int total_test_count() const;

int test_to_run_count() const;

TimeInMillis start_timestamp() const;

TimeInMillis elapsed_time() const;

bool Passed() const;

bool Failed() const;

const TestSuite* GetTestSuite(int i) const;

#ifndef GTEST_REMOVE_LEGACY_TEST_CASEAPI_
const TestCase* GetTestCase(int i) const;
#endif  

const TestResult& ad_hoc_test_result() const;

TestEventListeners& listeners();

private:
Environment* AddEnvironment(Environment* env);

void AddTestPartResult(TestPartResult::Type result_type,
const char* file_name,
int line_number,
const std::string& message,
const std::string& os_stack_trace)
GTEST_LOCK_EXCLUDED_(mutex_);

void RecordProperty(const std::string& key, const std::string& value);

TestSuite* GetMutableTestSuite(int i);

internal::UnitTestImpl* impl() { return impl_; }
const internal::UnitTestImpl* impl() const { return impl_; }

friend class ScopedTrace;
friend class Test;
friend class internal::AssertHelper;
friend class internal::StreamingListenerTest;
friend class internal::UnitTestRecordPropertyTestHelper;
friend Environment* AddGlobalTestEnvironment(Environment* env);
friend std::set<std::string>* internal::GetIgnoredParameterizedTestSuites();
friend internal::UnitTestImpl* internal::GetUnitTestImpl();
friend void internal::ReportFailureInUnknownLocation(
TestPartResult::Type result_type,
const std::string& message);

UnitTest();

virtual ~UnitTest();

void PushGTestTrace(const internal::TraceInfo& trace)
GTEST_LOCK_EXCLUDED_(mutex_);

void PopGTestTrace()
GTEST_LOCK_EXCLUDED_(mutex_);

mutable internal::Mutex mutex_;

internal::UnitTestImpl* impl_;

GTEST_DISALLOW_COPY_AND_ASSIGN_(UnitTest);
};

inline Environment* AddGlobalTestEnvironment(Environment* env) {
return UnitTest::GetInstance()->AddEnvironment(env);
}

GTEST_API_ void InitGoogleTest(int* argc, char** argv);

GTEST_API_ void InitGoogleTest(int* argc, wchar_t** argv);

GTEST_API_ void InitGoogleTest();

namespace internal {

template <typename T1, typename T2>
AssertionResult CmpHelperEQFailure(const char* lhs_expression,
const char* rhs_expression,
const T1& lhs, const T2& rhs) {
return EqFailure(lhs_expression,
rhs_expression,
FormatForComparisonFailureMessage(lhs, rhs),
FormatForComparisonFailureMessage(rhs, lhs),
false);
}

struct faketype {};
inline bool operator==(faketype, faketype) { return true; }
inline bool operator!=(faketype, faketype) { return false; }

template <typename T1, typename T2>
AssertionResult CmpHelperEQ(const char* lhs_expression,
const char* rhs_expression,
const T1& lhs,
const T2& rhs) {
if (lhs == rhs) {
return AssertionSuccess();
}

return CmpHelperEQFailure(lhs_expression, rhs_expression, lhs, rhs);
}

GTEST_API_ AssertionResult CmpHelperEQ(const char* lhs_expression,
const char* rhs_expression,
BiggestInt lhs,
BiggestInt rhs);

class EqHelper {
public:
template <
typename T1, typename T2,
typename std::enable_if<!std::is_integral<T1>::value ||
!std::is_pointer<T2>::value>::type* = nullptr>
static AssertionResult Compare(const char* lhs_expression,
const char* rhs_expression, const T1& lhs,
const T2& rhs) {
return CmpHelperEQ(lhs_expression, rhs_expression, lhs, rhs);
}

static AssertionResult Compare(const char* lhs_expression,
const char* rhs_expression,
BiggestInt lhs,
BiggestInt rhs) {
return CmpHelperEQ(lhs_expression, rhs_expression, lhs, rhs);
}

template <typename T>
static AssertionResult Compare(
const char* lhs_expression, const char* rhs_expression,
std::nullptr_t , T* rhs) {
return CmpHelperEQ(lhs_expression, rhs_expression, static_cast<T*>(nullptr),
rhs);
}
};

template <typename T1, typename T2>
AssertionResult CmpHelperOpFailure(const char* expr1, const char* expr2,
const T1& val1, const T2& val2,
const char* op) {
return AssertionFailure()
<< "Expected: (" << expr1 << ") " << op << " (" << expr2
<< "), actual: " << FormatForComparisonFailureMessage(val1, val2)
<< " vs " << FormatForComparisonFailureMessage(val2, val1);
}


#define GTEST_IMPL_CMP_HELPER_(op_name, op)\
template <typename T1, typename T2>\
AssertionResult CmpHelper##op_name(const char* expr1, const char* expr2, \
const T1& val1, const T2& val2) {\
if (val1 op val2) {\
return AssertionSuccess();\
} else {\
return CmpHelperOpFailure(expr1, expr2, val1, val2, #op);\
}\
}\
GTEST_API_ AssertionResult CmpHelper##op_name(\
const char* expr1, const char* expr2, BiggestInt val1, BiggestInt val2)


GTEST_IMPL_CMP_HELPER_(NE, !=);
GTEST_IMPL_CMP_HELPER_(LE, <=);
GTEST_IMPL_CMP_HELPER_(LT, <);
GTEST_IMPL_CMP_HELPER_(GE, >=);
GTEST_IMPL_CMP_HELPER_(GT, >);

#undef GTEST_IMPL_CMP_HELPER_

GTEST_API_ AssertionResult CmpHelperSTREQ(const char* s1_expression,
const char* s2_expression,
const char* s1,
const char* s2);

GTEST_API_ AssertionResult CmpHelperSTRCASEEQ(const char* s1_expression,
const char* s2_expression,
const char* s1,
const char* s2);

GTEST_API_ AssertionResult CmpHelperSTRNE(const char* s1_expression,
const char* s2_expression,
const char* s1,
const char* s2);

GTEST_API_ AssertionResult CmpHelperSTRCASENE(const char* s1_expression,
const char* s2_expression,
const char* s1,
const char* s2);


GTEST_API_ AssertionResult CmpHelperSTREQ(const char* s1_expression,
const char* s2_expression,
const wchar_t* s1,
const wchar_t* s2);

GTEST_API_ AssertionResult CmpHelperSTRNE(const char* s1_expression,
const char* s2_expression,
const wchar_t* s1,
const wchar_t* s2);

}  

GTEST_API_ AssertionResult IsSubstring(
const char* needle_expr, const char* haystack_expr,
const char* needle, const char* haystack);
GTEST_API_ AssertionResult IsSubstring(
const char* needle_expr, const char* haystack_expr,
const wchar_t* needle, const wchar_t* haystack);
GTEST_API_ AssertionResult IsNotSubstring(
const char* needle_expr, const char* haystack_expr,
const char* needle, const char* haystack);
GTEST_API_ AssertionResult IsNotSubstring(
const char* needle_expr, const char* haystack_expr,
const wchar_t* needle, const wchar_t* haystack);
GTEST_API_ AssertionResult IsSubstring(
const char* needle_expr, const char* haystack_expr,
const ::std::string& needle, const ::std::string& haystack);
GTEST_API_ AssertionResult IsNotSubstring(
const char* needle_expr, const char* haystack_expr,
const ::std::string& needle, const ::std::string& haystack);

#if GTEST_HAS_STD_WSTRING
GTEST_API_ AssertionResult IsSubstring(
const char* needle_expr, const char* haystack_expr,
const ::std::wstring& needle, const ::std::wstring& haystack);
GTEST_API_ AssertionResult IsNotSubstring(
const char* needle_expr, const char* haystack_expr,
const ::std::wstring& needle, const ::std::wstring& haystack);
#endif  

namespace internal {

template <typename RawType>
AssertionResult CmpHelperFloatingPointEQ(const char* lhs_expression,
const char* rhs_expression,
RawType lhs_value,
RawType rhs_value) {
const FloatingPoint<RawType> lhs(lhs_value), rhs(rhs_value);

if (lhs.AlmostEquals(rhs)) {
return AssertionSuccess();
}

::std::stringstream lhs_ss;
lhs_ss << std::setprecision(std::numeric_limits<RawType>::digits10 + 2)
<< lhs_value;

::std::stringstream rhs_ss;
rhs_ss << std::setprecision(std::numeric_limits<RawType>::digits10 + 2)
<< rhs_value;

return EqFailure(lhs_expression,
rhs_expression,
StringStreamToString(&lhs_ss),
StringStreamToString(&rhs_ss),
false);
}

GTEST_API_ AssertionResult DoubleNearPredFormat(const char* expr1,
const char* expr2,
const char* abs_error_expr,
double val1,
double val2,
double abs_error);

class GTEST_API_ AssertHelper {
public:
AssertHelper(TestPartResult::Type type,
const char* file,
int line,
const char* message);
~AssertHelper();

void operator=(const Message& message) const;

private:
struct AssertHelperData {
AssertHelperData(TestPartResult::Type t,
const char* srcfile,
int line_num,
const char* msg)
: type(t), file(srcfile), line(line_num), message(msg) { }

TestPartResult::Type const type;
const char* const file;
int const line;
std::string const message;

private:
GTEST_DISALLOW_COPY_AND_ASSIGN_(AssertHelperData);
};

AssertHelperData* const data_;

GTEST_DISALLOW_COPY_AND_ASSIGN_(AssertHelper);
};

}  


template <typename T>
class WithParamInterface {
public:
typedef T ParamType;
virtual ~WithParamInterface() {}

static const ParamType& GetParam() {
GTEST_CHECK_(parameter_ != nullptr)
<< "GetParam() can only be called inside a value-parameterized test "
<< "-- did you intend to write TEST_P instead of TEST_F?";
return *parameter_;
}

private:
static void SetParam(const ParamType* parameter) {
parameter_ = parameter;
}

static const ParamType* parameter_;

template <class TestClass> friend class internal::ParameterizedTestFactory;
};

template <typename T>
const T* WithParamInterface<T>::parameter_ = nullptr;


template <typename T>
class TestWithParam : public Test, public WithParamInterface<T> {
};


#define GTEST_SKIP() GTEST_SKIP_("")


#define ADD_FAILURE() GTEST_NONFATAL_FAILURE_("Failed")

#define ADD_FAILURE_AT(file, line) \
GTEST_MESSAGE_AT_(file, line, "Failed", \
::testing::TestPartResult::kNonFatalFailure)

#define GTEST_FAIL() GTEST_FATAL_FAILURE_("Failed")

#define GTEST_FAIL_AT(file, line)         \
GTEST_MESSAGE_AT_(file, line, "Failed", \
::testing::TestPartResult::kFatalFailure)

#if !GTEST_DONT_DEFINE_FAIL
# define FAIL() GTEST_FAIL()
#endif

#define GTEST_SUCCEED() GTEST_SUCCESS_("Succeeded")

#if !GTEST_DONT_DEFINE_SUCCEED
# define SUCCEED() GTEST_SUCCEED()
#endif


#define EXPECT_THROW(statement, expected_exception) \
GTEST_TEST_THROW_(statement, expected_exception, GTEST_NONFATAL_FAILURE_)
#define EXPECT_NO_THROW(statement) \
GTEST_TEST_NO_THROW_(statement, GTEST_NONFATAL_FAILURE_)
#define EXPECT_ANY_THROW(statement) \
GTEST_TEST_ANY_THROW_(statement, GTEST_NONFATAL_FAILURE_)
#define ASSERT_THROW(statement, expected_exception) \
GTEST_TEST_THROW_(statement, expected_exception, GTEST_FATAL_FAILURE_)
#define ASSERT_NO_THROW(statement) \
GTEST_TEST_NO_THROW_(statement, GTEST_FATAL_FAILURE_)
#define ASSERT_ANY_THROW(statement) \
GTEST_TEST_ANY_THROW_(statement, GTEST_FATAL_FAILURE_)

#define EXPECT_TRUE(condition) \
GTEST_TEST_BOOLEAN_(condition, #condition, false, true, \
GTEST_NONFATAL_FAILURE_)
#define EXPECT_FALSE(condition) \
GTEST_TEST_BOOLEAN_(!(condition), #condition, true, false, \
GTEST_NONFATAL_FAILURE_)
#define ASSERT_TRUE(condition) \
GTEST_TEST_BOOLEAN_(condition, #condition, false, true, \
GTEST_FATAL_FAILURE_)
#define ASSERT_FALSE(condition) \
GTEST_TEST_BOOLEAN_(!(condition), #condition, true, false, \
GTEST_FATAL_FAILURE_)


#define EXPECT_EQ(val1, val2) \
EXPECT_PRED_FORMAT2(::testing::internal::EqHelper::Compare, val1, val2)
#define EXPECT_NE(val1, val2) \
EXPECT_PRED_FORMAT2(::testing::internal::CmpHelperNE, val1, val2)
#define EXPECT_LE(val1, val2) \
EXPECT_PRED_FORMAT2(::testing::internal::CmpHelperLE, val1, val2)
#define EXPECT_LT(val1, val2) \
EXPECT_PRED_FORMAT2(::testing::internal::CmpHelperLT, val1, val2)
#define EXPECT_GE(val1, val2) \
EXPECT_PRED_FORMAT2(::testing::internal::CmpHelperGE, val1, val2)
#define EXPECT_GT(val1, val2) \
EXPECT_PRED_FORMAT2(::testing::internal::CmpHelperGT, val1, val2)

#define GTEST_ASSERT_EQ(val1, val2) \
ASSERT_PRED_FORMAT2(::testing::internal::EqHelper::Compare, val1, val2)
#define GTEST_ASSERT_NE(val1, val2) \
ASSERT_PRED_FORMAT2(::testing::internal::CmpHelperNE, val1, val2)
#define GTEST_ASSERT_LE(val1, val2) \
ASSERT_PRED_FORMAT2(::testing::internal::CmpHelperLE, val1, val2)
#define GTEST_ASSERT_LT(val1, val2) \
ASSERT_PRED_FORMAT2(::testing::internal::CmpHelperLT, val1, val2)
#define GTEST_ASSERT_GE(val1, val2) \
ASSERT_PRED_FORMAT2(::testing::internal::CmpHelperGE, val1, val2)
#define GTEST_ASSERT_GT(val1, val2) \
ASSERT_PRED_FORMAT2(::testing::internal::CmpHelperGT, val1, val2)


#if !GTEST_DONT_DEFINE_ASSERT_EQ
# define ASSERT_EQ(val1, val2) GTEST_ASSERT_EQ(val1, val2)
#endif

#if !GTEST_DONT_DEFINE_ASSERT_NE
# define ASSERT_NE(val1, val2) GTEST_ASSERT_NE(val1, val2)
#endif

#if !GTEST_DONT_DEFINE_ASSERT_LE
# define ASSERT_LE(val1, val2) GTEST_ASSERT_LE(val1, val2)
#endif

#if !GTEST_DONT_DEFINE_ASSERT_LT
# define ASSERT_LT(val1, val2) GTEST_ASSERT_LT(val1, val2)
#endif

#if !GTEST_DONT_DEFINE_ASSERT_GE
# define ASSERT_GE(val1, val2) GTEST_ASSERT_GE(val1, val2)
#endif

#if !GTEST_DONT_DEFINE_ASSERT_GT
# define ASSERT_GT(val1, val2) GTEST_ASSERT_GT(val1, val2)
#endif


#define EXPECT_STREQ(s1, s2) \
EXPECT_PRED_FORMAT2(::testing::internal::CmpHelperSTREQ, s1, s2)
#define EXPECT_STRNE(s1, s2) \
EXPECT_PRED_FORMAT2(::testing::internal::CmpHelperSTRNE, s1, s2)
#define EXPECT_STRCASEEQ(s1, s2) \
EXPECT_PRED_FORMAT2(::testing::internal::CmpHelperSTRCASEEQ, s1, s2)
#define EXPECT_STRCASENE(s1, s2)\
EXPECT_PRED_FORMAT2(::testing::internal::CmpHelperSTRCASENE, s1, s2)

#define ASSERT_STREQ(s1, s2) \
ASSERT_PRED_FORMAT2(::testing::internal::CmpHelperSTREQ, s1, s2)
#define ASSERT_STRNE(s1, s2) \
ASSERT_PRED_FORMAT2(::testing::internal::CmpHelperSTRNE, s1, s2)
#define ASSERT_STRCASEEQ(s1, s2) \
ASSERT_PRED_FORMAT2(::testing::internal::CmpHelperSTRCASEEQ, s1, s2)
#define ASSERT_STRCASENE(s1, s2)\
ASSERT_PRED_FORMAT2(::testing::internal::CmpHelperSTRCASENE, s1, s2)


#define EXPECT_FLOAT_EQ(val1, val2)\
EXPECT_PRED_FORMAT2(::testing::internal::CmpHelperFloatingPointEQ<float>, \
val1, val2)

#define EXPECT_DOUBLE_EQ(val1, val2)\
EXPECT_PRED_FORMAT2(::testing::internal::CmpHelperFloatingPointEQ<double>, \
val1, val2)

#define ASSERT_FLOAT_EQ(val1, val2)\
ASSERT_PRED_FORMAT2(::testing::internal::CmpHelperFloatingPointEQ<float>, \
val1, val2)

#define ASSERT_DOUBLE_EQ(val1, val2)\
ASSERT_PRED_FORMAT2(::testing::internal::CmpHelperFloatingPointEQ<double>, \
val1, val2)

#define EXPECT_NEAR(val1, val2, abs_error)\
EXPECT_PRED_FORMAT3(::testing::internal::DoubleNearPredFormat, \
val1, val2, abs_error)

#define ASSERT_NEAR(val1, val2, abs_error)\
ASSERT_PRED_FORMAT3(::testing::internal::DoubleNearPredFormat, \
val1, val2, abs_error)


GTEST_API_ AssertionResult FloatLE(const char* expr1, const char* expr2,
float val1, float val2);
GTEST_API_ AssertionResult DoubleLE(const char* expr1, const char* expr2,
double val1, double val2);


#if GTEST_OS_WINDOWS

# define EXPECT_HRESULT_SUCCEEDED(expr) \
EXPECT_PRED_FORMAT1(::testing::internal::IsHRESULTSuccess, (expr))

# define ASSERT_HRESULT_SUCCEEDED(expr) \
ASSERT_PRED_FORMAT1(::testing::internal::IsHRESULTSuccess, (expr))

# define EXPECT_HRESULT_FAILED(expr) \
EXPECT_PRED_FORMAT1(::testing::internal::IsHRESULTFailure, (expr))

# define ASSERT_HRESULT_FAILED(expr) \
ASSERT_PRED_FORMAT1(::testing::internal::IsHRESULTFailure, (expr))

#endif  

#define ASSERT_NO_FATAL_FAILURE(statement) \
GTEST_TEST_NO_FATAL_FAILURE_(statement, GTEST_FATAL_FAILURE_)
#define EXPECT_NO_FATAL_FAILURE(statement) \
GTEST_TEST_NO_FATAL_FAILURE_(statement, GTEST_NONFATAL_FAILURE_)

class GTEST_API_ ScopedTrace {
public:

template <typename T>
ScopedTrace(const char* file, int line, const T& message) {
PushTrace(file, line, (Message() << message).GetString());
}

ScopedTrace(const char* file, int line, const char* message) {
PushTrace(file, line, message ? message : "(null)");
}

ScopedTrace(const char* file, int line, const std::string& message) {
PushTrace(file, line, message);
}

~ScopedTrace();

private:
void PushTrace(const char* file, int line, std::string message);

GTEST_DISALLOW_COPY_AND_ASSIGN_(ScopedTrace);
} GTEST_ATTRIBUTE_UNUSED_;  

#define SCOPED_TRACE(message) \
::testing::ScopedTrace GTEST_CONCAT_TOKEN_(gtest_trace_, __LINE__)(\
__FILE__, __LINE__, (message))

template <typename T1, typename T2>
constexpr bool StaticAssertTypeEq() noexcept {
static_assert(std::is_same<T1, T2>::value, "T1 and T2 are not the same type");
return true;
}


#define GTEST_TEST(test_suite_name, test_name)             \
GTEST_TEST_(test_suite_name, test_name, ::testing::Test, \
::testing::internal::GetTestTypeId())

#if !GTEST_DONT_DEFINE_TEST
#define TEST(test_suite_name, test_name) GTEST_TEST(test_suite_name, test_name)
#endif

#if !GTEST_DONT_DEFINE_TEST
#define TEST_F(test_fixture, test_name)\
GTEST_TEST_(test_fixture, test_name, test_fixture, \
::testing::internal::GetTypeId<test_fixture>())
#endif  

GTEST_API_ std::string TempDir();

#ifdef _MSC_VER
#  pragma warning(pop)
#endif

template <int&... ExplicitParameterBarrier, typename Factory>
TestInfo* RegisterTest(const char* test_suite_name, const char* test_name,
const char* type_param, const char* value_param,
const char* file, int line, Factory factory) {
using TestT = typename std::remove_pointer<decltype(factory())>::type;

class FactoryImpl : public internal::TestFactoryBase {
public:
explicit FactoryImpl(Factory f) : factory_(std::move(f)) {}
Test* CreateTest() override { return factory_(); }

private:
Factory factory_;
};

return internal::MakeAndRegisterTestInfo(
test_suite_name, test_name, type_param, value_param,
internal::CodeLocation(file, line), internal::GetTypeId<TestT>(),
internal::SuiteApiResolver<TestT>::GetSetUpCaseOrSuite(file, line),
internal::SuiteApiResolver<TestT>::GetTearDownCaseOrSuite(file, line),
new FactoryImpl{std::move(factory)});
}

}  

int RUN_ALL_TESTS() GTEST_MUST_USE_RESULT_;

inline int RUN_ALL_TESTS() {
return ::testing::UnitTest::GetInstance()->Run();
}

GTEST_DISABLE_MSC_WARNINGS_POP_()  

#endif  
