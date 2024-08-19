
#ifndef GTEST_INCLUDE_GTEST_INTERNAL_GTEST_PORT_H_
#define GTEST_INCLUDE_GTEST_INTERNAL_GTEST_PORT_H_



#include <ctype.h>   
#include <stddef.h>  
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#ifndef _WIN32_WCE
# include <sys/types.h>
# include <sys/stat.h>
#endif  

#if defined __APPLE__
# include <AvailabilityMacros.h>
# include <TargetConditionals.h>
#endif

#include <iostream>  
#include <sstream>  
#include <string>  

#define GTEST_DEV_EMAIL_ "googletestframework@@googlegroups.com"
#define GTEST_FLAG_PREFIX_ "gtest_"
#define GTEST_FLAG_PREFIX_DASH_ "gtest-"
#define GTEST_FLAG_PREFIX_UPPER_ "GTEST_"
#define GTEST_NAME_ "Google Test"
#define GTEST_PROJECT_URL_ "http:

#ifdef __GNUC__
# define GTEST_GCC_VER_ \
(__GNUC__*10000 + __GNUC_MINOR__*100 + __GNUC_PATCHLEVEL__)
#endif  

#ifdef __CYGWIN__
# define GTEST_OS_CYGWIN 1
#elif defined __SYMBIAN32__
# define GTEST_OS_SYMBIAN 1
#elif defined _WIN32
# define GTEST_OS_WINDOWS 1
# ifdef _WIN32_WCE
#  define GTEST_OS_WINDOWS_MOBILE 1
# elif defined(__MINGW__) || defined(__MINGW32__)
#  define GTEST_OS_WINDOWS_MINGW 1
# else
#  define GTEST_OS_WINDOWS_DESKTOP 1
# endif  
#elif defined __APPLE__
# define GTEST_OS_MAC 1
# if TARGET_OS_IPHONE
#  define GTEST_OS_IOS 1
#  if TARGET_IPHONE_SIMULATOR
#   define GTEST_OS_IOS_SIMULATOR 1
#  endif
# endif
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
#elif defined __OpenBSD__
# define GTEST_OS_OPENBSD 1
#elif defined __QNX__
# define GTEST_OS_QNX 1
#endif  

#ifndef GTEST_LANG_CXX11
# if __GXX_EXPERIMENTAL_CXX0X__ || __cplusplus >= 201103L
#  define GTEST_LANG_CXX11 1
# else
#  define GTEST_LANG_CXX11 0
# endif
#endif

#if !GTEST_OS_WINDOWS
# include <unistd.h>
# include <strings.h>
#elif !GTEST_OS_WINDOWS_MOBILE
# include <direct.h>
# include <io.h>
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

#if GTEST_HAS_POSIX_RE

# include <regex.h>  

# define GTEST_USES_POSIX_RE 1

#elif GTEST_OS_WINDOWS

# define GTEST_USES_SIMPLE_RE 1

#else

# define GTEST_USES_SIMPLE_RE 1

#endif  

#ifndef GTEST_HAS_EXCEPTIONS
# if defined(_MSC_VER) || defined(__BORLANDC__)
#  ifndef _HAS_EXCEPTIONS
#   define _HAS_EXCEPTIONS 1
#  endif  
#  define GTEST_HAS_EXCEPTIONS _HAS_EXCEPTIONS
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

#if !defined(GTEST_HAS_STD_STRING)
# define GTEST_HAS_STD_STRING 1
#elif !GTEST_HAS_STD_STRING
# error "Google Test cannot be used where ::std::string isn't available."
#endif  

#ifndef GTEST_HAS_GLOBAL_STRING

# define GTEST_HAS_GLOBAL_STRING 0

#endif  

#ifndef GTEST_HAS_STD_WSTRING

# define GTEST_HAS_STD_WSTRING \
(!(GTEST_OS_LINUX_ANDROID || GTEST_OS_CYGWIN || GTEST_OS_SOLARIS))

#endif  

#ifndef GTEST_HAS_GLOBAL_WSTRING
# define GTEST_HAS_GLOBAL_WSTRING \
(GTEST_HAS_STD_WSTRING && GTEST_HAS_GLOBAL_STRING)
#endif  

#ifndef GTEST_HAS_RTTI

# ifdef _MSC_VER

#  ifdef _CPPRTTI  
#   define GTEST_HAS_RTTI 1
#  else
#   define GTEST_HAS_RTTI 0
#  endif

# elif defined(__GNUC__) && (GTEST_GCC_VER_ >= 40302)

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
# define GTEST_HAS_PTHREAD (GTEST_OS_LINUX || GTEST_OS_MAC || GTEST_OS_HPUX \
|| GTEST_OS_QNX)
#endif  

#if GTEST_HAS_PTHREAD
# include <pthread.h>  

# include <time.h>  
#endif

#ifndef GTEST_HAS_TR1_TUPLE
# if GTEST_OS_LINUX_ANDROID && defined(_STLPORT_MAJOR)
#  define GTEST_HAS_TR1_TUPLE 0
# else
#  define GTEST_HAS_TR1_TUPLE 1
# endif
#endif  

#ifndef GTEST_USE_OWN_TR1_TUPLE

# if (defined(__GNUC__) && !defined(__CUDACC__) && (GTEST_GCC_VER_ >= 40000) \
&& !GTEST_OS_QNX && !defined(_LIBCPP_VERSION)) || _MSC_VER >= 1600
#  define GTEST_ENV_HAS_TR1_TUPLE_ 1
# endif

# if GTEST_LANG_CXX11 && (!defined(__GLIBCXX__) || __GLIBCXX__ > 20110325)
#  define GTEST_ENV_HAS_STD_TUPLE_ 1
# endif

# if GTEST_ENV_HAS_TR1_TUPLE_ || GTEST_ENV_HAS_STD_TUPLE_
#  define GTEST_USE_OWN_TR1_TUPLE 0
# else
#  define GTEST_USE_OWN_TR1_TUPLE 1
# endif

#endif  

#if GTEST_HAS_TR1_TUPLE

# if GTEST_USE_OWN_TR1_TUPLE
#  include "gtest/internal/gtest-tuple.h"
# elif GTEST_ENV_HAS_STD_TUPLE_
#  include <tuple>
namespace std {
namespace tr1 {
using ::std::get;
using ::std::make_tuple;
using ::std::tuple;
using ::std::tuple_element;
using ::std::tuple_size;
}
}

# elif GTEST_OS_SYMBIAN

#  ifdef BOOST_HAS_TR1_TUPLE
#   undef BOOST_HAS_TR1_TUPLE
#  endif  

#  define BOOST_TR1_DETAIL_CONFIG_HPP_INCLUDED
#  include <tuple>

# elif defined(__GNUC__) && (GTEST_GCC_VER_ >= 40000)

#  if !GTEST_HAS_RTTI && GTEST_GCC_VER_ < 40302
#   define _TR1_FUNCTIONAL 1
#   include <tr1/tuple>
#   undef _TR1_FUNCTIONAL  
#  else
#   include <tr1/tuple>  
#  endif  

# else
#  include <tuple>  
# endif  

#endif  

#ifndef GTEST_HAS_CLONE

# if GTEST_OS_LINUX && !defined(__ia64__)
#  if GTEST_OS_LINUX_ANDROID
#    if defined(__arm__) && __ANDROID_API__ >= 9
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
# if GTEST_OS_WINDOWS_MOBILE || GTEST_OS_SYMBIAN
#  define GTEST_HAS_STREAM_REDIRECTION 0
# else
#  define GTEST_HAS_STREAM_REDIRECTION 1
# endif  
#endif  

#if (GTEST_OS_LINUX || GTEST_OS_CYGWIN || GTEST_OS_SOLARIS || \
(GTEST_OS_MAC && !GTEST_OS_IOS) || GTEST_OS_IOS_SIMULATOR || \
(GTEST_OS_WINDOWS_DESKTOP && _MSC_VER >= 1400) || \
GTEST_OS_WINDOWS_MINGW || GTEST_OS_AIX || GTEST_OS_HPUX || \
GTEST_OS_OPENBSD || GTEST_OS_QNX)
# define GTEST_HAS_DEATH_TEST 1
# include <vector>  
#endif

#define GTEST_HAS_PARAM_TEST 1


#if defined(__GNUC__) || (_MSC_VER >= 1400) || defined(__SUNPRO_CC) || \
defined(__IBMCPP__) || defined(__HP_aCC)
# define GTEST_HAS_TYPED_TEST 1
# define GTEST_HAS_TYPED_TEST_P 1
#endif

#if GTEST_HAS_PARAM_TEST && GTEST_HAS_TR1_TUPLE && !defined(__SUNPRO_CC)
# define GTEST_HAS_COMBINE 1
#endif

#define GTEST_WIDE_STRING_USES_UTF16_ \
(GTEST_OS_WINDOWS || GTEST_OS_CYGWIN || GTEST_OS_SYMBIAN || GTEST_OS_AIX)

#if GTEST_OS_LINUX
# define GTEST_CAN_STREAM_RESULTS_ 1
#endif


#ifdef __INTEL_COMPILER
# define GTEST_AMBIGUOUS_ELSE_BLOCKER_
#else
# define GTEST_AMBIGUOUS_ELSE_BLOCKER_ switch (0) case 0: default:  
#endif

#if defined(__GNUC__) && !defined(COMPILER_ICC)
# define GTEST_ATTRIBUTE_UNUSED_ __attribute__ ((unused))
#else
# define GTEST_ATTRIBUTE_UNUSED_
#endif

#define GTEST_DISALLOW_ASSIGN_(type)\
void operator=(type const &)

#define GTEST_DISALLOW_COPY_AND_ASSIGN_(type)\
type(type const &);\
GTEST_DISALLOW_ASSIGN_(type)

#if defined(__GNUC__) && (GTEST_GCC_VER_ >= 30400) && !defined(COMPILER_ICC)
# define GTEST_MUST_USE_RESULT_ __attribute__ ((warn_unused_result))
#else
# define GTEST_MUST_USE_RESULT_
#endif  

#ifndef GTEST_HAS_SEH

# if defined(_MSC_VER) || defined(__BORLANDC__)
#  define GTEST_HAS_SEH 1
# else
#  define GTEST_HAS_SEH 0
# endif

#endif  

#ifdef _MSC_VER

# if GTEST_LINKED_AS_SHARED_LIBRARY
#  define GTEST_API_ __declspec(dllimport)
# elif GTEST_CREATE_SHARED_LIBRARY
#  define GTEST_API_ __declspec(dllexport)
# endif

#endif  

#ifndef GTEST_API_
# define GTEST_API_
#endif

#ifdef __GNUC__
# define GTEST_NO_INLINE_ __attribute__((noinline))
#else
# define GTEST_NO_INLINE_
#endif

#if defined(__GLIBCXX__) || defined(_LIBCPP_VERSION)
# define GTEST_HAS_CXXABI_H_ 1
#else
# define GTEST_HAS_CXXABI_H_ 0
#endif

namespace testing {

class Message;

namespace internal {

class Secret;


template <bool>
struct CompileAssert {
};

#define GTEST_COMPILE_ASSERT_(expr, msg) \
typedef ::testing::internal::CompileAssert<(static_cast<bool>(expr))> \
msg[static_cast<bool>(expr) ? 1 : -1] GTEST_ATTRIBUTE_UNUSED_


template <typename T1, typename T2>
struct StaticAssertTypeEqHelper;

template <typename T>
struct StaticAssertTypeEqHelper<T, T> {};

#if GTEST_HAS_GLOBAL_STRING
typedef ::string string;
#else
typedef ::std::string string;
#endif  

#if GTEST_HAS_GLOBAL_WSTRING
typedef ::wstring wstring;
#elif GTEST_HAS_STD_WSTRING
typedef ::std::wstring wstring;
#endif  

GTEST_API_ bool IsTrue(bool condition);


template <typename T>
class scoped_ptr {
public:
typedef T element_type;

explicit scoped_ptr(T* p = NULL) : ptr_(p) {}
~scoped_ptr() { reset(); }

T& operator*() const { return *ptr_; }
T* operator->() const { return ptr_; }
T* get() const { return ptr_; }

T* release() {
T* const ptr = ptr_;
ptr_ = NULL;
return ptr;
}

void reset(T* p = NULL) {
if (p != ptr_) {
if (IsTrue(sizeof(T) > 0)) {  
delete ptr_;
}
ptr_ = p;
}
}

private:
T* ptr_;

GTEST_DISALLOW_COPY_AND_ASSIGN_(scoped_ptr);
};


class GTEST_API_ RE {
public:
RE(const RE& other) { Init(other.pattern()); }

RE(const ::std::string& regex) { Init(regex.c_str()); }  

#if GTEST_HAS_GLOBAL_STRING

RE(const ::string& regex) { Init(regex.c_str()); }  

#endif  

RE(const char* regex) { Init(regex); }  
~RE();

const char* pattern() const { return pattern_; }

static bool FullMatch(const ::std::string& str, const RE& re) {
return FullMatch(str.c_str(), re);
}
static bool PartialMatch(const ::std::string& str, const RE& re) {
return PartialMatch(str.c_str(), re);
}

#if GTEST_HAS_GLOBAL_STRING

static bool FullMatch(const ::string& str, const RE& re) {
return FullMatch(str.c_str(), re);
}
static bool PartialMatch(const ::string& str, const RE& re) {
return PartialMatch(str.c_str(), re);
}

#endif  

static bool FullMatch(const char* str, const RE& re);
static bool PartialMatch(const char* str, const RE& re);

private:
void Init(const char* regex);

const char* pattern_;
bool is_valid_;

#if GTEST_USES_POSIX_RE

regex_t full_regex_;     
regex_t partial_regex_;  

#else  

const char* full_pattern_;  

#endif

GTEST_DISALLOW_ASSIGN_(RE);
};

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

#define GTEST_LOG_(severity) \
::testing::internal::GTestLog(::testing::internal::GTEST_##severity, \
__FILE__, __LINE__).GetStream()

inline void LogToStderr() {}
inline void FlushInfoLog() { fflush(NULL); }

#define GTEST_CHECK_(condition) \
GTEST_AMBIGUOUS_ELSE_BLOCKER_ \
if (::testing::internal::IsTrue(condition)) \
; \
else \
GTEST_LOG_(FATAL) << "Condition " #condition " failed. "

#define GTEST_CHECK_POSIX_SUCCESS_(posix_call) \
if (const int gtest_error = (posix_call)) \
GTEST_LOG_(FATAL) << #posix_call << "failed with error " \
<< gtest_error

template<typename To>
inline To ImplicitCast_(To x) { return x; }

template<typename To, typename From>  
inline To DownCast_(From* f) {  
if (false) {
const To to = NULL;
::testing::internal::ImplicitCast_<From*>(to);
}

#if GTEST_HAS_RTTI
GTEST_CHECK_(f == NULL || dynamic_cast<To>(f) != NULL);
#endif
return static_cast<To>(f);
}

template <class Derived, class Base>
Derived* CheckedDowncastToActualType(Base* base) {
#if GTEST_HAS_RTTI
GTEST_CHECK_(typeid(*base) == typeid(Derived));
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


#if GTEST_HAS_DEATH_TEST

const ::std::vector<testing::internal::string>& GetInjectableArgvs();
void SetInjectableArgvs(const ::std::vector<testing::internal::string>*
new_argvs);

extern ::std::vector<testing::internal::string> g_argvs;

#endif  


#if GTEST_HAS_PTHREAD

inline void SleepMilliseconds(int n) {
const timespec time = {
0,                  
n * 1000L * 1000L,  
};
nanosleep(&time, NULL);
}

class Notification {
public:
Notification() : notified_(false) {
GTEST_CHECK_POSIX_SUCCESS_(pthread_mutex_init(&mutex_, NULL));
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

class ThreadWithParamBase {
public:
virtual ~ThreadWithParamBase() {}
virtual void Run() = 0;
};

extern "C" inline void* ThreadFuncWithCLinkage(void* thread) {
static_cast<ThreadWithParamBase*>(thread)->Run();
return NULL;
}

template <typename T>
class ThreadWithParam : public ThreadWithParamBase {
public:
typedef void (*UserThreadFunc)(T);

ThreadWithParam(
UserThreadFunc func, T param, Notification* thread_can_start)
: func_(func),
param_(param),
thread_can_start_(thread_can_start),
finished_(false) {
ThreadWithParamBase* const base = this;
GTEST_CHECK_POSIX_SUCCESS_(
pthread_create(&thread_, 0, &ThreadFuncWithCLinkage, base));
}
~ThreadWithParam() { Join(); }

void Join() {
if (!finished_) {
GTEST_CHECK_POSIX_SUCCESS_(pthread_join(thread_, 0));
finished_ = true;
}
}

virtual void Run() {
if (thread_can_start_ != NULL)
thread_can_start_->WaitForNotification();
func_(param_);
}

private:
const UserThreadFunc func_;  
const T param_;  
Notification* const thread_can_start_;
bool finished_;  
pthread_t thread_;  

GTEST_DISALLOW_COPY_AND_ASSIGN_(ThreadWithParam);
};

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

# define GTEST_DECLARE_STATIC_MUTEX_(mutex) \
extern ::testing::internal::MutexBase mutex

# define GTEST_DEFINE_STATIC_MUTEX_(mutex) \
::testing::internal::MutexBase mutex = { PTHREAD_MUTEX_INITIALIZER, false }

class Mutex : public MutexBase {
public:
Mutex() {
GTEST_CHECK_POSIX_SUCCESS_(pthread_mutex_init(&mutex_, NULL));
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
class ThreadLocal {
public:
ThreadLocal() : key_(CreateKey()),
default_() {}
explicit ThreadLocal(const T& value) : key_(CreateKey()),
default_(value) {}

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
if (holder != NULL) {
return CheckedDowncastToActualType<ValueHolder>(holder)->pointer();
}

ValueHolder* const new_holder = new ValueHolder(default_);
ThreadLocalValueHolderBase* const holder_base = new_holder;
GTEST_CHECK_POSIX_SUCCESS_(pthread_setspecific(key_, holder_base));
return new_holder->pointer();
}

const pthread_key_t key_;
const T default_;  

GTEST_DISALLOW_COPY_AND_ASSIGN_(ThreadLocal);
};

# define GTEST_IS_THREADSAFE 1

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
class ThreadLocal {
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

# define GTEST_IS_THREADSAFE 0

#endif  

GTEST_API_ size_t GetThreadCount();

#if defined(__SYMBIAN32__) || defined(__IBMCPP__) || defined(__SUNPRO_CC)
# define GTEST_ELLIPSIS_NEEDS_POD_ 1
#else
# define GTEST_CAN_COMPARE_NULL 1
#endif

#if defined(__SYMBIAN32__) || defined(__IBMCPP__)
# define GTEST_NEEDS_IS_POINTER_ 1
#endif

template <bool bool_value>
struct bool_constant {
typedef bool_constant<bool_value> type;
static const bool value = bool_value;
};
template <bool bool_value> const bool bool_constant<bool_value>::value;

typedef bool_constant<false> false_type;
typedef bool_constant<true> true_type;

template <typename T>
struct is_pointer : public false_type {};

template <typename T>
struct is_pointer<T*> : public true_type {};

template <typename Iterator>
struct IteratorTraits {
typedef typename Iterator::value_type value_type;
};

template <typename T>
struct IteratorTraits<T*> {
typedef T value_type;
};

template <typename T>
struct IteratorTraits<const T*> {
typedef T value_type;
};

#if GTEST_OS_WINDOWS
# define GTEST_PATH_SEP_ "\\"
# define GTEST_HAS_ALT_PATH_SEP_ 1
typedef __int64 BiggestInt;
#else
# define GTEST_PATH_SEP_ "/"
# define GTEST_HAS_ALT_PATH_SEP_ 0
typedef long long BiggestInt;  
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


namespace posix {


#if GTEST_OS_WINDOWS

typedef struct _stat StatStruct;

# ifdef __BORLANDC__
inline int IsATTY(int fd) { return isatty(fd); }
inline int StrCaseCmp(const char* s1, const char* s2) {
return stricmp(s1, s2);
}
inline char* StrDup(const char* src) { return strdup(src); }
# else  
#  if GTEST_OS_WINDOWS_MOBILE
inline int IsATTY(int ) { return 0; }
#  else
inline int IsATTY(int fd) { return _isatty(fd); }
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

#else

typedef struct stat StatStruct;

inline int FileNo(FILE* file) { return fileno(file); }
inline int IsATTY(int fd) { return isatty(fd); }
inline int Stat(const char* path, StatStruct* buf) { return stat(path, buf); }
inline int StrCaseCmp(const char* s1, const char* s2) {
return strcasecmp(s1, s2);
}
inline char* StrDup(const char* src) { return strdup(src); }
inline int RmDir(const char* dir) { return rmdir(dir); }
inline bool IsDir(const StatStruct& st) { return S_ISDIR(st.st_mode); }

#endif  


#ifdef _MSC_VER
# pragma warning(push)
# pragma warning(disable:4996)
#endif

inline const char* StrNCpy(char* dest, const char* src, size_t n) {
return strncpy(dest, src, n);
}


#if !GTEST_OS_WINDOWS_MOBILE
inline int ChDir(const char* dir) { return chdir(dir); }
#endif
inline FILE* FOpen(const char* path, const char* mode) {
return fopen(path, mode);
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
#if GTEST_OS_WINDOWS_MOBILE
return NULL;
#elif defined(__BORLANDC__) || defined(__SunOS_5_8) || defined(__SunOS_5_9)
const char* const env = getenv(name);
return (env != NULL && env[0] != '\0') ? env : NULL;
#else
return getenv(name);
#endif
}

#ifdef _MSC_VER
# pragma warning(pop)  
#endif

#if GTEST_OS_WINDOWS_MOBILE
void Abort();
#else
inline void Abort() { abort(); }
#endif  

}  

#if _MSC_VER >= 1400 && !GTEST_OS_WINDOWS_MOBILE
# define GTEST_SNPRINTF_(buffer, size, format, ...) \
_snprintf_s(buffer, size, size, format, __VA_ARGS__)
#elif defined(_MSC_VER)
# define GTEST_SNPRINTF_ _snprintf
#else
# define GTEST_SNPRINTF_ snprintf
#endif

const BiggestInt kMaxBiggestInt =
~(static_cast<BiggestInt>(1) << (8*sizeof(BiggestInt) - 1));

template <size_t size>
class TypeWithSize {
public:
typedef void UInt;
};

template <>
class TypeWithSize<4> {
public:
typedef int Int;
typedef unsigned int UInt;
};

template <>
class TypeWithSize<8> {
public:
#if GTEST_OS_WINDOWS
typedef __int64 Int;
typedef unsigned __int64 UInt;
#else
typedef long long Int;  
typedef unsigned long long UInt;  
#endif  
};

typedef TypeWithSize<4>::Int Int32;
typedef TypeWithSize<4>::UInt UInt32;
typedef TypeWithSize<8>::Int Int64;
typedef TypeWithSize<8>::UInt UInt64;
typedef TypeWithSize<8>::Int TimeInMillis;  


#define GTEST_FLAG(name) FLAGS_gtest_##name

#define GTEST_DECLARE_bool_(name) GTEST_API_ extern bool GTEST_FLAG(name)
#define GTEST_DECLARE_int32_(name) \
GTEST_API_ extern ::testing::internal::Int32 GTEST_FLAG(name)
#define GTEST_DECLARE_string_(name) \
GTEST_API_ extern ::std::string GTEST_FLAG(name)

#define GTEST_DEFINE_bool_(name, default_val, doc) \
GTEST_API_ bool GTEST_FLAG(name) = (default_val)
#define GTEST_DEFINE_int32_(name, default_val, doc) \
GTEST_API_ ::testing::internal::Int32 GTEST_FLAG(name) = (default_val)
#define GTEST_DEFINE_string_(name, default_val, doc) \
GTEST_API_ ::std::string GTEST_FLAG(name) = (default_val)

#define GTEST_EXCLUSIVE_LOCK_REQUIRED_(locks)
#define GTEST_LOCK_EXCLUDED_(locks)

bool ParseInt32(const Message& src_text, const char* str, Int32* value);

bool BoolFromGTestEnv(const char* flag, bool default_val);
GTEST_API_ Int32 Int32FromGTestEnv(const char* flag, Int32 default_val);
const char* StringFromGTestEnv(const char* flag, const char* default_val);

}  
}  

#endif  
