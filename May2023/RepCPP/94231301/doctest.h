
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic ignored "-Waggregate-return"
#endif

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-pragmas"
#pragma clang diagnostic ignored "-Wnon-virtual-dtor"
#pragma clang diagnostic ignored "-Wweak-vtables"
#pragma clang diagnostic ignored "-Wpadded"
#pragma clang diagnostic ignored "-Wdeprecated"
#pragma clang diagnostic ignored "-Wmissing-prototypes"
#pragma clang diagnostic ignored "-Wunused-local-typedef"
#pragma clang diagnostic ignored "-Wzero-as-null-pointer-constant"
#pragma clang diagnostic ignored "-Wc++11-long-long"
#endif 

#if defined(__GNUC__) && !defined(__clang__)
#if __GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ > 6)
#pragma GCC diagnostic push
#endif 
#pragma GCC diagnostic ignored "-Wunknown-pragmas"
#pragma GCC diagnostic ignored "-Weffc++"
#pragma GCC diagnostic ignored "-Wstrict-overflow"
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#pragma GCC diagnostic ignored "-Wctor-dtor-privacy"
#pragma GCC diagnostic ignored "-Wmissing-declarations"
#pragma GCC diagnostic ignored "-Wnon-virtual-dtor"
#pragma GCC diagnostic ignored "-Winline"
#pragma GCC diagnostic ignored "-Wlong-long"
#if __GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ > 6)
#pragma GCC diagnostic ignored "-Wzero-as-null-pointer-constant"
#endif 
#if __GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ > 7)
#pragma GCC diagnostic ignored "-Wunused-local-typedefs"
#endif 
#if __GNUC__ > 5 || (__GNUC__ == 5 && __GNUC_MINOR__ > 3)
#pragma GCC diagnostic ignored "-Wuseless-cast"
#endif 
#endif 

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4996) 
#pragma warning(disable : 4706) 
#pragma warning(disable : 4512) 
#pragma warning(disable : 4127) 
#endif                          

#ifndef DOCTEST_LIBRARY_INCLUDED
#define DOCTEST_LIBRARY_INCLUDED

#define DOCTEST_VERSION_MAJOR 1
#define DOCTEST_VERSION_MINOR 2
#define DOCTEST_VERSION_PATCH 1
#define DOCTEST_VERSION_STR "1.2.1"

#define DOCTEST_VERSION                                                                            \
(DOCTEST_VERSION_MAJOR * 10000 + DOCTEST_VERSION_MINOR * 100 + DOCTEST_VERSION_PATCH)


#if __cplusplus >= 201103L
#ifndef DOCTEST_CONFIG_WITH_DELETED_FUNCTIONS
#define DOCTEST_CONFIG_WITH_DELETED_FUNCTIONS
#endif 
#ifndef DOCTEST_CONFIG_WITH_RVALUE_REFERENCES
#define DOCTEST_CONFIG_WITH_RVALUE_REFERENCES
#endif 
#ifndef DOCTEST_CONFIG_WITH_NULLPTR
#define DOCTEST_CONFIG_WITH_NULLPTR
#endif 
#ifndef DOCTEST_CONFIG_WITH_LONG_LONG
#define DOCTEST_CONFIG_WITH_LONG_LONG
#endif 
#ifndef DOCTEST_CONFIG_WITH_STATIC_ASSERT
#define DOCTEST_CONFIG_WITH_STATIC_ASSERT
#endif 
#ifndef DOCTEST_CONFIG_WITH_VARIADIC_MACROS
#define DOCTEST_CONFIG_WITH_VARIADIC_MACROS
#endif 
#endif 

#ifndef __has_feature
#define __has_feature(x) 0
#endif 



#ifndef DOCTEST_CONFIG_WITH_DELETED_FUNCTIONS
#if defined(_MSC_VER) && (_MSC_VER >= 1800)
#define DOCTEST_CONFIG_WITH_DELETED_FUNCTIONS
#endif 
#if defined(__clang__) && __has_feature(cxx_deleted_functions)
#define DOCTEST_CONFIG_WITH_DELETED_FUNCTIONS
#endif 
#if defined(__GNUC__) && ((__GNUC__ == 4 && __GNUC_MINOR__ >= 4) || __GNUC__ > 4) &&               \
defined(__GXX_EXPERIMENTAL_CXX0X__)
#define DOCTEST_CONFIG_WITH_DELETED_FUNCTIONS
#endif 
#endif 

#if defined(DOCTEST_CONFIG_NO_DELETED_FUNCTIONS) && defined(DOCTEST_CONFIG_WITH_DELETED_FUNCTIONS)
#undef DOCTEST_CONFIG_WITH_DELETED_FUNCTIONS
#endif 


#ifndef DOCTEST_CONFIG_WITH_RVALUE_REFERENCES
#if defined(_MSC_VER) && (_MSC_VER >= 1600)
#define DOCTEST_CONFIG_WITH_RVALUE_REFERENCES
#endif 
#if defined(__clang__) && __has_feature(cxx_rvalue_references)
#define DOCTEST_CONFIG_WITH_RVALUE_REFERENCES
#endif 
#if defined(__GNUC__) && ((__GNUC__ == 4 && __GNUC_MINOR__ >= 3) || __GNUC__ > 4) &&               \
defined(__GXX_EXPERIMENTAL_CXX0X__)
#define DOCTEST_CONFIG_WITH_RVALUE_REFERENCES
#endif 
#endif 

#if defined(DOCTEST_CONFIG_NO_RVALUE_REFERENCES) && defined(DOCTEST_CONFIG_WITH_RVALUE_REFERENCES)
#undef DOCTEST_CONFIG_WITH_RVALUE_REFERENCES
#endif 


#ifndef DOCTEST_CONFIG_WITH_NULLPTR
#if defined(__clang__) && __has_feature(cxx_nullptr)
#define DOCTEST_CONFIG_WITH_NULLPTR
#endif 
#if defined(__GNUC__) && ((__GNUC__ == 4 && __GNUC_MINOR__ >= 6) || __GNUC__ > 4) &&               \
defined(__GXX_EXPERIMENTAL_CXX0X__)
#define DOCTEST_CONFIG_WITH_NULLPTR
#endif                                      
#if defined(_MSC_VER) && (_MSC_VER >= 1600) 
#define DOCTEST_CONFIG_WITH_NULLPTR
#endif 
#endif 

#if defined(DOCTEST_CONFIG_NO_NULLPTR) && defined(DOCTEST_CONFIG_WITH_NULLPTR)
#undef DOCTEST_CONFIG_WITH_NULLPTR
#endif 


#ifndef DOCTEST_CONFIG_WITH_VARIADIC_MACROS
#if defined(_MSC_VER) && _MSC_VER > 1400 && !defined(__EDGE__)
#define DOCTEST_CONFIG_WITH_VARIADIC_MACROS
#endif 
#if defined(__GNUC__) && ((__GNUC__ == 4 && __GNUC_MINOR__ >= 1) || __GNUC__ > 4) &&               \
defined(__GXX_EXPERIMENTAL_CXX0X__)
#define DOCTEST_CONFIG_WITH_VARIADIC_MACROS
#endif 
#endif 

#if defined(DOCTEST_CONFIG_NO_VARIADIC_MACROS) && defined(DOCTEST_CONFIG_WITH_VARIADIC_MACROS)
#undef DOCTEST_CONFIG_WITH_VARIADIC_MACROS
#endif 


#ifndef DOCTEST_CONFIG_WITH_LONG_LONG
#if defined(_MSC_VER) && (_MSC_VER >= 1400)
#define DOCTEST_CONFIG_WITH_LONG_LONG
#endif 
#if(defined(__clang__) ||                                                                          \
(defined(__GNUC__) && ((__GNUC__ == 4 && __GNUC_MINOR__ >= 5) || __GNUC__ > 4))) &&            \
defined(__GXX_EXPERIMENTAL_CXX0X__)
#define DOCTEST_CONFIG_WITH_LONG_LONG
#endif 
#endif 

#if defined(DOCTEST_CONFIG_NO_LONG_LONG) && defined(DOCTEST_CONFIG_WITH_LONG_LONG)
#undef DOCTEST_CONFIG_WITH_LONG_LONG
#endif 


#ifndef DOCTEST_CONFIG_WITH_STATIC_ASSERT
#if defined(__clang__) && __has_feature(cxx_static_assert)
#define DOCTEST_CONFIG_WITH_STATIC_ASSERT
#endif 
#if defined(__GNUC__) && ((__GNUC__ == 4 && __GNUC_MINOR__ >= 3) || __GNUC__ > 4) &&               \
defined(__GXX_EXPERIMENTAL_CXX0X__)
#define DOCTEST_CONFIG_WITH_STATIC_ASSERT
#endif                                      
#if defined(_MSC_VER) && (_MSC_VER >= 1600) 
#define DOCTEST_CONFIG_WITH_STATIC_ASSERT
#endif 
#endif 

#if defined(DOCTEST_CONFIG_NO_STATIC_ASSERT) && defined(DOCTEST_CONFIG_WITH_STATIC_ASSERT)
#undef DOCTEST_CONFIG_WITH_STATIC_ASSERT
#endif 


#if defined(DOCTEST_CONFIG_WITH_RVALUE_REFERENCES) || defined(DOCTEST_CONFIG_WITH_LONG_LONG) ||    \
defined(DOCTEST_CONFIG_WITH_DELETED_FUNCTIONS) || defined(DOCTEST_CONFIG_WITH_NULLPTR) ||  \
defined(DOCTEST_CONFIG_WITH_VARIADIC_MACROS) || defined(DOCTEST_CONFIG_WITH_STATIC_ASSERT)
#define DOCTEST_NO_CPP11_COMPAT
#endif 

#if defined(__clang__) && defined(DOCTEST_NO_CPP11_COMPAT)
#pragma clang diagnostic ignored "-Wc++98-compat"
#pragma clang diagnostic ignored "-Wc++98-compat-pedantic"
#endif 

#if defined(_MSC_VER) && !defined(DOCTEST_CONFIG_WINDOWS_SEH)
#define DOCTEST_CONFIG_WINDOWS_SEH
#endif 
#if defined(DOCTEST_CONFIG_NO_WINDOWS_SEH) && defined(DOCTEST_CONFIG_WINDOWS_SEH)
#undef DOCTEST_CONFIG_WINDOWS_SEH
#endif 

#if !defined(_WIN32) && !defined(__QNX__) && !defined(DOCTEST_CONFIG_POSIX_SIGNALS)
#define DOCTEST_CONFIG_POSIX_SIGNALS
#endif 
#if defined(DOCTEST_CONFIG_NO_POSIX_SIGNALS) && defined(DOCTEST_CONFIG_POSIX_SIGNALS)
#undef DOCTEST_CONFIG_POSIX_SIGNALS
#endif 

#ifndef DOCTEST_CONFIG_NO_EXCEPTIONS
#if defined(__GNUC__) && !defined(__EXCEPTIONS)
#define DOCTEST_CONFIG_NO_EXCEPTIONS
#endif 
#endif 

#ifdef DOCTEST_CONFIG_NO_EXCEPTIONS_BUT_WITH_ALL_ASSERTS
#ifndef DOCTEST_CONFIG_NO_EXCEPTIONS
#define DOCTEST_CONFIG_NO_EXCEPTIONS
#endif 
#endif 

#if defined(DOCTEST_CONFIG_NO_EXCEPTIONS) && !defined(DOCTEST_CONFIG_NO_TRY_CATCH_IN_ASSERTS)
#define DOCTEST_CONFIG_NO_TRY_CATCH_IN_ASSERTS
#endif 

#if defined(DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN) && !defined(DOCTEST_CONFIG_IMPLEMENT)
#define DOCTEST_CONFIG_IMPLEMENT
#endif 

#if defined _WIN32 || defined __CYGWIN__
#ifdef __GNUC__
#define DOCTEST_SYMBOL_EXPORT __attribute__((dllexport))
#define DOCTEST_SYMBOL_IMPORT __attribute__((dllimport))
#else 
#define DOCTEST_SYMBOL_EXPORT __declspec(dllexport)
#define DOCTEST_SYMBOL_IMPORT __declspec(dllimport)
#endif 
#else  
#define DOCTEST_SYMBOL_EXPORT __attribute__((visibility("default")))
#define DOCTEST_SYMBOL_IMPORT
#endif 

#ifdef DOCTEST_CONFIG_IMPLEMENTATION_IN_DLL
#ifdef DOCTEST_CONFIG_IMPLEMENT
#define DOCTEST_INTERFACE DOCTEST_SYMBOL_EXPORT
#else 
#define DOCTEST_INTERFACE DOCTEST_SYMBOL_IMPORT
#endif 
#else  
#define DOCTEST_INTERFACE
#endif 

#ifdef _MSC_VER
#define DOCTEST_NOINLINE __declspec(noinline)
#else 
#define DOCTEST_NOINLINE __attribute__((noinline))
#endif 

#ifndef DOCTEST_CONFIG_NUM_CAPTURES_ON_STACK
#define DOCTEST_CONFIG_NUM_CAPTURES_ON_STACK 5
#endif 


#define DOCTEST_CAT_IMPL(s1, s2) s1##s2
#define DOCTEST_CAT(s1, s2) DOCTEST_CAT_IMPL(s1, s2)
#ifdef __COUNTER__ 
#define DOCTEST_ANONYMOUS(x) DOCTEST_CAT(x, __COUNTER__)
#else 
#define DOCTEST_ANONYMOUS(x) DOCTEST_CAT(x, __LINE__)
#endif 

#ifdef DOCTEST_CONFIG_WITH_VARIADIC_MACROS
#define DOCTEST_TOSTR_IMPL(...) #__VA_ARGS__
#define DOCTEST_TOSTR(...) DOCTEST_TOSTR_IMPL(__VA_ARGS__)
#else 
#define DOCTEST_TOSTR_IMPL(x) #x
#define DOCTEST_TOSTR(x) DOCTEST_TOSTR_IMPL(x)
#endif 

#define DOCTEST_STR_CONCAT_TOSTR(s1, s2) DOCTEST_TOSTR(s1) DOCTEST_TOSTR(s2)

#define DOCTEST_COUNTOF(x) (sizeof(x) / sizeof(x[0]))

#ifndef DOCTEST_CONFIG_ASSERTION_PARAMETERS_BY_VALUE
#define DOCTEST_REF_WRAP(x) x&
#else 
#define DOCTEST_REF_WRAP(x) x
#endif 

#if defined(__MAC_OS_X_VERSION_MIN_REQUIRED)
#define DOCTEST_PLATFORM_MAC
#elif defined(__IPHONE_OS_VERSION_MIN_REQUIRED)
#define DOCTEST_PLATFORM_IPHONE
#elif defined(_WIN32) || defined(_MSC_VER)
#define DOCTEST_PLATFORM_WINDOWS
#else
#define DOCTEST_PLATFORM_LINUX
#endif

#if defined(__clang__)
#define DOCTEST_GLOBAL_NO_WARNINGS(var)                                                            \
_Pragma("clang diagnostic push")                                                               \
_Pragma("clang diagnostic ignored \"-Wglobal-constructors\"") static int var
#define DOCTEST_GLOBAL_NO_WARNINGS_END() _Pragma("clang diagnostic pop")
#elif defined(__GNUC__)
#define DOCTEST_GLOBAL_NO_WARNINGS(var) static int var __attribute__((unused))
#define DOCTEST_GLOBAL_NO_WARNINGS_END()
#else 
#define DOCTEST_GLOBAL_NO_WARNINGS(var) static int var
#define DOCTEST_GLOBAL_NO_WARNINGS_END()
#endif 

#ifdef DOCTEST_PLATFORM_MAC
#define DOCTEST_BREAK_INTO_DEBUGGER() __asm__("int $3\n" : :)
#elif defined(_MSC_VER)
#define DOCTEST_BREAK_INTO_DEBUGGER() __debugbreak()
#elif defined(__MINGW32__)
extern "C" __declspec(dllimport) void __stdcall DebugBreak();
#define DOCTEST_BREAK_INTO_DEBUGGER() ::DebugBreak()
#else 
#define DOCTEST_BREAK_INTO_DEBUGGER() ((void)0)
#endif 

#ifdef __clang__
#include <ciso646>
#endif 

#ifdef _LIBCPP_VERSION
#include <iosfwd>
#else 
#ifndef DOCTEST_CONFIG_USE_IOSFWD
namespace std
{
template <class charT>
struct char_traits;
template <>
struct char_traits<char>;
template <class charT, class traits>
class basic_ostream;
typedef basic_ostream<char, char_traits<char> > ostream;
} 
#else 
#include <iosfwd>
#endif 
#endif 

#ifdef DOCTEST_CONFIG_WITH_STATIC_ASSERT
#define DOCTEST_STATIC_ASSERT(expression, message) static_assert(expression, #message)
#else 
#define DOCTEST_STATIC_ASSERT(expression, message)                                                 \
struct DOCTEST_CAT(__static_assertion_at_line_, __LINE__)                                      \
{                                                                                              \
doctest::detail::static_assert_impl::StaticAssertion<static_cast<bool>((expression))>      \
DOCTEST_CAT(DOCTEST_CAT(DOCTEST_CAT(STATIC_ASSERTION_FAILED_AT_LINE_, __LINE__),   \
_),                                                        \
message);                                                              \
};                                                                                             \
typedef doctest::detail::static_assert_impl::StaticAssertionTest<static_cast<int>(             \
sizeof(DOCTEST_CAT(__static_assertion_at_line_, __LINE__)))>                           \
DOCTEST_CAT(__static_assertion_test_at_line_, __LINE__)
#endif 

#ifdef DOCTEST_CONFIG_WITH_NULLPTR
#ifdef _LIBCPP_VERSION
#include <cstddef>
#else  
namespace std
{ typedef decltype(nullptr) nullptr_t; }
#endif 
#endif 

#ifndef DOCTEST_CONFIG_DISABLE
namespace doctest
{
namespace detail
{
struct TestSuite
{
const char* m_test_suite;
const char* m_description;
bool        m_skip;
bool        m_may_fail;
bool        m_should_fail;
int         m_expected_failures;
double      m_timeout;

TestSuite& operator*(const char* in) {
m_test_suite = in;
m_description       = 0;
m_skip              = false;
m_may_fail          = false;
m_should_fail       = false;
m_expected_failures = 0;
m_timeout           = 0;
return *this;
}

template <typename T>
TestSuite& operator*(const T& in) {
in.fill(*this);
return *this;
}
};
} 
} 

namespace doctest_detail_test_suite_ns
{
DOCTEST_INTERFACE doctest::detail::TestSuite& getCurrentTestSuite();
} 

#endif 

namespace doctest
{
class DOCTEST_INTERFACE String
{
static const unsigned len  = 24;      
static const unsigned last = len - 1; 

struct view 
{
char*    ptr;
unsigned size;
unsigned capacity;
};

union
{
char buf[len];
view data;
};

void copy(const String& other);

void setOnHeap() { *reinterpret_cast<unsigned char*>(&buf[last]) = 128; }
void setLast(unsigned in = last) { buf[last] = char(in); }

public:
String() {
buf[0] = '\0';
setLast();
}

String(const char* in);

String(const String& other) { copy(other); }

~String() {
if(!isOnStack())
delete[] data.ptr;
}

DOCTEST_NOINLINE String& operator=(const String& other) {
if(!isOnStack())
delete[] data.ptr;

copy(other);

return *this;
}
String& operator+=(const String& other);

String operator+(const String& other) const { return String(*this) += other; }

#ifdef DOCTEST_CONFIG_WITH_RVALUE_REFERENCES
String(String&& other);
String& operator=(String&& other);
#endif 

bool isOnStack() const { return (buf[last] & 128) == 0; }

char operator[](unsigned i) const { return const_cast<String*>(this)->operator[](i); } 
char& operator[](unsigned i) {
if(isOnStack())
return reinterpret_cast<char*>(buf)[i];
return data.ptr[i];
}

const char* c_str() const { return const_cast<String*>(this)->c_str(); } 
char*       c_str() {
if(isOnStack())
return reinterpret_cast<char*>(buf);
return data.ptr;
}

unsigned size() const {
if(isOnStack())
return last - (unsigned(buf[last]) & 31); 
return data.size;
}

unsigned capacity() const {
if(isOnStack())
return len;
return data.capacity;
}

int compare(const char* other, bool no_case = false) const;
int compare(const String& other, bool no_case = false) const;
};

inline bool operator==(const String& lhs, const String& rhs) { return lhs.compare(rhs) == 0; }
inline bool operator!=(const String& lhs, const String& rhs) { return lhs.compare(rhs) != 0; }
inline bool operator< (const String& lhs, const String& rhs) { return lhs.compare(rhs) < 0; }
inline bool operator> (const String& lhs, const String& rhs) { return lhs.compare(rhs) > 0; }
inline bool operator<=(const String& lhs, const String& rhs) { return (lhs != rhs) ? lhs.compare(rhs) < 0 : true; }
inline bool operator>=(const String& lhs, const String& rhs) { return (lhs != rhs) ? lhs.compare(rhs) > 0 : true; }

DOCTEST_INTERFACE std::ostream& operator<<(std::ostream& stream, const String& in);

namespace detail
{
#ifndef DOCTEST_CONFIG_WITH_STATIC_ASSERT
namespace static_assert_impl
{
template <bool>
struct StaticAssertion;

template <>
struct StaticAssertion<true>
{};

template <int i>
struct StaticAssertionTest
{};
}  
#endif 

namespace traits
{
template <typename T>
struct remove_const
{ typedef T type; };

template <typename T>
struct remove_const<const T>
{ typedef T type; };

template <typename T>
struct remove_volatile
{ typedef T type; };

template <typename T>
struct remove_volatile<volatile T>
{ typedef T type; };

template <typename T>
struct remove_cv
{ typedef typename remove_volatile<typename remove_const<T>::type>::type type; };

template <typename T>
struct is_pointer_helper
{ static const bool value = false; };

template <typename T>
struct is_pointer_helper<T*>
{ static const bool value = true; };

template <typename T>
struct is_pointer
{ static const bool value = is_pointer_helper<typename remove_cv<T>::type>::value; };

template <bool CONDITION, typename TYPE = void>
struct enable_if
{};

template <typename TYPE>
struct enable_if<true, TYPE>
{ typedef TYPE type; };

template <typename T>
struct remove_reference
{ typedef T type; };

template <typename T>
struct remove_reference<T&>
{ typedef T type; };

template <typename T, typename AT_1 = void>
class is_constructible_impl
{
private:
template <typename C_T, typename C_AT_1>
static bool test(typename enable_if< 
sizeof(C_T) ==
sizeof(C_T(static_cast<C_AT_1>(
*static_cast<typename remove_reference<C_AT_1>::type*>(
0))))>::type*);

template <typename, typename>
static int test(...); 

public:
static const bool value = sizeof(test<T, AT_1>(0)) == sizeof(bool);
};

template <typename T>
class is_constructible_impl<T, void>
{
private:
template <typename C_T>
static C_T testFun(C_T); 

template <typename C_T>
static bool test(typename enable_if< 
sizeof(C_T) == sizeof(testFun(C_T()))>::type*);

template <typename>
static int test(...); 

public:
static const bool value = sizeof(test<T>(0)) == sizeof(bool);
};

#ifndef _MSC_VER
template <typename T, typename AT_1 = void>
class is_constructible
{
public:
static const bool value = is_pointer<typename remove_reference<T>::type>::value ?
false :
is_constructible_impl<T, AT_1>::value;
};
#elif defined(_MSC_VER) && (_MSC_VER >= 1700)
template <typename T, typename AT_1>
struct is_constructible
{ static const bool value = __is_constructible(T, AT_1); };
#elif defined(_MSC_VER)
template <typename T, typename AT_1>
struct is_constructible
{ static const bool value = false; };
#endif 
}  

template <typename T>
struct deferred_false
{ static const bool value = false; };

inline void* getNull() { return 0; }

namespace has_insertion_operator_impl
{
typedef char no;
typedef char yes[2];

struct any_t
{
template <typename T>
any_t(const DOCTEST_REF_WRAP(T));
};

yes& testStreamable(std::ostream&);
no   testStreamable(no);

no operator<<(const std::ostream&, const any_t&);

template <typename T>
struct has_insertion_operator
{
static std::ostream& s;
static const DOCTEST_REF_WRAP(T) t;
static const bool value = sizeof(testStreamable(s << t)) == sizeof(yes);
};
} 

template <typename T>
struct has_insertion_operator : has_insertion_operator_impl::has_insertion_operator<T>
{};

DOCTEST_INTERFACE void my_memcpy(void* dest, const void* src, unsigned num);
DOCTEST_INTERFACE unsigned my_strlen(const char* in);

DOCTEST_INTERFACE std::ostream* createStream();
DOCTEST_INTERFACE String getStreamResult(std::ostream*);
DOCTEST_INTERFACE void   freeStream(std::ostream*);

template <bool C>
struct StringMakerBase
{
template <typename T>
static String convert(const DOCTEST_REF_WRAP(T)) {
return "{?}";
}
};

template <>
struct StringMakerBase<true>
{
template <typename T>
static String convert(const DOCTEST_REF_WRAP(T) in) {
std::ostream* stream = createStream();
*stream << in;
String result = getStreamResult(stream);
freeStream(stream);
return result;
}
};

DOCTEST_INTERFACE String rawMemoryToString(const void* object, unsigned size);

template <typename T>
String rawMemoryToString(const DOCTEST_REF_WRAP(T) object) {
return rawMemoryToString(&object, sizeof(object));
}

class NullType
{};

template <class T, class U>
struct Typelist
{
typedef T Head;
typedef U Tail;
};

template <class TList, class Callable>
struct ForEachType;

template <class Head, class Tail, class Callable>
struct ForEachType<Typelist<Head, Tail>, Callable> : public ForEachType<Tail, Callable>
{
enum
{
value = 1 + ForEachType<Tail, Callable>::value
};

explicit ForEachType(Callable& callable)
: ForEachType<Tail, Callable>(callable) {
#if defined(_MSC_VER) && _MSC_VER <= 1900
callable.operator()<value, Head>();
#else  
callable.template operator()<value, Head>();
#endif 
}
};

template <class Head, class Callable>
struct ForEachType<Typelist<Head, NullType>, Callable>
{
public:
enum
{
value = 0
};

explicit ForEachType(Callable& callable) {
#if defined(_MSC_VER) && _MSC_VER <= 1900
callable.operator()<value, Head>();
#else  
callable.template operator()<value, Head>();
#endif 
}
};

template <typename T>
const char* type_to_string() {
return "<>";
}
} 

template <typename T1 = detail::NullType, typename T2 = detail::NullType,
typename T3 = detail::NullType, typename T4 = detail::NullType,
typename T5 = detail::NullType, typename T6 = detail::NullType,
typename T7 = detail::NullType, typename T8 = detail::NullType,
typename T9 = detail::NullType, typename T10 = detail::NullType,
typename T11 = detail::NullType, typename T12 = detail::NullType,
typename T13 = detail::NullType, typename T14 = detail::NullType,
typename T15 = detail::NullType, typename T16 = detail::NullType,
typename T17 = detail::NullType, typename T18 = detail::NullType,
typename T19 = detail::NullType, typename T20 = detail::NullType,
typename T21 = detail::NullType, typename T22 = detail::NullType,
typename T23 = detail::NullType, typename T24 = detail::NullType,
typename T25 = detail::NullType, typename T26 = detail::NullType,
typename T27 = detail::NullType, typename T28 = detail::NullType,
typename T29 = detail::NullType, typename T30 = detail::NullType,
typename T31 = detail::NullType, typename T32 = detail::NullType,
typename T33 = detail::NullType, typename T34 = detail::NullType,
typename T35 = detail::NullType, typename T36 = detail::NullType,
typename T37 = detail::NullType, typename T38 = detail::NullType,
typename T39 = detail::NullType, typename T40 = detail::NullType,
typename T41 = detail::NullType, typename T42 = detail::NullType,
typename T43 = detail::NullType, typename T44 = detail::NullType,
typename T45 = detail::NullType, typename T46 = detail::NullType,
typename T47 = detail::NullType, typename T48 = detail::NullType,
typename T49 = detail::NullType, typename T50 = detail::NullType,
typename T51 = detail::NullType, typename T52 = detail::NullType,
typename T53 = detail::NullType, typename T54 = detail::NullType,
typename T55 = detail::NullType, typename T56 = detail::NullType,
typename T57 = detail::NullType, typename T58 = detail::NullType,
typename T59 = detail::NullType, typename T60 = detail::NullType>
struct Types
{
private:
typedef typename Types<T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16, T17,
T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28, T29, T30, T31,
T32, T33, T34, T35, T36, T37, T38, T39, T40, T41, T42, T43, T44, T45,
T46, T47, T48, T49, T50, T51, T52, T53, T54, T55, T56, T57, T58, T59,
T60>::Result TailResult;

public:
typedef detail::Typelist<T1, TailResult> Result;
};

template <>
struct Types<>
{ typedef detail::NullType Result; };

template <typename T>
struct StringMaker : detail::StringMakerBase<detail::has_insertion_operator<T>::value>
{};

template <typename T>
struct StringMaker<T*>
{
template <typename U>
static String convert(U* p) {
if(p)
return detail::rawMemoryToString(p);
return "NULL";
}
};

template <typename R, typename C>
struct StringMaker<R C::*>
{
static String convert(R C::*p) {
if(p)
return detail::rawMemoryToString(p);
return "NULL";
}
};

template <typename T>
String toString(const DOCTEST_REF_WRAP(T) value) {
return StringMaker<T>::convert(value);
}

#ifdef DOCTEST_CONFIG_TREAT_CHAR_STAR_AS_STRING
DOCTEST_INTERFACE String toString(char* in);
DOCTEST_INTERFACE String toString(const char* in);
#endif 
DOCTEST_INTERFACE String toString(bool in);
DOCTEST_INTERFACE String toString(float in);
DOCTEST_INTERFACE String toString(double in);
DOCTEST_INTERFACE String toString(double long in);

DOCTEST_INTERFACE String toString(char in);
DOCTEST_INTERFACE String toString(char signed in);
DOCTEST_INTERFACE String toString(char unsigned in);
DOCTEST_INTERFACE String toString(int short in);
DOCTEST_INTERFACE String toString(int short unsigned in);
DOCTEST_INTERFACE String toString(int in);
DOCTEST_INTERFACE String toString(int unsigned in);
DOCTEST_INTERFACE String toString(int long in);
DOCTEST_INTERFACE String toString(int long unsigned in);

#ifdef DOCTEST_CONFIG_WITH_LONG_LONG
DOCTEST_INTERFACE String toString(int long long in);
DOCTEST_INTERFACE String toString(int long long unsigned in);
#endif 

#ifdef DOCTEST_CONFIG_WITH_NULLPTR
DOCTEST_INTERFACE String toString(std::nullptr_t in);
#endif 

class DOCTEST_INTERFACE Approx
{
public:
explicit Approx(double value);

Approx operator()(double value) const {
Approx approx(value);
approx.epsilon(m_epsilon);
approx.scale(m_scale);
return approx;
}

template <typename T>
explicit Approx(const T& value,
typename detail::traits::enable_if<
detail::traits::is_constructible<double, T>::value>::type* =
static_cast<T*>(detail::getNull())) {
*this = Approx(static_cast<double>(value));
}

DOCTEST_INTERFACE friend bool operator==(double lhs, Approx const& rhs);
friend bool operator==(Approx const& lhs, double rhs) { return operator==(rhs, lhs); }
friend bool operator!=(double lhs, Approx const& rhs) { return !operator==(lhs, rhs); }
friend bool operator!=(Approx const& lhs, double rhs) { return !operator==(rhs, lhs); }
friend bool operator<=(double lhs, Approx const& rhs) { return lhs < rhs.m_value || lhs == rhs; }
friend bool operator<=(Approx const& lhs, double rhs) { return lhs.m_value < rhs || lhs == rhs; }
friend bool operator>=(double lhs, Approx const& rhs) { return lhs > rhs.m_value || lhs == rhs; }
friend bool operator>=(Approx const& lhs, double rhs) { return lhs.m_value > rhs || lhs == rhs; }
friend bool operator< (double lhs, Approx const& rhs) { return lhs < rhs.m_value && lhs != rhs; }
friend bool operator< (Approx const& lhs, double rhs) { return lhs.m_value < rhs && lhs != rhs; }
friend bool operator> (double lhs, Approx const& rhs) { return lhs > rhs.m_value && lhs != rhs; }
friend bool operator> (Approx const& lhs, double rhs) { return lhs.m_value > rhs && lhs != rhs; }

#define DOCTEST_APPROX_PREFIX \
template <typename T> friend typename detail::traits::enable_if<detail::traits::is_constructible<double, T>::value, bool>::type

DOCTEST_APPROX_PREFIX operator==(const T& lhs, const Approx& rhs) { return operator==(double(lhs), rhs); }
DOCTEST_APPROX_PREFIX operator==(const Approx& lhs, const T& rhs) { return operator==(rhs, lhs); }
DOCTEST_APPROX_PREFIX operator!=(const T& lhs, const Approx& rhs) { return !operator==(lhs, rhs); }
DOCTEST_APPROX_PREFIX operator!=(const Approx& lhs, const T& rhs) { return !operator==(rhs, lhs); }
DOCTEST_APPROX_PREFIX operator<=(const T& lhs, const Approx& rhs) { return double(lhs) < rhs.m_value || lhs == rhs; }
DOCTEST_APPROX_PREFIX operator<=(const Approx& lhs, const T& rhs) { return lhs.m_value < double(rhs) || lhs == rhs; }
DOCTEST_APPROX_PREFIX operator>=(const T& lhs, const Approx& rhs) { return double(lhs) > rhs.m_value || lhs == rhs; }
DOCTEST_APPROX_PREFIX operator>=(const Approx& lhs, const T& rhs) { return lhs.m_value > double(rhs) || lhs == rhs; }
DOCTEST_APPROX_PREFIX operator< (const T& lhs, const Approx& rhs) { return double(lhs) < rhs.m_value && lhs != rhs; }
DOCTEST_APPROX_PREFIX operator< (const Approx& lhs, const T& rhs) { return lhs.m_value < double(rhs) && lhs != rhs; }
DOCTEST_APPROX_PREFIX operator> (const T& lhs, const Approx& rhs) { return double(lhs) > rhs.m_value && lhs != rhs; }
DOCTEST_APPROX_PREFIX operator> (const Approx& lhs, const T& rhs) { return lhs.m_value > double(rhs) && lhs != rhs; }
#undef DOCTEST_APPROX_PREFIX

Approx& epsilon(double newEpsilon) {
m_epsilon = (newEpsilon);
return *this;
}

template <typename T>
typename detail::traits::enable_if<detail::traits::is_constructible<double, T>::value,
Approx&>::type
epsilon(const T& newEpsilon) {
m_epsilon = static_cast<double>(newEpsilon);
return *this;
}

Approx& scale(double newScale) {
m_scale = (newScale);
return *this;
}

template <typename T>
typename detail::traits::enable_if<detail::traits::is_constructible<double, T>::value,
Approx&>::type
scale(const T& newScale) {
m_scale = static_cast<double>(newScale);
return *this;
}

String toString() const;

private:
double m_epsilon;
double m_scale;
double m_value;
};

template <>
inline String toString<Approx>(const DOCTEST_REF_WRAP(Approx) value) {
return value.toString();
}

#if !defined(DOCTEST_CONFIG_DISABLE)

namespace detail
{
typedef void (*funcType)();

namespace assertType
{
enum Enum
{

is_warn    = 1,
is_check   = 2,
is_require = 4,

is_throws    = 8,
is_throws_as = 16,
is_nothrow   = 32,

is_fast  = 64, 
is_false = 128,
is_unary = 256,

is_eq = 512,
is_ne = 1024,

is_lt = 2048,
is_gt = 4096,

is_ge = 8192,
is_le = 16384,


DT_WARN    = is_warn,
DT_CHECK   = is_check,
DT_REQUIRE = is_require,

DT_WARN_FALSE    = is_false | is_warn,
DT_CHECK_FALSE   = is_false | is_check,
DT_REQUIRE_FALSE = is_false | is_require,

DT_WARN_THROWS    = is_throws | is_warn,
DT_CHECK_THROWS   = is_throws | is_check,
DT_REQUIRE_THROWS = is_throws | is_require,

DT_WARN_THROWS_AS    = is_throws_as | is_warn,
DT_CHECK_THROWS_AS   = is_throws_as | is_check,
DT_REQUIRE_THROWS_AS = is_throws_as | is_require,

DT_WARN_NOTHROW    = is_nothrow | is_warn,
DT_CHECK_NOTHROW   = is_nothrow | is_check,
DT_REQUIRE_NOTHROW = is_nothrow | is_require,

DT_WARN_EQ    = is_eq | is_warn,
DT_CHECK_EQ   = is_eq | is_check,
DT_REQUIRE_EQ = is_eq | is_require,

DT_WARN_NE    = is_ne | is_warn,
DT_CHECK_NE   = is_ne | is_check,
DT_REQUIRE_NE = is_ne | is_require,

DT_WARN_GT    = is_gt | is_warn,
DT_CHECK_GT   = is_gt | is_check,
DT_REQUIRE_GT = is_gt | is_require,

DT_WARN_LT    = is_lt | is_warn,
DT_CHECK_LT   = is_lt | is_check,
DT_REQUIRE_LT = is_lt | is_require,

DT_WARN_GE    = is_ge | is_warn,
DT_CHECK_GE   = is_ge | is_check,
DT_REQUIRE_GE = is_ge | is_require,

DT_WARN_LE    = is_le | is_warn,
DT_CHECK_LE   = is_le | is_check,
DT_REQUIRE_LE = is_le | is_require,

DT_WARN_UNARY    = is_unary | is_warn,
DT_CHECK_UNARY   = is_unary | is_check,
DT_REQUIRE_UNARY = is_unary | is_require,

DT_WARN_UNARY_FALSE    = is_false | is_unary | is_warn,
DT_CHECK_UNARY_FALSE   = is_false | is_unary | is_check,
DT_REQUIRE_UNARY_FALSE = is_false | is_unary | is_require,

DT_FAST_WARN_EQ    = is_fast | is_eq | is_warn,
DT_FAST_CHECK_EQ   = is_fast | is_eq | is_check,
DT_FAST_REQUIRE_EQ = is_fast | is_eq | is_require,

DT_FAST_WARN_NE    = is_fast | is_ne | is_warn,
DT_FAST_CHECK_NE   = is_fast | is_ne | is_check,
DT_FAST_REQUIRE_NE = is_fast | is_ne | is_require,

DT_FAST_WARN_GT    = is_fast | is_gt | is_warn,
DT_FAST_CHECK_GT   = is_fast | is_gt | is_check,
DT_FAST_REQUIRE_GT = is_fast | is_gt | is_require,

DT_FAST_WARN_LT    = is_fast | is_lt | is_warn,
DT_FAST_CHECK_LT   = is_fast | is_lt | is_check,
DT_FAST_REQUIRE_LT = is_fast | is_lt | is_require,

DT_FAST_WARN_GE    = is_fast | is_ge | is_warn,
DT_FAST_CHECK_GE   = is_fast | is_ge | is_check,
DT_FAST_REQUIRE_GE = is_fast | is_ge | is_require,

DT_FAST_WARN_LE    = is_fast | is_le | is_warn,
DT_FAST_CHECK_LE   = is_fast | is_le | is_check,
DT_FAST_REQUIRE_LE = is_fast | is_le | is_require,

DT_FAST_WARN_UNARY    = is_fast | is_unary | is_warn,
DT_FAST_CHECK_UNARY   = is_fast | is_unary | is_check,
DT_FAST_REQUIRE_UNARY = is_fast | is_unary | is_require,

DT_FAST_WARN_UNARY_FALSE    = is_fast | is_false | is_unary | is_warn,
DT_FAST_CHECK_UNARY_FALSE   = is_fast | is_false | is_unary | is_check,
DT_FAST_REQUIRE_UNARY_FALSE = is_fast | is_false | is_unary | is_require
};
} 

DOCTEST_INTERFACE const char* getAssertString(assertType::Enum val);

template<class T>               struct decay_array       { typedef T type; };
template<class T, unsigned N>   struct decay_array<T[N]> { typedef T* type; };
template<class T>               struct decay_array<T[]>  { typedef T* type; };

template<class T>   struct not_char_pointer              { enum { value = 1 }; };
template<>          struct not_char_pointer<char*>       { enum { value = 0 }; };
template<>          struct not_char_pointer<const char*> { enum { value = 0 }; };

template<class T> struct can_use_op : not_char_pointer<typename decay_array<T>::type> {};

struct TestFailureException
{};

DOCTEST_INTERFACE bool checkIfShouldThrow(assertType::Enum assert_type);
DOCTEST_INTERFACE void fastAssertThrowIfFlagSet(int flags);
DOCTEST_INTERFACE void throwException();

struct TestAccessibleContextState
{
bool no_throw; 
bool success;  
};

struct ContextState;

DOCTEST_INTERFACE TestAccessibleContextState* getTestsContextState();

struct DOCTEST_INTERFACE SubcaseSignature
{
const char* m_name;
const char* m_file;
int         m_line;

SubcaseSignature(const char* name, const char* file, int line)
: m_name(name)
, m_file(file)
, m_line(line) {}

bool operator<(const SubcaseSignature& other) const;
};

struct DOCTEST_INTERFACE Subcase
{
SubcaseSignature m_signature;
bool             m_entered;

Subcase(const char* name, const char* file, int line);
Subcase(const Subcase& other);
~Subcase();

operator bool() const { return m_entered; }
};

template <typename L, typename R>
String stringifyBinaryExpr(const DOCTEST_REF_WRAP(L) lhs, const char* op,
const DOCTEST_REF_WRAP(R) rhs) {
return toString(lhs) + op + toString(rhs);
}

struct DOCTEST_INTERFACE Result
{
bool   m_passed;
String m_decomposition;

~Result();

DOCTEST_NOINLINE Result(bool passed = false, const String& decomposition = String())
: m_passed(passed)
, m_decomposition(decomposition) {}

DOCTEST_NOINLINE Result(const Result& other)
: m_passed(other.m_passed)
, m_decomposition(other.m_decomposition) {}

Result& operator=(const Result& other);

operator bool() { return !m_passed; }

template <typename R> Result& operator&  (const R&) { DOCTEST_STATIC_ASSERT(deferred_false<R>::value, Expression_Too_Complex_Please_Rewrite_As_Binary_Comparison); return *this; }
template <typename R> Result& operator^  (const R&) { DOCTEST_STATIC_ASSERT(deferred_false<R>::value, Expression_Too_Complex_Please_Rewrite_As_Binary_Comparison); return *this; }
template <typename R> Result& operator|  (const R&) { DOCTEST_STATIC_ASSERT(deferred_false<R>::value, Expression_Too_Complex_Please_Rewrite_As_Binary_Comparison); return *this; }
template <typename R> Result& operator&& (const R&) { DOCTEST_STATIC_ASSERT(deferred_false<R>::value, Expression_Too_Complex_Please_Rewrite_As_Binary_Comparison); return *this; }
template <typename R> Result& operator|| (const R&) { DOCTEST_STATIC_ASSERT(deferred_false<R>::value, Expression_Too_Complex_Please_Rewrite_As_Binary_Comparison); return *this; }
template <typename R> Result& operator== (const R&) { DOCTEST_STATIC_ASSERT(deferred_false<R>::value, Expression_Too_Complex_Please_Rewrite_As_Binary_Comparison); return *this; }
template <typename R> Result& operator!= (const R&) { DOCTEST_STATIC_ASSERT(deferred_false<R>::value, Expression_Too_Complex_Please_Rewrite_As_Binary_Comparison); return *this; }
template <typename R> Result& operator<  (const R&) { DOCTEST_STATIC_ASSERT(deferred_false<R>::value, Expression_Too_Complex_Please_Rewrite_As_Binary_Comparison); return *this; }
template <typename R> Result& operator>  (const R&) { DOCTEST_STATIC_ASSERT(deferred_false<R>::value, Expression_Too_Complex_Please_Rewrite_As_Binary_Comparison); return *this; }
template <typename R> Result& operator<= (const R&) { DOCTEST_STATIC_ASSERT(deferred_false<R>::value, Expression_Too_Complex_Please_Rewrite_As_Binary_Comparison); return *this; }
template <typename R> Result& operator>= (const R&) { DOCTEST_STATIC_ASSERT(deferred_false<R>::value, Expression_Too_Complex_Please_Rewrite_As_Binary_Comparison); return *this; }
template <typename R> Result& operator=  (const R&) { DOCTEST_STATIC_ASSERT(deferred_false<R>::value, Expression_Too_Complex_Please_Rewrite_As_Binary_Comparison); return *this; }
template <typename R> Result& operator+= (const R&) { DOCTEST_STATIC_ASSERT(deferred_false<R>::value, Expression_Too_Complex_Please_Rewrite_As_Binary_Comparison); return *this; }
template <typename R> Result& operator-= (const R&) { DOCTEST_STATIC_ASSERT(deferred_false<R>::value, Expression_Too_Complex_Please_Rewrite_As_Binary_Comparison); return *this; }
template <typename R> Result& operator*= (const R&) { DOCTEST_STATIC_ASSERT(deferred_false<R>::value, Expression_Too_Complex_Please_Rewrite_As_Binary_Comparison); return *this; }
template <typename R> Result& operator/= (const R&) { DOCTEST_STATIC_ASSERT(deferred_false<R>::value, Expression_Too_Complex_Please_Rewrite_As_Binary_Comparison); return *this; }
template <typename R> Result& operator%= (const R&) { DOCTEST_STATIC_ASSERT(deferred_false<R>::value, Expression_Too_Complex_Please_Rewrite_As_Binary_Comparison); return *this; }
template <typename R> Result& operator<<=(const R&) { DOCTEST_STATIC_ASSERT(deferred_false<R>::value, Expression_Too_Complex_Please_Rewrite_As_Binary_Comparison); return *this; }
template <typename R> Result& operator>>=(const R&) { DOCTEST_STATIC_ASSERT(deferred_false<R>::value, Expression_Too_Complex_Please_Rewrite_As_Binary_Comparison); return *this; }
template <typename R> Result& operator&= (const R&) { DOCTEST_STATIC_ASSERT(deferred_false<R>::value, Expression_Too_Complex_Please_Rewrite_As_Binary_Comparison); return *this; }
template <typename R> Result& operator^= (const R&) { DOCTEST_STATIC_ASSERT(deferred_false<R>::value, Expression_Too_Complex_Please_Rewrite_As_Binary_Comparison); return *this; }
template <typename R> Result& operator|= (const R&) { DOCTEST_STATIC_ASSERT(deferred_false<R>::value, Expression_Too_Complex_Please_Rewrite_As_Binary_Comparison); return *this; }
};

#ifndef DOCTEST_CONFIG_NO_COMPARISON_WARNING_SUPPRESSION

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wsign-conversion"
#pragma clang diagnostic ignored "-Wsign-compare"
#endif 

#if defined(__GNUC__) && !defined(__clang__)
#if __GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ > 6)
#pragma GCC diagnostic push
#endif 
#pragma GCC diagnostic ignored "-Wsign-conversion"
#pragma GCC diagnostic ignored "-Wsign-compare"
#if __GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ > 5)
#endif 
#endif 

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4389) 
#pragma warning(disable : 4018) 
#endif 

#endif 

#ifndef DOCTEST_CONFIG_TREAT_CHAR_STAR_AS_STRING
#define DOCTEST_COMPARISON_RETURN_TYPE bool
#else 
#define DOCTEST_COMPARISON_RETURN_TYPE typename traits::enable_if<can_use_op<L>::value || can_use_op<R>::value, bool>::type
inline bool eq(const char* lhs, const char* rhs) { return String(lhs) == String(rhs); }
inline bool ne(const char* lhs, const char* rhs) { return String(lhs) != String(rhs); }
inline bool lt(const char* lhs, const char* rhs) { return String(lhs) <  String(rhs); }
inline bool gt(const char* lhs, const char* rhs) { return String(lhs) >  String(rhs); }
inline bool le(const char* lhs, const char* rhs) { return String(lhs) <= String(rhs); }
inline bool ge(const char* lhs, const char* rhs) { return String(lhs) >= String(rhs); }
#endif 

template <typename L, typename R> DOCTEST_COMPARISON_RETURN_TYPE eq(const DOCTEST_REF_WRAP(L) lhs, const DOCTEST_REF_WRAP(R) rhs) { return lhs == rhs; }
template <typename L, typename R> DOCTEST_COMPARISON_RETURN_TYPE ne(const DOCTEST_REF_WRAP(L) lhs, const DOCTEST_REF_WRAP(R) rhs) { return lhs != rhs; }
template <typename L, typename R> DOCTEST_COMPARISON_RETURN_TYPE lt(const DOCTEST_REF_WRAP(L) lhs, const DOCTEST_REF_WRAP(R) rhs) { return lhs <  rhs; }
template <typename L, typename R> DOCTEST_COMPARISON_RETURN_TYPE gt(const DOCTEST_REF_WRAP(L) lhs, const DOCTEST_REF_WRAP(R) rhs) { return lhs >  rhs; }
template <typename L, typename R> DOCTEST_COMPARISON_RETURN_TYPE le(const DOCTEST_REF_WRAP(L) lhs, const DOCTEST_REF_WRAP(R) rhs) { return lhs <= rhs; }
template <typename L, typename R> DOCTEST_COMPARISON_RETURN_TYPE ge(const DOCTEST_REF_WRAP(L) lhs, const DOCTEST_REF_WRAP(R) rhs) { return lhs >= rhs; }

#ifndef DOCTEST_CONFIG_TREAT_CHAR_STAR_AS_STRING
#define DOCTEST_CMP_EQ(l, r) l == r
#define DOCTEST_CMP_NE(l, r) l != r
#define DOCTEST_CMP_GT(l, r) l > r
#define DOCTEST_CMP_LT(l, r) l < r
#define DOCTEST_CMP_GE(l, r) l >= r
#define DOCTEST_CMP_LE(l, r) l <= r
#else 
#define DOCTEST_CMP_EQ(l, r) eq(l, r)
#define DOCTEST_CMP_NE(l, r) ne(l, r)
#define DOCTEST_CMP_GT(l, r) gt(l, r)
#define DOCTEST_CMP_LT(l, r) lt(l, r)
#define DOCTEST_CMP_GE(l, r) ge(l, r)
#define DOCTEST_CMP_LE(l, r) le(l, r)
#endif 

#define DOCTEST_DO_BINARY_EXPRESSION_COMPARISON(op, op_str, op_macro)                              \
template <typename R>                                                                          \
DOCTEST_NOINLINE Result operator op(const DOCTEST_REF_WRAP(R) rhs) {                           \
bool res = op_macro(lhs, rhs);                                                             \
if(m_assert_type & assertType::is_false)                                                   \
res = !res;                                                                            \
if(!res || doctest::detail::getTestsContextState()->success)                               \
return Result(res, stringifyBinaryExpr(lhs, op_str, rhs));                             \
return Result(res);                                                                        \
}

#define DOCTEST_FORBIT_EXPRESSION(op)                                                              \
template <typename R>                                                                          \
Expression_lhs& operator op(const R&) {                                                        \
DOCTEST_STATIC_ASSERT(deferred_false<R>::value,                                            \
Expression_Too_Complex_Please_Rewrite_As_Binary_Comparison);         \
return *this;                                                                              \
}

template <typename L>
struct Expression_lhs
{
L                lhs;
assertType::Enum m_assert_type;

explicit Expression_lhs(L in, assertType::Enum assert_type)
: lhs(in)
, m_assert_type(assert_type) {}

Expression_lhs(const Expression_lhs& other)
: lhs(other.lhs) {}

DOCTEST_NOINLINE operator Result() {
bool res = !!lhs;
if(m_assert_type & assertType::is_false) 
res = !res;

if(!res || getTestsContextState()->success)
return Result(res, toString(lhs));
return Result(res);
}

DOCTEST_DO_BINARY_EXPRESSION_COMPARISON(==, " == ", DOCTEST_CMP_EQ) 
DOCTEST_DO_BINARY_EXPRESSION_COMPARISON(!=, " != ", DOCTEST_CMP_NE) 
DOCTEST_DO_BINARY_EXPRESSION_COMPARISON(>, " >  ", DOCTEST_CMP_GT) 
DOCTEST_DO_BINARY_EXPRESSION_COMPARISON(<, " <  ", DOCTEST_CMP_LT) 
DOCTEST_DO_BINARY_EXPRESSION_COMPARISON(>=, " >= ", DOCTEST_CMP_GE) 
DOCTEST_DO_BINARY_EXPRESSION_COMPARISON(<=, " <= ", DOCTEST_CMP_LE) 

DOCTEST_FORBIT_EXPRESSION(&)
DOCTEST_FORBIT_EXPRESSION (^)
DOCTEST_FORBIT_EXPRESSION(|)
DOCTEST_FORBIT_EXPRESSION(&&)
DOCTEST_FORBIT_EXPRESSION(||)
DOCTEST_FORBIT_EXPRESSION(=)
DOCTEST_FORBIT_EXPRESSION(+=)
DOCTEST_FORBIT_EXPRESSION(-=)
DOCTEST_FORBIT_EXPRESSION(*=)
DOCTEST_FORBIT_EXPRESSION(/=)
DOCTEST_FORBIT_EXPRESSION(%=)
DOCTEST_FORBIT_EXPRESSION(<<=)
DOCTEST_FORBIT_EXPRESSION(>>=)
DOCTEST_FORBIT_EXPRESSION(&=)
DOCTEST_FORBIT_EXPRESSION(^=)
DOCTEST_FORBIT_EXPRESSION(|=)
DOCTEST_FORBIT_EXPRESSION(<<)
DOCTEST_FORBIT_EXPRESSION(>>)
};

#ifndef DOCTEST_CONFIG_NO_COMPARISON_WARNING_SUPPRESSION

#if defined(__clang__)
#pragma clang diagnostic pop
#endif 

#if defined(__GNUC__) && !defined(__clang__)
#if __GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ > 6)
#pragma GCC diagnostic pop
#endif 
#endif 

#ifdef _MSC_VER
#pragma warning(pop)
#endif 

#endif 

struct ExpressionDecomposer
{
assertType::Enum m_assert_type;

ExpressionDecomposer(assertType::Enum assert_type)
: m_assert_type(assert_type) {}

template <typename L>
Expression_lhs<const DOCTEST_REF_WRAP(L)> operator<<(const DOCTEST_REF_WRAP(L) operand) {
return Expression_lhs<const DOCTEST_REF_WRAP(L)>(operand, m_assert_type);
}
};

struct DOCTEST_INTERFACE TestCase
{
funcType m_test;    
String m_full_name; 
const char* m_name;       
const char* m_type;       
const char* m_test_suite; 
const char* m_description;
bool        m_skip;
bool        m_may_fail;
bool        m_should_fail;
int         m_expected_failures;
double      m_timeout;

const char* m_file; 
unsigned    m_line; 
int m_template_id; 

TestCase(funcType test, const char* file, unsigned line, const TestSuite& test_suite,
const char* type = "", int template_id = -1);

DOCTEST_NOINLINE ~TestCase() {}

TestCase& operator*(const char* in);

template <typename T>
TestCase& operator*(const T& in) {
in.fill(*this);
return *this;
}

TestCase(const TestCase& other) { *this = other; }

TestCase& operator=(const TestCase& other);

bool operator<(const TestCase& other) const;
};

DOCTEST_INTERFACE int regTest(const TestCase& tc);
DOCTEST_INTERFACE int setTestSuite(const TestSuite& ts);

DOCTEST_INTERFACE void addFailedAssert(assertType::Enum assert_type);

DOCTEST_INTERFACE void logTestStart(const TestCase& tc);
DOCTEST_INTERFACE void logTestEnd();

DOCTEST_INTERFACE void logTestException(const String& what, bool crash = false);

DOCTEST_INTERFACE void logAssert(bool passed, const char* decomposition, bool threw,
const String& exception, const char* expr,
assertType::Enum assert_type, const char* file, int line);

DOCTEST_INTERFACE void logAssertThrows(bool threw, const char* expr,
assertType::Enum assert_type, const char* file,
int line);

DOCTEST_INTERFACE void logAssertThrowsAs(bool threw, bool threw_as, const char* as,
const String& exception, const char* expr,
assertType::Enum assert_type, const char* file,
int line);

DOCTEST_INTERFACE void logAssertNothrow(bool threw, const String& exception, const char* expr,
assertType::Enum assert_type, const char* file,
int line);

DOCTEST_INTERFACE bool isDebuggerActive();
DOCTEST_INTERFACE void writeToDebugConsole(const String&);

namespace binaryAssertComparison
{
enum Enum
{
eq = 0,
ne,
gt,
lt,
ge,
le
};
} 

template <int, class L, class R> struct RelationalComparator     { bool operator()(const DOCTEST_REF_WRAP(L),     const DOCTEST_REF_WRAP(R)    ) const { return false;        } };
template <class L, class R> struct RelationalComparator<0, L, R> { bool operator()(const DOCTEST_REF_WRAP(L) lhs, const DOCTEST_REF_WRAP(R) rhs) const { return eq(lhs, rhs); } };
template <class L, class R> struct RelationalComparator<1, L, R> { bool operator()(const DOCTEST_REF_WRAP(L) lhs, const DOCTEST_REF_WRAP(R) rhs) const { return ne(lhs, rhs); } };
template <class L, class R> struct RelationalComparator<2, L, R> { bool operator()(const DOCTEST_REF_WRAP(L) lhs, const DOCTEST_REF_WRAP(R) rhs) const { return gt(lhs, rhs); } };
template <class L, class R> struct RelationalComparator<3, L, R> { bool operator()(const DOCTEST_REF_WRAP(L) lhs, const DOCTEST_REF_WRAP(R) rhs) const { return lt(lhs, rhs); } };
template <class L, class R> struct RelationalComparator<4, L, R> { bool operator()(const DOCTEST_REF_WRAP(L) lhs, const DOCTEST_REF_WRAP(R) rhs) const { return ge(lhs, rhs); } };
template <class L, class R> struct RelationalComparator<5, L, R> { bool operator()(const DOCTEST_REF_WRAP(L) lhs, const DOCTEST_REF_WRAP(R) rhs) const { return le(lhs, rhs); } };

struct DOCTEST_INTERFACE ResultBuilder
{
assertType::Enum m_assert_type;
const char*      m_file;
int              m_line;
const char*      m_expr;
const char*      m_exception_type;

Result m_result;
bool   m_threw;
bool   m_threw_as;
bool   m_failed;
String m_exception;

ResultBuilder(assertType::Enum assert_type, const char* file, int line, const char* expr,
const char* exception_type = "");

~ResultBuilder();

void setResult(const Result& res) { m_result = res; }

template <int         comparison, typename L, typename R>
DOCTEST_NOINLINE void binary_assert(const DOCTEST_REF_WRAP(L) lhs,
const DOCTEST_REF_WRAP(R) rhs) {
m_result.m_passed = RelationalComparator<comparison, L, R>()(lhs, rhs);
if(!m_result.m_passed || getTestsContextState()->success)
m_result.m_decomposition = stringifyBinaryExpr(lhs, ", ", rhs);
}

template <typename L>
DOCTEST_NOINLINE void unary_assert(const DOCTEST_REF_WRAP(L) val) {
m_result.m_passed = !!val;

if(m_assert_type & assertType::is_false) 
m_result.m_passed = !m_result.m_passed;

if(!m_result.m_passed || getTestsContextState()->success)
m_result.m_decomposition = toString(val);
}

void unexpectedExceptionOccurred();

bool log();
void react() const;
};

namespace assertAction
{
enum Enum
{
nothing     = 0,
dbgbreak    = 1,
shouldthrow = 2
};
} 

template <int        comparison, typename L, typename R>
DOCTEST_NOINLINE int fast_binary_assert(assertType::Enum assert_type, const char* file,
int line, const char* expr,
const DOCTEST_REF_WRAP(L) lhs,
const DOCTEST_REF_WRAP(R) rhs) {
ResultBuilder rb(assert_type, file, line, expr);

rb.m_result.m_passed = RelationalComparator<comparison, L, R>()(lhs, rhs);

if(!rb.m_result.m_passed || getTestsContextState()->success)
rb.m_result.m_decomposition = stringifyBinaryExpr(lhs, ", ", rhs);

int res = 0;

if(rb.log())
res |= assertAction::dbgbreak;

if(rb.m_failed && checkIfShouldThrow(assert_type))
res |= assertAction::shouldthrow;

#ifdef DOCTEST_CONFIG_SUPER_FAST_ASSERTS
if(res & assertAction::dbgbreak)
DOCTEST_BREAK_INTO_DEBUGGER();
fastAssertThrowIfFlagSet(res);
#endif 

return res;
}

template <typename L>
DOCTEST_NOINLINE int fast_unary_assert(assertType::Enum assert_type, const char* file, int line,
const char* val_str, const DOCTEST_REF_WRAP(L) val) {
ResultBuilder rb(assert_type, file, line, val_str);

rb.m_result.m_passed = !!val;

if(assert_type & assertType::is_false) 
rb.m_result.m_passed = !rb.m_result.m_passed;

if(!rb.m_result.m_passed || getTestsContextState()->success)
rb.m_result.m_decomposition = toString(val);

int res = 0;

if(rb.log())
res |= assertAction::dbgbreak;

if(rb.m_failed && checkIfShouldThrow(assert_type))
res |= assertAction::shouldthrow;

#ifdef DOCTEST_CONFIG_SUPER_FAST_ASSERTS
if(res & assertAction::dbgbreak)
DOCTEST_BREAK_INTO_DEBUGGER();
fastAssertThrowIfFlagSet(res);
#endif 

return res;
}

struct DOCTEST_INTERFACE IExceptionTranslator 
{
virtual ~IExceptionTranslator();
virtual bool translate(String&) const = 0;
};

template <typename T>
class ExceptionTranslator : public IExceptionTranslator 
{
public:
explicit ExceptionTranslator(String (*translateFunction)(T))
: m_translateFunction(translateFunction) {}

bool translate(String& res) const {
#ifndef DOCTEST_CONFIG_NO_EXCEPTIONS
try {
throw;
} catch(T ex) {                    
res = m_translateFunction(ex); 
return true;
} catch(...) {} 
#endif                      
((void)res);    
return false;
}

protected:
String (*m_translateFunction)(T);
};

DOCTEST_INTERFACE void registerExceptionTranslatorImpl(
const IExceptionTranslator* translateFunction);

DOCTEST_INTERFACE void writeStringToStream(std::ostream* stream, const String& str);

template <bool C>
struct StringStreamBase
{
template <typename T>
static void convert(std::ostream* stream, const T& in) {
writeStringToStream(stream, toString(in));
}

static void convert(std::ostream* stream, const char* in) {
writeStringToStream(stream, String(in));
}
};

template <>
struct StringStreamBase<true>
{
template <typename T>
static void convert(std::ostream* stream, const T& in) {
*stream << in;
}
};

template <typename T>
struct StringStream : StringStreamBase<has_insertion_operator<T>::value>
{};

template <typename T>
void toStream(std::ostream* stream, const T& value) {
StringStream<T>::convert(stream, value);
}

#ifdef DOCTEST_CONFIG_TREAT_CHAR_STAR_AS_STRING
DOCTEST_INTERFACE void toStream(std::ostream* stream, char* in);
DOCTEST_INTERFACE void toStream(std::ostream* stream, const char* in);
#endif 
DOCTEST_INTERFACE void toStream(std::ostream* stream, bool in);
DOCTEST_INTERFACE void toStream(std::ostream* stream, float in);
DOCTEST_INTERFACE void toStream(std::ostream* stream, double in);
DOCTEST_INTERFACE void toStream(std::ostream* stream, double long in);

DOCTEST_INTERFACE void toStream(std::ostream* stream, char in);
DOCTEST_INTERFACE void toStream(std::ostream* stream, char signed in);
DOCTEST_INTERFACE void toStream(std::ostream* stream, char unsigned in);
DOCTEST_INTERFACE void toStream(std::ostream* stream, int short in);
DOCTEST_INTERFACE void toStream(std::ostream* stream, int short unsigned in);
DOCTEST_INTERFACE void toStream(std::ostream* stream, int in);
DOCTEST_INTERFACE void toStream(std::ostream* stream, int unsigned in);
DOCTEST_INTERFACE void toStream(std::ostream* stream, int long in);
DOCTEST_INTERFACE void toStream(std::ostream* stream, int long unsigned in);

#ifdef DOCTEST_CONFIG_WITH_LONG_LONG
DOCTEST_INTERFACE void toStream(std::ostream* stream, int long long in);
DOCTEST_INTERFACE void toStream(std::ostream* stream, int long long unsigned in);
#endif 

struct IContextScope 
{ virtual void build(std::ostream*) = 0; };

DOCTEST_INTERFACE void addToContexts(IContextScope* ptr);
DOCTEST_INTERFACE void popFromContexts();
DOCTEST_INTERFACE void useContextIfExceptionOccurred(IContextScope* ptr);

class ContextBuilder
{
friend class ContextScope;

struct ICapture 
{ virtual void toStream(std::ostream*) const = 0; };

template <typename T>
struct Capture : ICapture 
{
const T* capture;

explicit Capture(const T* in)
: capture(in) {}
virtual void toStream(std::ostream* stream) const { 
doctest::detail::toStream(stream, *capture);
}
};

struct Chunk
{
char buf[sizeof(Capture<char>)]; 
};

struct Node
{
Chunk chunk;
Node* next;
};

Chunk stackChunks[DOCTEST_CONFIG_NUM_CAPTURES_ON_STACK];
int   numCaptures;
Node* head;
Node* tail;

void build(std::ostream* stream) const {
int curr = 0;
while(curr < numCaptures && curr < DOCTEST_CONFIG_NUM_CAPTURES_ON_STACK)
reinterpret_cast<const ICapture*>(stackChunks[curr++].buf)->toStream(stream);
Node* curr_elem = head;
while(curr < numCaptures) {
reinterpret_cast<const ICapture*>(curr_elem->chunk.buf)->toStream(stream);
curr_elem = curr_elem->next;
++curr;
}
}

DOCTEST_NOINLINE ContextBuilder(ContextBuilder& other)
: numCaptures(other.numCaptures)
, head(other.head)
, tail(other.tail) {
other.numCaptures = 0;
other.head        = 0;
other.tail        = 0;
my_memcpy(stackChunks, other.stackChunks,
unsigned(int(sizeof(Chunk)) * DOCTEST_CONFIG_NUM_CAPTURES_ON_STACK));
}

public:
DOCTEST_NOINLINE ContextBuilder() 
: numCaptures(0)
, head(0)
, tail(0) {}

template <typename T>
DOCTEST_NOINLINE ContextBuilder& operator<<(T& in) {
Capture<T> temp(&in);

if(numCaptures < DOCTEST_CONFIG_NUM_CAPTURES_ON_STACK) {
my_memcpy(stackChunks[numCaptures].buf, &temp, sizeof(Chunk));
} else {
Node* curr = new Node;
curr->next = 0;
if(tail) {
tail->next = curr;
tail       = curr;
} else {
head = tail = curr;
}

my_memcpy(tail->chunk.buf, &temp, sizeof(Chunk));
}
++numCaptures;
return *this;
}

DOCTEST_NOINLINE ~ContextBuilder() {
while(head) {
Node* next = head->next;
delete head;
head = next;
}
}

#ifdef DOCTEST_CONFIG_WITH_RVALUE_REFERENCES
template <typename T>
ContextBuilder& operator<<(const T&&) {
DOCTEST_STATIC_ASSERT(
deferred_false<T>::value,
Cannot_pass_temporaries_or_rvalues_to_the_streaming_operator_because_it_caches_pointers_to_the_passed_objects_for_lazy_evaluation);
return *this;
}
#endif 
};

class ContextScope : public IContextScope 
{
ContextBuilder contextBuilder;
bool           built;

public:
DOCTEST_NOINLINE explicit ContextScope(ContextBuilder& temp)
: contextBuilder(temp)
, built(false) {
addToContexts(this);
}

DOCTEST_NOINLINE ~ContextScope() {
if(!built)
useContextIfExceptionOccurred(this);
popFromContexts();
}

void build(std::ostream* stream) {
built = true;
contextBuilder.build(stream);
}
};

class DOCTEST_INTERFACE MessageBuilder
{
std::ostream*                     m_stream;
const char*                       m_file;
int                               m_line;
doctest::detail::assertType::Enum m_severity;

public:
MessageBuilder(const char* file, int line, doctest::detail::assertType::Enum severity);
~MessageBuilder();

template <typename T>
MessageBuilder& operator<<(const T& in) {
doctest::detail::toStream(m_stream, in);
return *this;
}

bool log();
void react();
};
} 

struct test_suite
{
const char* data;
test_suite(const char* in)
: data(in) {}
void fill(detail::TestCase& state) const { state.m_test_suite = data; }
void fill(detail::TestSuite& state) const { state.m_test_suite = data; }
};

struct description
{
const char* data;
description(const char* in)
: data(in) {}
void fill(detail::TestCase& state) const { state.m_description = data; }
void fill(detail::TestSuite& state) const { state.m_description = data; }
};

struct skip
{
bool data;
skip(bool in = true)
: data(in) {}
void fill(detail::TestCase& state) const { state.m_skip = data; }
void fill(detail::TestSuite& state) const { state.m_skip = data; }
};

struct timeout
{
double data;
timeout(double in)
: data(in) {}
void fill(detail::TestCase& state) const { state.m_timeout = data; }
void fill(detail::TestSuite& state) const { state.m_timeout = data; }
};

struct may_fail
{
bool data;
may_fail(bool in = true)
: data(in) {}
void fill(detail::TestCase& state) const { state.m_may_fail = data; }
void fill(detail::TestSuite& state) const { state.m_may_fail = data; }
};

struct should_fail
{
bool data;
should_fail(bool in = true)
: data(in) {}
void fill(detail::TestCase& state) const { state.m_should_fail = data; }
void fill(detail::TestSuite& state) const { state.m_should_fail = data; }
};

struct expected_failures
{
int data;
expected_failures(int in)
: data(in) {}
void fill(detail::TestCase& state) const { state.m_expected_failures = data; }
void fill(detail::TestSuite& state) const { state.m_expected_failures = data; }
};

#endif 

#ifndef DOCTEST_CONFIG_DISABLE
template <typename T>
int registerExceptionTranslator(String (*translateFunction)(T)) {
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wexit-time-destructors"
#endif 
static detail::ExceptionTranslator<T> exceptionTranslator(translateFunction);
#if defined(__clang__)
#pragma clang diagnostic pop
#endif 
detail::registerExceptionTranslatorImpl(&exceptionTranslator);
return 0;
}

#else  
template <typename T>
int registerExceptionTranslator(String (*)(T)) {
return 0;
}
#endif 

DOCTEST_INTERFACE bool isRunningInTest();

class DOCTEST_INTERFACE Context
{
#if !defined(DOCTEST_CONFIG_DISABLE)
detail::ContextState* p;

void parseArgs(int argc, const char* const* argv, bool withDefaults = false);

#endif 

public:
explicit Context(int argc = 0, const char* const* argv = 0);

~Context();

void applyCommandLine(int argc, const char* const* argv);

void addFilter(const char* filter, const char* value);
void clearFilters();
void setOption(const char* option, int value);
void setOption(const char* option, const char* value);

bool shouldExit();

int run();
};

} 

#if !defined(DOCTEST_CONFIG_DISABLE)

#ifdef DOCTEST_CONFIG_WITH_VARIADIC_MACROS
#define DOCTEST_EXPAND_VA_ARGS(...) __VA_ARGS__
#else 
#define DOCTEST_EXPAND_VA_ARGS
#endif 

#define DOCTEST_STRIP_PARENS(x) x
#define DOCTEST_HANDLE_BRACED_VA_ARGS(expr) DOCTEST_STRIP_PARENS(DOCTEST_EXPAND_VA_ARGS expr)

#define DOCTEST_REGISTER_FUNCTION(f, decorators)                                                   \
DOCTEST_GLOBAL_NO_WARNINGS(DOCTEST_ANONYMOUS(_DOCTEST_ANON_VAR_)) = doctest::detail::regTest(  \
doctest::detail::TestCase(f, __FILE__, __LINE__,                                       \
doctest_detail_test_suite_ns::getCurrentTestSuite()) *       \
decorators);                                                                           \
DOCTEST_GLOBAL_NO_WARNINGS_END()

#define DOCTEST_IMPLEMENT_FIXTURE(der, base, func, decorators)                                     \
namespace                                                                                      \
{                                                                                              \
struct der : base                                                                          \
{ void f(); };                                                                             \
static void func() {                                                                       \
der v;                                                                                 \
v.f();                                                                                 \
}                                                                                          \
DOCTEST_REGISTER_FUNCTION(func, decorators)                                                \
}                                                                                              \
inline DOCTEST_NOINLINE void der::f()

#define DOCTEST_CREATE_AND_REGISTER_FUNCTION(f, decorators)                                        \
static void f();                                                                               \
DOCTEST_REGISTER_FUNCTION(f, decorators)                                                       \
static void f()

#define DOCTEST_TEST_CASE(decorators)                                                              \
DOCTEST_CREATE_AND_REGISTER_FUNCTION(DOCTEST_ANONYMOUS(_DOCTEST_ANON_FUNC_), decorators)

#define DOCTEST_TEST_CASE_FIXTURE(c, decorators)                                                   \
DOCTEST_IMPLEMENT_FIXTURE(DOCTEST_ANONYMOUS(_DOCTEST_ANON_CLASS_), c,                          \
DOCTEST_ANONYMOUS(_DOCTEST_ANON_FUNC_), decorators)

#ifdef DOCTEST_CONFIG_WITH_VARIADIC_MACROS
#define DOCTEST_TYPE_TO_STRING_IMPL(...)                                                           \
template <>                                                                                    \
inline const char* type_to_string<__VA_ARGS__>() {                                             \
return "<" #__VA_ARGS__ ">";                                                               \
}
#define DOCTEST_TYPE_TO_STRING(...)                                                                \
namespace doctest                                                                              \
{                                                                                              \
namespace detail                                                                           \
{ DOCTEST_TYPE_TO_STRING_IMPL(__VA_ARGS__) }                                               \
}                                                                                              \
typedef int DOCTEST_ANONYMOUS(_DOCTEST_ANON_FOR_SEMICOLON_)
#else 
#define DOCTEST_TYPE_TO_STRING_IMPL(x)                                                             \
template <>                                                                                    \
inline const char* type_to_string<x>() {                                                       \
return "<" #x ">";                                                                         \
}
#define DOCTEST_TYPE_TO_STRING(x)                                                                  \
namespace doctest                                                                              \
{                                                                                              \
namespace detail                                                                           \
{ DOCTEST_TYPE_TO_STRING_IMPL(x) }                                                         \
}                                                                                              \
typedef int DOCTEST_ANONYMOUS(_DOCTEST_ANON_FOR_SEMICOLON_)
#endif 

#define DOCTEST_TEST_CASE_TEMPLATE_IMPL(decorators, T, types, anon)                                \
template <typename T>                                                                          \
inline void anon();                                                                            \
struct DOCTEST_CAT(anon, FUNCTOR)                                                              \
{                                                                                              \
template <int Index, typename Type>                                                        \
void          operator()() {                                                               \
doctest::detail::regTest(                                                              \
doctest::detail::TestCase(anon<Type>, __FILE__, __LINE__,                      \
doctest_detail_test_suite_ns::getCurrentTestSuite(), \
doctest::detail::type_to_string<Type>(), Index) *    \
decorators);                                                                   \
}                                                                                          \
};                                                                                             \
inline int DOCTEST_CAT(anon, REG_FUNC)() {                                                     \
DOCTEST_CAT(anon, FUNCTOR) registrar;                                                      \
doctest::detail::ForEachType<DOCTEST_HANDLE_BRACED_VA_ARGS(types)::Result,                 \
DOCTEST_CAT(anon, FUNCTOR)>                                   \
doIt(registrar);                                                                   \
return 0;                                                                                  \
}                                                                                              \
DOCTEST_GLOBAL_NO_WARNINGS(DOCTEST_CAT(anon, DUMMY)) = DOCTEST_CAT(anon, REG_FUNC)();          \
DOCTEST_GLOBAL_NO_WARNINGS_END()                                                               \
template <typename T>                                                                          \
inline void anon()

#ifdef DOCTEST_CONFIG_WITH_VARIADIC_MACROS
#define DOCTEST_TEST_CASE_TEMPLATE(decorators, T, ...)                                             \
DOCTEST_TEST_CASE_TEMPLATE_IMPL(decorators, T, (__VA_ARGS__),                                  \
DOCTEST_ANONYMOUS(_DOCTEST_ANON_TMP_))
#else 
#define DOCTEST_TEST_CASE_TEMPLATE(decorators, T, types)                                           \
DOCTEST_TEST_CASE_TEMPLATE_IMPL(decorators, T, types, DOCTEST_ANONYMOUS(_DOCTEST_ANON_TMP_))
#endif 

#define DOCTEST_TEST_CASE_TEMPLATE_DEFINE_IMPL(decorators, T, id, anon)                            \
template <typename T>                                                                          \
inline void anon();                                                                            \
struct DOCTEST_CAT(id, _FUNCTOR)                                                               \
{                                                                                              \
int m_line;                                                                                \
DOCTEST_CAT(id, _FUNCTOR)                                                                  \
(int line)                                                                                 \
: m_line(line) {}                                                                  \
template <int Index, typename Type>                                                        \
void          operator()() {                                                               \
doctest::detail::regTest(                                                              \
doctest::detail::TestCase(anon<Type>, __FILE__, __LINE__,                      \
doctest_detail_test_suite_ns::getCurrentTestSuite(), \
doctest::detail::type_to_string<Type>(),             \
m_line * 1000 + Index) *                             \
decorators);                                                                   \
}                                                                                          \
};                                                                                             \
template <typename T>                                                                          \
inline void anon()

#define DOCTEST_TEST_CASE_TEMPLATE_DEFINE(decorators, T, id)                                       \
DOCTEST_TEST_CASE_TEMPLATE_DEFINE_IMPL(decorators, T, id, DOCTEST_ANONYMOUS(_DOCTEST_ANON_TMP_))

#define DOCTEST_TEST_CASE_TEMPLATE_INSTANTIATE_IMPL(id, types, anon)                               \
static int DOCTEST_CAT(anon, REG_FUNC)() {                                                     \
DOCTEST_CAT(id, _FUNCTOR) registrar(__LINE__);                                             \
doctest::detail::ForEachType<DOCTEST_HANDLE_BRACED_VA_ARGS(types)::Result,                 \
DOCTEST_CAT(id, _FUNCTOR)>                                    \
doIt(registrar);                                                                   \
return 0;                                                                                  \
}                                                                                              \
DOCTEST_GLOBAL_NO_WARNINGS(DOCTEST_CAT(anon, DUMMY)) = DOCTEST_CAT(anon, REG_FUNC)();          \
DOCTEST_GLOBAL_NO_WARNINGS_END() typedef int DOCTEST_ANONYMOUS(_DOCTEST_ANON_FOR_SEMICOLON_)

#ifdef DOCTEST_CONFIG_WITH_VARIADIC_MACROS
#define DOCTEST_TEST_CASE_TEMPLATE_INSTANTIATE(id, ...)                                            \
DOCTEST_TEST_CASE_TEMPLATE_INSTANTIATE_IMPL(id, (__VA_ARGS__),                                 \
DOCTEST_ANONYMOUS(_DOCTEST_ANON_TMP_))
#else 
#define DOCTEST_TEST_CASE_TEMPLATE_INSTANTIATE(id, types)                                          \
DOCTEST_TEST_CASE_TEMPLATE_INSTANTIATE_IMPL(id, types, DOCTEST_ANONYMOUS(_DOCTEST_ANON_TMP_))
#endif 

#if defined(__GNUC__)
#define DOCTEST_SUBCASE(name)                                                                      \
if(const doctest::detail::Subcase & DOCTEST_ANONYMOUS(_DOCTEST_ANON_SUBCASE_)                  \
__attribute__((unused)) =                          \
doctest::detail::Subcase(name, __FILE__, __LINE__))
#else 
#define DOCTEST_SUBCASE(name)                                                                      \
if(const doctest::detail::Subcase & DOCTEST_ANONYMOUS(_DOCTEST_ANON_SUBCASE_) =                \
doctest::detail::Subcase(name, __FILE__, __LINE__))
#endif 

#define DOCTEST_TEST_SUITE_IMPL(decorators, ns_name)                                               \
namespace ns_name                                                                              \
{                                                                                              \
namespace doctest_detail_test_suite_ns                                                     \
{                                                                                          \
inline DOCTEST_NOINLINE doctest::detail::TestSuite& getCurrentTestSuite() {            \
static doctest::detail::TestSuite data;                                            \
static bool                       inited = false;                                  \
if(!inited) {                                                                      \
data* decorators;                                                              \
inited = true;                                                                 \
}                                                                                  \
return data;                                                                       \
}                                                                                      \
}                                                                                          \
}                                                                                              \
namespace ns_name

#define DOCTEST_TEST_SUITE(decorators)                                                             \
DOCTEST_TEST_SUITE_IMPL(decorators, DOCTEST_ANONYMOUS(_DOCTEST_ANON_SUITE_))

#define DOCTEST_TEST_SUITE_BEGIN(decorators)                                                       \
DOCTEST_GLOBAL_NO_WARNINGS(DOCTEST_ANONYMOUS(_DOCTEST_ANON_VAR_)) =                            \
doctest::detail::setTestSuite(doctest::detail::TestSuite() * decorators);              \
DOCTEST_GLOBAL_NO_WARNINGS_END()                                                               \
typedef int DOCTEST_ANONYMOUS(_DOCTEST_ANON_FOR_SEMICOLON_)

#define DOCTEST_TEST_SUITE_END                                                                     \
DOCTEST_GLOBAL_NO_WARNINGS(DOCTEST_ANONYMOUS(_DOCTEST_ANON_VAR_)) =                            \
doctest::detail::setTestSuite(doctest::detail::TestSuite() * "");                      \
DOCTEST_GLOBAL_NO_WARNINGS_END()                                                               \
typedef int DOCTEST_ANONYMOUS(_DOCTEST_ANON_FOR_SEMICOLON_)

#define DOCTEST_REGISTER_EXCEPTION_TRANSLATOR_IMPL(translatorName, signature)                      \
static doctest::String translatorName(signature);                                              \
DOCTEST_GLOBAL_NO_WARNINGS(DOCTEST_ANONYMOUS(_DOCTEST_ANON_TRANSLATOR_)) =                     \
doctest::registerExceptionTranslator(translatorName);                                  \
DOCTEST_GLOBAL_NO_WARNINGS_END()                                                               \
static doctest::String translatorName(signature)

#define DOCTEST_REGISTER_EXCEPTION_TRANSLATOR(signature)                                           \
DOCTEST_REGISTER_EXCEPTION_TRANSLATOR_IMPL(DOCTEST_ANONYMOUS(_DOCTEST_ANON_TRANSLATOR_),       \
signature)

#define DOCTEST_INFO(x)                                                                            \
doctest::detail::ContextScope DOCTEST_ANONYMOUS(_DOCTEST_CAPTURE_)(                            \
doctest::detail::ContextBuilder() << x)
#define DOCTEST_CAPTURE(x) DOCTEST_INFO(#x " := " << x)

#define DOCTEST_ADD_AT_IMPL(type, file, line, mb, x)                                               \
do {                                                                                           \
doctest::detail::MessageBuilder mb(file, line, doctest::detail::assertType::type);         \
mb << x;                                                                                   \
if(mb.log())                                                                               \
DOCTEST_BREAK_INTO_DEBUGGER();                                                         \
mb.react();                                                                                \
} while((void)0, 0)

#define DOCTEST_ADD_MESSAGE_AT(file, line, x) DOCTEST_ADD_AT_IMPL(is_warn, file, line, DOCTEST_ANONYMOUS(_DOCTEST_MESSAGE_), x)
#define DOCTEST_ADD_FAIL_CHECK_AT(file, line, x) DOCTEST_ADD_AT_IMPL(is_check, file, line, DOCTEST_ANONYMOUS(_DOCTEST_MESSAGE_), x)
#define DOCTEST_ADD_FAIL_AT(file, line, x) DOCTEST_ADD_AT_IMPL(is_require, file, line, DOCTEST_ANONYMOUS(_DOCTEST_MESSAGE_), x)

#define DOCTEST_MESSAGE(x) DOCTEST_ADD_MESSAGE_AT(__FILE__, __LINE__, x)
#define DOCTEST_FAIL_CHECK(x) DOCTEST_ADD_FAIL_CHECK_AT(__FILE__, __LINE__, x)
#define DOCTEST_FAIL(x) DOCTEST_ADD_FAIL_AT(__FILE__, __LINE__, x)

#if __cplusplus >= 201402L || (defined(_MSC_VER) && _MSC_VER >= 1910)
template <class T, T x>
constexpr T to_lvalue = x;
#define DOCTEST_TO_LVALUE(...) to_lvalue<decltype(__VA_ARGS__), __VA_ARGS__>
#else
#ifdef DOCTEST_CONFIG_WITH_VARIADIC_MACROS
#define DOCTEST_TO_LVALUE(...) TO_LVALUE_CAN_BE_USED_ONLY_IN_CPP14_MODE_OR_WITH_VS_2017_OR_NEWER
#else 
#define DOCTEST_TO_LVALUE(x) TO_LVALUE_CAN_BE_USED_ONLY_IN_CPP14_MODE_OR_WITH_VS_2017_OR_NEWER
#endif 
#endif 

#define DOCTEST_ASSERT_LOG_AND_REACT(rb)                                                           \
if(rb.log())                                                                                   \
DOCTEST_BREAK_INTO_DEBUGGER();                                                             \
rb.react()

#ifdef DOCTEST_CONFIG_NO_TRY_CATCH_IN_ASSERTS
#define DOCTEST_WRAP_IN_TRY(x) x;
#else 
#define DOCTEST_WRAP_IN_TRY(x)                                                                     \
try {                                                                                          \
x;                                                                                         \
} catch(...) { _DOCTEST_RB.unexpectedExceptionOccurred(); }
#endif 

#define DOCTEST_ASSERT_IMPLEMENT_3(expr, assert_type)                                              \
doctest::detail::ResultBuilder _DOCTEST_RB(                                                    \
doctest::detail::assertType::assert_type, __FILE__, __LINE__,                          \
DOCTEST_TOSTR(DOCTEST_HANDLE_BRACED_VA_ARGS(expr)));                                   \
DOCTEST_WRAP_IN_TRY(_DOCTEST_RB.setResult(                                                     \
doctest::detail::ExpressionDecomposer(doctest::detail::assertType::assert_type)        \
<< DOCTEST_HANDLE_BRACED_VA_ARGS(expr)))                                               \
DOCTEST_ASSERT_LOG_AND_REACT(_DOCTEST_RB)

#if defined(__clang__)
#define DOCTEST_ASSERT_IMPLEMENT_2(expr, assert_type)                                              \
_Pragma("clang diagnostic push")                                                               \
_Pragma("clang diagnostic ignored \"-Woverloaded-shift-op-parentheses\"")              \
DOCTEST_ASSERT_IMPLEMENT_3(expr, assert_type);                                 \
_Pragma("clang diagnostic pop")
#else 
#define DOCTEST_ASSERT_IMPLEMENT_2(expr, assert_type) DOCTEST_ASSERT_IMPLEMENT_3(expr, assert_type);
#endif 

#define DOCTEST_ASSERT_IMPLEMENT_1(expr, assert_type)                                              \
do {                                                                                           \
DOCTEST_ASSERT_IMPLEMENT_2(expr, assert_type);                                             \
} while((void)0, 0)

#ifdef DOCTEST_CONFIG_WITH_VARIADIC_MACROS
#define DOCTEST_WARN(...) DOCTEST_ASSERT_IMPLEMENT_1((__VA_ARGS__), DT_WARN)
#define DOCTEST_CHECK(...) DOCTEST_ASSERT_IMPLEMENT_1((__VA_ARGS__), DT_CHECK)
#define DOCTEST_REQUIRE(...) DOCTEST_ASSERT_IMPLEMENT_1((__VA_ARGS__), DT_REQUIRE)
#define DOCTEST_WARN_FALSE(...) DOCTEST_ASSERT_IMPLEMENT_1((__VA_ARGS__), DT_WARN_FALSE)
#define DOCTEST_CHECK_FALSE(...) DOCTEST_ASSERT_IMPLEMENT_1((__VA_ARGS__), DT_CHECK_FALSE)
#define DOCTEST_REQUIRE_FALSE(...) DOCTEST_ASSERT_IMPLEMENT_1((__VA_ARGS__), DT_REQUIRE_FALSE)
#else 
#define DOCTEST_WARN(expr) DOCTEST_ASSERT_IMPLEMENT_1(expr, DT_WARN)
#define DOCTEST_CHECK(expr) DOCTEST_ASSERT_IMPLEMENT_1(expr, DT_CHECK)
#define DOCTEST_REQUIRE(expr) DOCTEST_ASSERT_IMPLEMENT_1(expr, DT_REQUIRE)
#define DOCTEST_WARN_FALSE(expr) DOCTEST_ASSERT_IMPLEMENT_1(expr, DT_WARN_FALSE)
#define DOCTEST_CHECK_FALSE(expr) DOCTEST_ASSERT_IMPLEMENT_1(expr, DT_CHECK_FALSE)
#define DOCTEST_REQUIRE_FALSE(expr) DOCTEST_ASSERT_IMPLEMENT_1(expr, DT_REQUIRE_FALSE)
#endif 

#ifdef DOCTEST_CONFIG_WITH_VARIADIC_MACROS
#define DOCTEST_WARN_MESSAGE(cond, msg) do { DOCTEST_INFO(msg); DOCTEST_ASSERT_IMPLEMENT_2((cond), DT_WARN); } while((void)0, 0)
#define DOCTEST_CHECK_MESSAGE(cond, msg) do { DOCTEST_INFO(msg); DOCTEST_ASSERT_IMPLEMENT_2((cond), DT_CHECK); } while((void)0, 0)
#define DOCTEST_REQUIRE_MESSAGE(cond, msg) do { DOCTEST_INFO(msg); DOCTEST_ASSERT_IMPLEMENT_2((cond), DT_REQUIRE); } while((void)0, 0)
#define DOCTEST_WARN_FALSE_MESSAGE(cond, msg) do { DOCTEST_INFO(msg); DOCTEST_ASSERT_IMPLEMENT_2((cond), DT_WARN_FALSE); } while((void)0, 0)
#define DOCTEST_CHECK_FALSE_MESSAGE(cond, msg) do { DOCTEST_INFO(msg); DOCTEST_ASSERT_IMPLEMENT_2((cond), DT_CHECK_FALSE); } while((void)0, 0)
#define DOCTEST_REQUIRE_FALSE_MESSAGE(cond, msg) do { DOCTEST_INFO(msg); DOCTEST_ASSERT_IMPLEMENT_2((cond), DT_REQUIRE_FALSE); } while((void)0, 0)
#else 
#define DOCTEST_WARN_MESSAGE(cond, msg) do { DOCTEST_INFO(msg); DOCTEST_ASSERT_IMPLEMENT_2(cond, DT_WARN); } while((void)0, 0)
#define DOCTEST_CHECK_MESSAGE(cond, msg) do { DOCTEST_INFO(msg); DOCTEST_ASSERT_IMPLEMENT_2(cond, DT_CHECK); } while((void)0, 0)
#define DOCTEST_REQUIRE_MESSAGE(cond, msg) do { DOCTEST_INFO(msg); DOCTEST_ASSERT_IMPLEMENT_2(cond, DT_REQUIRE); } while((void)0, 0)
#define DOCTEST_WARN_FALSE_MESSAGE(cond, msg) do { DOCTEST_INFO(msg); DOCTEST_ASSERT_IMPLEMENT_2(cond, DT_WARN_FALSE); } while((void)0, 0)
#define DOCTEST_CHECK_FALSE_MESSAGE(cond, msg) do { DOCTEST_INFO(msg); DOCTEST_ASSERT_IMPLEMENT_2(cond, DT_CHECK_FALSE); } while((void)0, 0)
#define DOCTEST_REQUIRE_FALSE_MESSAGE(cond, msg) do { DOCTEST_INFO(msg); DOCTEST_ASSERT_IMPLEMENT_2(cond, DT_REQUIRE_FALSE); } while((void)0, 0)
#endif 

#define DOCTEST_ASSERT_THROWS(expr, assert_type)                                                   \
do {                                                                                           \
if(!doctest::detail::getTestsContextState()->no_throw) {                                   \
doctest::detail::ResultBuilder _DOCTEST_RB(doctest::detail::assertType::assert_type,   \
__FILE__, __LINE__, #expr);                 \
try {                                                                                  \
expr;                                                                              \
} catch(...) { _DOCTEST_RB.m_threw = true; }                                           \
DOCTEST_ASSERT_LOG_AND_REACT(_DOCTEST_RB);                                             \
}                                                                                          \
} while((void)0, 0)

#define DOCTEST_ASSERT_THROWS_AS(expr, as, assert_type)                                            \
do {                                                                                           \
if(!doctest::detail::getTestsContextState()->no_throw) {                                   \
doctest::detail::ResultBuilder _DOCTEST_RB(                                            \
doctest::detail::assertType::assert_type, __FILE__, __LINE__, #expr,           \
DOCTEST_TOSTR(DOCTEST_HANDLE_BRACED_VA_ARGS(as)));                             \
try {                                                                                  \
expr;                                                                              \
} catch(DOCTEST_HANDLE_BRACED_VA_ARGS(as)) {                                           \
_DOCTEST_RB.m_threw    = true;                                                     \
_DOCTEST_RB.m_threw_as = true;                                                     \
} catch(...) { _DOCTEST_RB.unexpectedExceptionOccurred(); }                            \
DOCTEST_ASSERT_LOG_AND_REACT(_DOCTEST_RB);                                             \
}                                                                                          \
} while((void)0, 0)

#define DOCTEST_ASSERT_NOTHROW(expr, assert_type)                                                  \
do {                                                                                           \
if(!doctest::detail::getTestsContextState()->no_throw) {                                   \
doctest::detail::ResultBuilder _DOCTEST_RB(doctest::detail::assertType::assert_type,   \
__FILE__, __LINE__, #expr);                 \
try {                                                                                  \
expr;                                                                              \
} catch(...) { _DOCTEST_RB.unexpectedExceptionOccurred(); }                            \
DOCTEST_ASSERT_LOG_AND_REACT(_DOCTEST_RB);                                             \
}                                                                                          \
} while((void)0, 0)

#define DOCTEST_WARN_THROWS(expr) DOCTEST_ASSERT_THROWS(expr, DT_WARN_THROWS)
#define DOCTEST_CHECK_THROWS(expr) DOCTEST_ASSERT_THROWS(expr, DT_CHECK_THROWS)
#define DOCTEST_REQUIRE_THROWS(expr) DOCTEST_ASSERT_THROWS(expr, DT_REQUIRE_THROWS)

#ifdef DOCTEST_CONFIG_WITH_VARIADIC_MACROS
#define DOCTEST_WARN_THROWS_AS(expr, ...) DOCTEST_ASSERT_THROWS_AS(expr, (__VA_ARGS__), DT_WARN_THROWS_AS)
#define DOCTEST_CHECK_THROWS_AS(expr, ...) DOCTEST_ASSERT_THROWS_AS(expr, (__VA_ARGS__), DT_CHECK_THROWS_AS)
#define DOCTEST_REQUIRE_THROWS_AS(expr, ...) DOCTEST_ASSERT_THROWS_AS(expr, (__VA_ARGS__), DT_REQUIRE_THROWS_AS)
#else 
#define DOCTEST_WARN_THROWS_AS(expr, ex) DOCTEST_ASSERT_THROWS_AS(expr, ex, DT_WARN_THROWS_AS)
#define DOCTEST_CHECK_THROWS_AS(expr, ex) DOCTEST_ASSERT_THROWS_AS(expr, ex, DT_CHECK_THROWS_AS)
#define DOCTEST_REQUIRE_THROWS_AS(expr, ex) DOCTEST_ASSERT_THROWS_AS(expr, ex, DT_REQUIRE_THROWS_AS)
#endif 

#define DOCTEST_WARN_NOTHROW(expr) DOCTEST_ASSERT_NOTHROW(expr, DT_WARN_NOTHROW)
#define DOCTEST_CHECK_NOTHROW(expr) DOCTEST_ASSERT_NOTHROW(expr, DT_CHECK_NOTHROW)
#define DOCTEST_REQUIRE_NOTHROW(expr) DOCTEST_ASSERT_NOTHROW(expr, DT_REQUIRE_NOTHROW)

#define DOCTEST_WARN_THROWS_MESSAGE(expr, msg) do { DOCTEST_INFO(msg); DOCTEST_WARN_THROWS(expr); } while((void)0, 0)
#define DOCTEST_CHECK_THROWS_MESSAGE(expr, msg) do { DOCTEST_INFO(msg); DOCTEST_CHECK_THROWS(expr); } while((void)0, 0)
#define DOCTEST_REQUIRE_THROWS_MESSAGE(expr, msg) do { DOCTEST_INFO(msg); DOCTEST_REQUIRE_THROWS(expr); } while((void)0, 0)
#define DOCTEST_WARN_THROWS_AS_MESSAGE(expr, ex, msg) do { DOCTEST_INFO(msg); DOCTEST_WARN_THROWS_AS(expr, ex); } while((void)0, 0)
#define DOCTEST_CHECK_THROWS_AS_MESSAGE(expr, ex, msg) do { DOCTEST_INFO(msg); DOCTEST_CHECK_THROWS_AS(expr, ex); } while((void)0, 0)
#define DOCTEST_REQUIRE_THROWS_AS_MESSAGE(expr, ex, msg) do { DOCTEST_INFO(msg); DOCTEST_REQUIRE_THROWS_AS(expr, ex); } while((void)0, 0)
#define DOCTEST_WARN_NOTHROW_MESSAGE(expr, msg) do { DOCTEST_INFO(msg); DOCTEST_WARN_NOTHROW(expr); } while((void)0, 0)
#define DOCTEST_CHECK_NOTHROW_MESSAGE(expr, msg) do { DOCTEST_INFO(msg); DOCTEST_CHECK_NOTHROW(expr); } while((void)0, 0)
#define DOCTEST_REQUIRE_NOTHROW_MESSAGE(expr, msg) do { DOCTEST_INFO(msg); DOCTEST_REQUIRE_NOTHROW(expr); } while((void)0, 0)

#ifdef DOCTEST_CONFIG_WITH_VARIADIC_MACROS
#define DOCTEST_BINARY_ASSERT(assert_type, expr, comp)                                             \
do {                                                                                           \
doctest::detail::ResultBuilder _DOCTEST_RB(                                                \
doctest::detail::assertType::assert_type, __FILE__, __LINE__,                      \
DOCTEST_TOSTR(DOCTEST_HANDLE_BRACED_VA_ARGS(expr)));                               \
DOCTEST_WRAP_IN_TRY(                                                                       \
_DOCTEST_RB.binary_assert<doctest::detail::binaryAssertComparison::comp>(          \
DOCTEST_HANDLE_BRACED_VA_ARGS(expr)))                                      \
DOCTEST_ASSERT_LOG_AND_REACT(_DOCTEST_RB);                                                 \
} while((void)0, 0)
#else 
#define DOCTEST_BINARY_ASSERT(assert_type, lhs, rhs, comp)                                         \
do {                                                                                           \
doctest::detail::ResultBuilder _DOCTEST_RB(doctest::detail::assertType::assert_type,       \
__FILE__, __LINE__, #lhs ", " #rhs);            \
DOCTEST_WRAP_IN_TRY(                                                                       \
_DOCTEST_RB.binary_assert<doctest::detail::binaryAssertComparison::comp>(lhs,      \
rhs))     \
DOCTEST_ASSERT_LOG_AND_REACT(_DOCTEST_RB);                                                 \
} while((void)0, 0)
#endif 

#define DOCTEST_UNARY_ASSERT(assert_type, expr)                                                    \
do {                                                                                           \
doctest::detail::ResultBuilder _DOCTEST_RB(                                                \
doctest::detail::assertType::assert_type, __FILE__, __LINE__,                      \
DOCTEST_TOSTR(DOCTEST_HANDLE_BRACED_VA_ARGS(expr)));                               \
DOCTEST_WRAP_IN_TRY(_DOCTEST_RB.unary_assert(DOCTEST_HANDLE_BRACED_VA_ARGS(expr)))         \
DOCTEST_ASSERT_LOG_AND_REACT(_DOCTEST_RB);                                                 \
} while((void)0, 0)

#ifdef DOCTEST_CONFIG_WITH_VARIADIC_MACROS
#define DOCTEST_WARN_EQ(...) DOCTEST_BINARY_ASSERT(DT_WARN_EQ, (__VA_ARGS__), eq)
#define DOCTEST_CHECK_EQ(...) DOCTEST_BINARY_ASSERT(DT_CHECK_EQ, (__VA_ARGS__), eq)
#define DOCTEST_REQUIRE_EQ(...) DOCTEST_BINARY_ASSERT(DT_REQUIRE_EQ, (__VA_ARGS__), eq)
#define DOCTEST_WARN_NE(...) DOCTEST_BINARY_ASSERT(DT_WARN_NE, (__VA_ARGS__), ne)
#define DOCTEST_CHECK_NE(...) DOCTEST_BINARY_ASSERT(DT_CHECK_NE, (__VA_ARGS__), ne)
#define DOCTEST_REQUIRE_NE(...) DOCTEST_BINARY_ASSERT(DT_REQUIRE_NE, (__VA_ARGS__), ne)
#define DOCTEST_WARN_GT(...) DOCTEST_BINARY_ASSERT(DT_WARN_GT, (__VA_ARGS__), gt)
#define DOCTEST_CHECK_GT(...) DOCTEST_BINARY_ASSERT(DT_CHECK_GT, (__VA_ARGS__), gt)
#define DOCTEST_REQUIRE_GT(...) DOCTEST_BINARY_ASSERT(DT_REQUIRE_GT, (__VA_ARGS__), gt)
#define DOCTEST_WARN_LT(...) DOCTEST_BINARY_ASSERT(DT_WARN_LT, (__VA_ARGS__), lt)
#define DOCTEST_CHECK_LT(...) DOCTEST_BINARY_ASSERT(DT_CHECK_LT, (__VA_ARGS__), lt)
#define DOCTEST_REQUIRE_LT(...) DOCTEST_BINARY_ASSERT(DT_REQUIRE_LT, (__VA_ARGS__), lt)
#define DOCTEST_WARN_GE(...) DOCTEST_BINARY_ASSERT(DT_WARN_GE, (__VA_ARGS__), ge)
#define DOCTEST_CHECK_GE(...) DOCTEST_BINARY_ASSERT(DT_CHECK_GE, (__VA_ARGS__), ge)
#define DOCTEST_REQUIRE_GE(...) DOCTEST_BINARY_ASSERT(DT_REQUIRE_GE, (__VA_ARGS__), ge)
#define DOCTEST_WARN_LE(...) DOCTEST_BINARY_ASSERT(DT_WARN_LE, (__VA_ARGS__), le)
#define DOCTEST_CHECK_LE(...) DOCTEST_BINARY_ASSERT(DT_CHECK_LE, (__VA_ARGS__), le)
#define DOCTEST_REQUIRE_LE(...) DOCTEST_BINARY_ASSERT(DT_REQUIRE_LE, (__VA_ARGS__), le)

#define DOCTEST_WARN_UNARY(...) DOCTEST_UNARY_ASSERT(DT_WARN_UNARY, (__VA_ARGS__))
#define DOCTEST_CHECK_UNARY(...) DOCTEST_UNARY_ASSERT(DT_CHECK_UNARY, (__VA_ARGS__))
#define DOCTEST_REQUIRE_UNARY(...) DOCTEST_UNARY_ASSERT(DT_REQUIRE_UNARY, (__VA_ARGS__))
#define DOCTEST_WARN_UNARY_FALSE(...) DOCTEST_UNARY_ASSERT(DT_WARN_UNARY_FALSE, (__VA_ARGS__))
#define DOCTEST_CHECK_UNARY_FALSE(...) DOCTEST_UNARY_ASSERT(DT_CHECK_UNARY_FALSE, (__VA_ARGS__))
#define DOCTEST_REQUIRE_UNARY_FALSE(...) DOCTEST_UNARY_ASSERT(DT_REQUIRE_UNARY_FALSE, (__VA_ARGS__))
#else 
#define DOCTEST_WARN_EQ(lhs, rhs) DOCTEST_BINARY_ASSERT(DT_WARN_EQ, lhs, rhs, eq)
#define DOCTEST_CHECK_EQ(lhs, rhs) DOCTEST_BINARY_ASSERT(DT_CHECK_EQ, lhs, rhs, eq)
#define DOCTEST_REQUIRE_EQ(lhs, rhs) DOCTEST_BINARY_ASSERT(DT_REQUIRE_EQ, lhs, rhs, eq)
#define DOCTEST_WARN_NE(lhs, rhs) DOCTEST_BINARY_ASSERT(DT_WARN_NE, lhs, rhs, ne)
#define DOCTEST_CHECK_NE(lhs, rhs) DOCTEST_BINARY_ASSERT(DT_CHECK_NE, lhs, rhs, ne)
#define DOCTEST_REQUIRE_NE(lhs, rhs) DOCTEST_BINARY_ASSERT(DT_REQUIRE_NE, lhs, rhs, ne)
#define DOCTEST_WARN_GT(lhs, rhs) DOCTEST_BINARY_ASSERT(DT_WARN_GT, lhs, rhs, gt)
#define DOCTEST_CHECK_GT(lhs, rhs) DOCTEST_BINARY_ASSERT(DT_CHECK_GT, lhs, rhs, gt)
#define DOCTEST_REQUIRE_GT(lhs, rhs) DOCTEST_BINARY_ASSERT(DT_REQUIRE_GT, lhs, rhs, gt)
#define DOCTEST_WARN_LT(lhs, rhs) DOCTEST_BINARY_ASSERT(DT_WARN_LT, lhs, rhs, lt)
#define DOCTEST_CHECK_LT(lhs, rhs) DOCTEST_BINARY_ASSERT(DT_CHECK_LT, lhs, rhs, lt)
#define DOCTEST_REQUIRE_LT(lhs, rhs) DOCTEST_BINARY_ASSERT(DT_REQUIRE_LT, lhs, rhs, lt)
#define DOCTEST_WARN_GE(lhs, rhs) DOCTEST_BINARY_ASSERT(DT_WARN_GE, lhs, rhs, ge)
#define DOCTEST_CHECK_GE(lhs, rhs) DOCTEST_BINARY_ASSERT(DT_CHECK_GE, lhs, rhs, ge)
#define DOCTEST_REQUIRE_GE(lhs, rhs) DOCTEST_BINARY_ASSERT(DT_REQUIRE_GE, lhs, rhs, ge)
#define DOCTEST_WARN_LE(lhs, rhs) DOCTEST_BINARY_ASSERT(DT_WARN_LE, lhs, rhs, le)
#define DOCTEST_CHECK_LE(lhs, rhs) DOCTEST_BINARY_ASSERT(DT_CHECK_LE, lhs, rhs, le)
#define DOCTEST_REQUIRE_LE(lhs, rhs) DOCTEST_BINARY_ASSERT(DT_REQUIRE_LE, lhs, rhs, le)

#define DOCTEST_WARN_UNARY(v) DOCTEST_UNARY_ASSERT(DT_WARN_UNARY, v)
#define DOCTEST_CHECK_UNARY(v) DOCTEST_UNARY_ASSERT(DT_CHECK_UNARY, v)
#define DOCTEST_REQUIRE_UNARY(v) DOCTEST_UNARY_ASSERT(DT_REQUIRE_UNARY, v)
#define DOCTEST_WARN_UNARY_FALSE(v) DOCTEST_UNARY_ASSERT(DT_WARN_UNARY_FALSE, v)
#define DOCTEST_CHECK_UNARY_FALSE(v) DOCTEST_UNARY_ASSERT(DT_CHECK_UNARY_FALSE, v)
#define DOCTEST_REQUIRE_UNARY_FALSE(v) DOCTEST_UNARY_ASSERT(DT_REQUIRE_UNARY_FALSE, v)
#endif 

#ifndef DOCTEST_CONFIG_SUPER_FAST_ASSERTS

#ifdef DOCTEST_CONFIG_WITH_VARIADIC_MACROS
#define DOCTEST_FAST_BINARY_ASSERT(assert_type, expr, comparison)                                  \
do {                                                                                           \
int _DOCTEST_FAST_RES = doctest::detail::fast_binary_assert<                               \
doctest::detail::binaryAssertComparison::comparison>(                              \
doctest::detail::assertType::assert_type, __FILE__, __LINE__,                      \
DOCTEST_TOSTR(DOCTEST_HANDLE_BRACED_VA_ARGS(expr)),                                \
DOCTEST_HANDLE_BRACED_VA_ARGS(expr));                                              \
if(_DOCTEST_FAST_RES & doctest::detail::assertAction::dbgbreak)                            \
DOCTEST_BREAK_INTO_DEBUGGER();                                                         \
doctest::detail::fastAssertThrowIfFlagSet(_DOCTEST_FAST_RES);                              \
} while((void)0, 0)
#else 
#define DOCTEST_FAST_BINARY_ASSERT(assert_type, lhs, rhs, comparison)                              \
do {                                                                                           \
int _DOCTEST_FAST_RES = doctest::detail::fast_binary_assert<                               \
doctest::detail::binaryAssertComparison::comparison>(                              \
doctest::detail::assertType::assert_type, __FILE__, __LINE__, #lhs ", " #rhs, lhs, \
rhs);                                                                              \
if(_DOCTEST_FAST_RES & doctest::detail::assertAction::dbgbreak)                            \
DOCTEST_BREAK_INTO_DEBUGGER();                                                         \
doctest::detail::fastAssertThrowIfFlagSet(_DOCTEST_FAST_RES);                              \
} while((void)0, 0)
#endif 

#define DOCTEST_FAST_UNARY_ASSERT(assert_type, expr)                                               \
do {                                                                                           \
int _DOCTEST_FAST_RES = doctest::detail::fast_unary_assert(                                \
doctest::detail::assertType::assert_type, __FILE__, __LINE__,                      \
DOCTEST_TOSTR(DOCTEST_HANDLE_BRACED_VA_ARGS(expr)),                                \
DOCTEST_HANDLE_BRACED_VA_ARGS(expr));                                              \
if(_DOCTEST_FAST_RES & doctest::detail::assertAction::dbgbreak)                            \
DOCTEST_BREAK_INTO_DEBUGGER();                                                         \
doctest::detail::fastAssertThrowIfFlagSet(_DOCTEST_FAST_RES);                              \
} while((void)0, 0)

#else 

#ifdef DOCTEST_CONFIG_WITH_VARIADIC_MACROS
#define DOCTEST_FAST_BINARY_ASSERT(assert_type, expr, comparison)                                  \
doctest::detail::fast_binary_assert<doctest::detail::binaryAssertComparison::comparison>(      \
doctest::detail::assertType::assert_type, __FILE__, __LINE__,                          \
DOCTEST_TOSTR(DOCTEST_HANDLE_BRACED_VA_ARGS(expr)),                                    \
DOCTEST_HANDLE_BRACED_VA_ARGS(expr))
#else 
#define DOCTEST_FAST_BINARY_ASSERT(assert_type, lhs, rhs, comparison)                              \
doctest::detail::fast_binary_assert<doctest::detail::binaryAssertComparison::comparison>(      \
doctest::detail::assertType::assert_type, __FILE__, __LINE__, #lhs ", " #rhs, lhs,     \
rhs)
#endif 

#define DOCTEST_FAST_UNARY_ASSERT(assert_type, expr)                                               \
doctest::detail::fast_unary_assert(doctest::detail::assertType::assert_type, __FILE__,         \
__LINE__,                                                   \
DOCTEST_TOSTR(DOCTEST_HANDLE_BRACED_VA_ARGS(expr)),         \
DOCTEST_HANDLE_BRACED_VA_ARGS(expr))

#endif 

#ifdef DOCTEST_CONFIG_WITH_VARIADIC_MACROS
#define DOCTEST_FAST_WARN_EQ(...) DOCTEST_FAST_BINARY_ASSERT(DT_FAST_WARN_EQ, (__VA_ARGS__), eq)
#define DOCTEST_FAST_CHECK_EQ(...) DOCTEST_FAST_BINARY_ASSERT(DT_FAST_CHECK_EQ, (__VA_ARGS__), eq)
#define DOCTEST_FAST_REQUIRE_EQ(...) DOCTEST_FAST_BINARY_ASSERT(DT_FAST_REQUIRE_EQ, (__VA_ARGS__), eq)
#define DOCTEST_FAST_WARN_NE(...) DOCTEST_FAST_BINARY_ASSERT(DT_FAST_WARN_NE, (__VA_ARGS__), ne)
#define DOCTEST_FAST_CHECK_NE(...) DOCTEST_FAST_BINARY_ASSERT(DT_FAST_CHECK_NE, (__VA_ARGS__), ne)
#define DOCTEST_FAST_REQUIRE_NE(...) DOCTEST_FAST_BINARY_ASSERT(DT_FAST_REQUIRE_NE, (__VA_ARGS__), ne)
#define DOCTEST_FAST_WARN_GT(...) DOCTEST_FAST_BINARY_ASSERT(DT_FAST_WARN_GT, (__VA_ARGS__), gt)
#define DOCTEST_FAST_CHECK_GT(...) DOCTEST_FAST_BINARY_ASSERT(DT_FAST_CHECK_GT, (__VA_ARGS__), gt)
#define DOCTEST_FAST_REQUIRE_GT(...) DOCTEST_FAST_BINARY_ASSERT(DT_FAST_REQUIRE_GT, (__VA_ARGS__), gt)
#define DOCTEST_FAST_WARN_LT(...) DOCTEST_FAST_BINARY_ASSERT(DT_FAST_WARN_LT, (__VA_ARGS__), lt)
#define DOCTEST_FAST_CHECK_LT(...) DOCTEST_FAST_BINARY_ASSERT(DT_FAST_CHECK_LT, (__VA_ARGS__), lt)
#define DOCTEST_FAST_REQUIRE_LT(...) DOCTEST_FAST_BINARY_ASSERT(DT_FAST_REQUIRE_LT, (__VA_ARGS__), lt)
#define DOCTEST_FAST_WARN_GE(...) DOCTEST_FAST_BINARY_ASSERT(DT_FAST_WARN_GE, (__VA_ARGS__), ge)
#define DOCTEST_FAST_CHECK_GE(...) DOCTEST_FAST_BINARY_ASSERT(DT_FAST_CHECK_GE, (__VA_ARGS__), ge)
#define DOCTEST_FAST_REQUIRE_GE(...) DOCTEST_FAST_BINARY_ASSERT(DT_FAST_REQUIRE_GE, (__VA_ARGS__), ge)
#define DOCTEST_FAST_WARN_LE(...) DOCTEST_FAST_BINARY_ASSERT(DT_FAST_WARN_LE, (__VA_ARGS__), le)
#define DOCTEST_FAST_CHECK_LE(...) DOCTEST_FAST_BINARY_ASSERT(DT_FAST_CHECK_LE, (__VA_ARGS__), le)
#define DOCTEST_FAST_REQUIRE_LE(...) DOCTEST_FAST_BINARY_ASSERT(DT_FAST_REQUIRE_LE, (__VA_ARGS__), le)

#define DOCTEST_FAST_WARN_UNARY(...) DOCTEST_FAST_UNARY_ASSERT(DT_FAST_WARN_UNARY, (__VA_ARGS__))
#define DOCTEST_FAST_CHECK_UNARY(...) DOCTEST_FAST_UNARY_ASSERT(DT_FAST_CHECK_UNARY, (__VA_ARGS__))
#define DOCTEST_FAST_REQUIRE_UNARY(...) DOCTEST_FAST_UNARY_ASSERT(DT_FAST_REQUIRE_UNARY, (__VA_ARGS__))
#define DOCTEST_FAST_WARN_UNARY_FALSE(...) DOCTEST_FAST_UNARY_ASSERT(DT_FAST_WARN_UNARY_FALSE, (__VA_ARGS__))
#define DOCTEST_FAST_CHECK_UNARY_FALSE(...) DOCTEST_FAST_UNARY_ASSERT(DT_FAST_CHECK_UNARY_FALSE, (__VA_ARGS__))
#define DOCTEST_FAST_REQUIRE_UNARY_FALSE(...) DOCTEST_FAST_UNARY_ASSERT(DT_FAST_REQUIRE_UNARY_FALSE, (__VA_ARGS__))
#else 
#define DOCTEST_FAST_WARN_EQ(l, r) DOCTEST_FAST_BINARY_ASSERT(DT_FAST_WARN_EQ, l, r, eq)
#define DOCTEST_FAST_CHECK_EQ(l, r) DOCTEST_FAST_BINARY_ASSERT(DT_FAST_CHECK_EQ, l, r, eq)
#define DOCTEST_FAST_REQUIRE_EQ(l, r) DOCTEST_FAST_BINARY_ASSERT(DT_FAST_REQUIRE_EQ, l, r, eq)
#define DOCTEST_FAST_WARN_NE(l, r) DOCTEST_FAST_BINARY_ASSERT(DT_FAST_WARN_NE, l, r, ne)
#define DOCTEST_FAST_CHECK_NE(l, r) DOCTEST_FAST_BINARY_ASSERT(DT_FAST_CHECK_NE, l, r, ne)
#define DOCTEST_FAST_REQUIRE_NE(l, r) DOCTEST_FAST_BINARY_ASSERT(DT_FAST_REQUIRE_NE, l, r, ne)
#define DOCTEST_FAST_WARN_GT(l, r) DOCTEST_FAST_BINARY_ASSERT(DT_FAST_WARN_GT, l, r, gt)
#define DOCTEST_FAST_CHECK_GT(l, r) DOCTEST_FAST_BINARY_ASSERT(DT_FAST_CHECK_GT, l, r, gt)
#define DOCTEST_FAST_REQUIRE_GT(l, r) DOCTEST_FAST_BINARY_ASSERT(DT_FAST_REQUIRE_GT, l, r, gt)
#define DOCTEST_FAST_WARN_LT(l, r) DOCTEST_FAST_BINARY_ASSERT(DT_FAST_WARN_LT, l, r, lt)
#define DOCTEST_FAST_CHECK_LT(l, r) DOCTEST_FAST_BINARY_ASSERT(DT_FAST_CHECK_LT, l, r, lt)
#define DOCTEST_FAST_REQUIRE_LT(l, r) DOCTEST_FAST_BINARY_ASSERT(DT_FAST_REQUIRE_LT, l, r, lt)
#define DOCTEST_FAST_WARN_GE(l, r) DOCTEST_FAST_BINARY_ASSERT(DT_FAST_WARN_GE, l, r, ge)
#define DOCTEST_FAST_CHECK_GE(l, r) DOCTEST_FAST_BINARY_ASSERT(DT_FAST_CHECK_GE, l, r, ge)
#define DOCTEST_FAST_REQUIRE_GE(l, r) DOCTEST_FAST_BINARY_ASSERT(DT_FAST_REQUIRE_GE, l, r, ge)
#define DOCTEST_FAST_WARN_LE(l, r) DOCTEST_FAST_BINARY_ASSERT(DT_FAST_WARN_LE, l, r, le)
#define DOCTEST_FAST_CHECK_LE(l, r) DOCTEST_FAST_BINARY_ASSERT(DT_FAST_CHECK_LE, l, r, le)
#define DOCTEST_FAST_REQUIRE_LE(l, r) DOCTEST_FAST_BINARY_ASSERT(DT_FAST_REQUIRE_LE, l, r, le)

#define DOCTEST_FAST_WARN_UNARY(v) DOCTEST_FAST_UNARY_ASSERT(DT_FAST_WARN_UNARY, v)
#define DOCTEST_FAST_CHECK_UNARY(v) DOCTEST_FAST_UNARY_ASSERT(DT_FAST_CHECK_UNARY, v)
#define DOCTEST_FAST_REQUIRE_UNARY(v) DOCTEST_FAST_UNARY_ASSERT(DT_FAST_REQUIRE_UNARY, v)
#define DOCTEST_FAST_WARN_UNARY_FALSE(v) DOCTEST_FAST_UNARY_ASSERT(DT_FAST_WARN_UNARY_FALSE, v)
#define DOCTEST_FAST_CHECK_UNARY_FALSE(v) DOCTEST_FAST_UNARY_ASSERT(DT_FAST_CHECK_UNARY_FALSE, v)
#define DOCTEST_FAST_REQUIRE_UNARY_FALSE(v) DOCTEST_FAST_UNARY_ASSERT(DT_FAST_REQUIRE_UNARY_FALSE, v)
#endif 

#ifdef DOCTEST_CONFIG_NO_EXCEPTIONS

#undef DOCTEST_WARN_THROWS
#undef DOCTEST_CHECK_THROWS
#undef DOCTEST_REQUIRE_THROWS
#undef DOCTEST_WARN_THROWS_AS
#undef DOCTEST_CHECK_THROWS_AS
#undef DOCTEST_REQUIRE_THROWS_AS
#undef DOCTEST_WARN_NOTHROW
#undef DOCTEST_CHECK_NOTHROW
#undef DOCTEST_REQUIRE_NOTHROW

#undef DOCTEST_WARN_THROWS_MESSAGE
#undef DOCTEST_CHECK_THROWS_MESSAGE
#undef DOCTEST_REQUIRE_THROWS_MESSAGE
#undef DOCTEST_WARN_THROWS_AS_MESSAGE
#undef DOCTEST_CHECK_THROWS_AS_MESSAGE
#undef DOCTEST_REQUIRE_THROWS_AS_MESSAGE
#undef DOCTEST_WARN_NOTHROW_MESSAGE
#undef DOCTEST_CHECK_NOTHROW_MESSAGE
#undef DOCTEST_REQUIRE_NOTHROW_MESSAGE

#ifdef DOCTEST_CONFIG_NO_EXCEPTIONS_BUT_WITH_ALL_ASSERTS

#define DOCTEST_WARN_THROWS(expr) ((void)0)
#define DOCTEST_CHECK_THROWS(expr) ((void)0)
#define DOCTEST_REQUIRE_THROWS(expr) ((void)0)
#ifdef DOCTEST_CONFIG_WITH_VARIADIC_MACROS
#define DOCTEST_WARN_THROWS_AS(expr, ...) ((void)0)
#define DOCTEST_CHECK_THROWS_AS(expr, ...) ((void)0)
#define DOCTEST_REQUIRE_THROWS_AS(expr, ...) ((void)0)
#else 
#define DOCTEST_WARN_THROWS_AS(expr, ex) ((void)0)
#define DOCTEST_CHECK_THROWS_AS(expr, ex) ((void)0)
#define DOCTEST_REQUIRE_THROWS_AS(expr, ex) ((void)0)
#endif 
#define DOCTEST_WARN_NOTHROW(expr) ((void)0)
#define DOCTEST_CHECK_NOTHROW(expr) ((void)0)
#define DOCTEST_REQUIRE_NOTHROW(expr) ((void)0)

#define DOCTEST_WARN_THROWS_MESSAGE(expr, msg) ((void)0)
#define DOCTEST_CHECK_THROWS_MESSAGE(expr, msg) ((void)0)
#define DOCTEST_REQUIRE_THROWS_MESSAGE(expr, msg) ((void)0)
#define DOCTEST_WARN_THROWS_AS_MESSAGE(expr, ex, msg) ((void)0)
#define DOCTEST_CHECK_THROWS_AS_MESSAGE(expr, ex, msg) ((void)0)
#define DOCTEST_REQUIRE_THROWS_AS_MESSAGE(expr, ex, msg) ((void)0)
#define DOCTEST_WARN_NOTHROW_MESSAGE(expr, msg) ((void)0)
#define DOCTEST_CHECK_NOTHROW_MESSAGE(expr, msg) ((void)0)
#define DOCTEST_REQUIRE_NOTHROW_MESSAGE(expr, msg) ((void)0)

#else 

#undef DOCTEST_REQUIRE
#undef DOCTEST_REQUIRE_FALSE
#undef DOCTEST_REQUIRE_MESSAGE
#undef DOCTEST_REQUIRE_FALSE_MESSAGE
#undef DOCTEST_REQUIRE_EQ
#undef DOCTEST_REQUIRE_NE
#undef DOCTEST_REQUIRE_GT
#undef DOCTEST_REQUIRE_LT
#undef DOCTEST_REQUIRE_GE
#undef DOCTEST_REQUIRE_LE
#undef DOCTEST_REQUIRE_UNARY
#undef DOCTEST_REQUIRE_UNARY_FALSE
#undef DOCTEST_FAST_REQUIRE_EQ
#undef DOCTEST_FAST_REQUIRE_NE
#undef DOCTEST_FAST_REQUIRE_GT
#undef DOCTEST_FAST_REQUIRE_LT
#undef DOCTEST_FAST_REQUIRE_GE
#undef DOCTEST_FAST_REQUIRE_LE
#undef DOCTEST_FAST_REQUIRE_UNARY
#undef DOCTEST_FAST_REQUIRE_UNARY_FALSE

#endif 

#endif 

#else 

#define DOCTEST_IMPLEMENT_FIXTURE(der, base, func, name)                                           \
namespace                                                                                      \
{                                                                                              \
template <typename T>                                                                      \
struct der : base                                                                          \
{ void f(); };                                                                             \
}                                                                                              \
template <typename T>                                                                          \
inline void der<T>::f()

#define DOCTEST_CREATE_AND_REGISTER_FUNCTION(f, name)                                              \
template <typename T>                                                                          \
static inline void f()

#define DOCTEST_TEST_CASE(name)                                                                    \
DOCTEST_CREATE_AND_REGISTER_FUNCTION(DOCTEST_ANONYMOUS(_DOCTEST_ANON_FUNC_), name)

#define DOCTEST_TEST_CASE_FIXTURE(x, name)                                                         \
DOCTEST_IMPLEMENT_FIXTURE(DOCTEST_ANONYMOUS(_DOCTEST_ANON_CLASS_), x,                          \
DOCTEST_ANONYMOUS(_DOCTEST_ANON_FUNC_), name)

#ifdef DOCTEST_CONFIG_WITH_VARIADIC_MACROS
#define DOCTEST_TYPE_TO_STRING(...) typedef int DOCTEST_ANONYMOUS(_DOCTEST_ANON_FOR_SEMICOLON_)
#define DOCTEST_TYPE_TO_STRING_IMPL(...)
#else 
#define DOCTEST_TYPE_TO_STRING(x) typedef int DOCTEST_ANONYMOUS(_DOCTEST_ANON_FOR_SEMICOLON_)
#define DOCTEST_TYPE_TO_STRING_IMPL(x)
#endif 

#define DOCTEST_TEST_CASE_TEMPLATE(name, type, types)                                              \
template <typename type>                                                                       \
inline void DOCTEST_ANONYMOUS(_DOCTEST_ANON_TMP_)()

#define DOCTEST_TEST_CASE_TEMPLATE_DEFINE(name, type, id)                                          \
template <typename type>                                                                       \
inline void DOCTEST_ANONYMOUS(_DOCTEST_ANON_TMP_)()

#define DOCTEST_TEST_CASE_TEMPLATE_INSTANTIATE(id, types)                                          \
typedef int DOCTEST_ANONYMOUS(_DOCTEST_ANON_FOR_SEMICOLON_)

#define DOCTEST_SUBCASE(name)

#define DOCTEST_TEST_SUITE(name) namespace

#define DOCTEST_TEST_SUITE_BEGIN(name) typedef int DOCTEST_ANONYMOUS(_DOCTEST_ANON_FOR_SEMICOLON_)

#define DOCTEST_TEST_SUITE_END typedef int DOCTEST_ANONYMOUS(_DOCTEST_ANON_FOR_SEMICOLON_)

#define DOCTEST_REGISTER_EXCEPTION_TRANSLATOR(signature)                                           \
template <typename T>                                                                          \
static inline doctest::String DOCTEST_ANONYMOUS(_DOCTEST_ANON_TRANSLATOR_)(signature)

#define DOCTEST_INFO(x) ((void)0)
#define DOCTEST_CAPTURE(x) ((void)0)
#define DOCTEST_ADD_MESSAGE_AT(file, line, x) ((void)0)
#define DOCTEST_ADD_FAIL_CHECK_AT(file, line, x) ((void)0)
#define DOCTEST_ADD_FAIL_AT(file, line, x) ((void)0)
#define DOCTEST_MESSAGE(x) ((void)0)
#define DOCTEST_FAIL_CHECK(x) ((void)0)
#define DOCTEST_FAIL(x) ((void)0)

#ifdef DOCTEST_CONFIG_WITH_VARIADIC_MACROS
#define DOCTEST_WARN(...) ((void)0)
#define DOCTEST_CHECK(...) ((void)0)
#define DOCTEST_REQUIRE(...) ((void)0)
#define DOCTEST_WARN_FALSE(...) ((void)0)
#define DOCTEST_CHECK_FALSE(...) ((void)0)
#define DOCTEST_REQUIRE_FALSE(...) ((void)0)
#else 
#define DOCTEST_WARN(expr) ((void)0)
#define DOCTEST_CHECK(expr) ((void)0)
#define DOCTEST_REQUIRE(expr) ((void)0)
#define DOCTEST_WARN_FALSE(expr) ((void)0)
#define DOCTEST_CHECK_FALSE(expr) ((void)0)
#define DOCTEST_REQUIRE_FALSE(expr) ((void)0)
#endif 

#define DOCTEST_WARN_MESSAGE(cond, msg) ((void)0)
#define DOCTEST_CHECK_MESSAGE(cond, msg) ((void)0)
#define DOCTEST_REQUIRE_MESSAGE(cond, msg) ((void)0)
#define DOCTEST_WARN_FALSE_MESSAGE(cond, msg) ((void)0)
#define DOCTEST_CHECK_FALSE_MESSAGE(cond, msg) ((void)0)
#define DOCTEST_REQUIRE_FALSE_MESSAGE(cond, msg) ((void)0)

#define DOCTEST_WARN_THROWS(expr) ((void)0)
#define DOCTEST_CHECK_THROWS(expr) ((void)0)
#define DOCTEST_REQUIRE_THROWS(expr) ((void)0)
#ifdef DOCTEST_CONFIG_WITH_VARIADIC_MACROS
#define DOCTEST_WARN_THROWS_AS(expr, ...) ((void)0)
#define DOCTEST_CHECK_THROWS_AS(expr, ...) ((void)0)
#define DOCTEST_REQUIRE_THROWS_AS(expr, ...) ((void)0)
#else 
#define DOCTEST_WARN_THROWS_AS(expr, ex) ((void)0)
#define DOCTEST_CHECK_THROWS_AS(expr, ex) ((void)0)
#define DOCTEST_REQUIRE_THROWS_AS(expr, ex) ((void)0)
#endif 
#define DOCTEST_WARN_NOTHROW(expr) ((void)0)
#define DOCTEST_CHECK_NOTHROW(expr) ((void)0)
#define DOCTEST_REQUIRE_NOTHROW(expr) ((void)0)

#define DOCTEST_WARN_THROWS_MESSAGE(expr, msg) ((void)0)
#define DOCTEST_CHECK_THROWS_MESSAGE(expr, msg) ((void)0)
#define DOCTEST_REQUIRE_THROWS_MESSAGE(expr, msg) ((void)0)
#define DOCTEST_WARN_THROWS_AS_MESSAGE(expr, ex, msg) ((void)0)
#define DOCTEST_CHECK_THROWS_AS_MESSAGE(expr, ex, msg) ((void)0)
#define DOCTEST_REQUIRE_THROWS_AS_MESSAGE(expr, ex, msg) ((void)0)
#define DOCTEST_WARN_NOTHROW_MESSAGE(expr, msg) ((void)0)
#define DOCTEST_CHECK_NOTHROW_MESSAGE(expr, msg) ((void)0)
#define DOCTEST_REQUIRE_NOTHROW_MESSAGE(expr, msg) ((void)0)

#ifdef DOCTEST_CONFIG_WITH_VARIADIC_MACROS

#define DOCTEST_WARN_EQ(...) ((void)0)
#define DOCTEST_CHECK_EQ(...) ((void)0)
#define DOCTEST_REQUIRE_EQ(...) ((void)0)
#define DOCTEST_WARN_NE(...) ((void)0)
#define DOCTEST_CHECK_NE(...) ((void)0)
#define DOCTEST_REQUIRE_NE(...) ((void)0)
#define DOCTEST_WARN_GT(...) ((void)0)
#define DOCTEST_CHECK_GT(...) ((void)0)
#define DOCTEST_REQUIRE_GT(...) ((void)0)
#define DOCTEST_WARN_LT(...) ((void)0)
#define DOCTEST_CHECK_LT(...) ((void)0)
#define DOCTEST_REQUIRE_LT(...) ((void)0)
#define DOCTEST_WARN_GE(...) ((void)0)
#define DOCTEST_CHECK_GE(...) ((void)0)
#define DOCTEST_REQUIRE_GE(...) ((void)0)
#define DOCTEST_WARN_LE(...) ((void)0)
#define DOCTEST_CHECK_LE(...) ((void)0)
#define DOCTEST_REQUIRE_LE(...) ((void)0)

#define DOCTEST_WARN_UNARY(...) ((void)0)
#define DOCTEST_CHECK_UNARY(...) ((void)0)
#define DOCTEST_REQUIRE_UNARY(...) ((void)0)
#define DOCTEST_WARN_UNARY_FALSE(...) ((void)0)
#define DOCTEST_CHECK_UNARY_FALSE(...) ((void)0)
#define DOCTEST_REQUIRE_UNARY_FALSE(...) ((void)0)

#define DOCTEST_FAST_WARN_EQ(...) ((void)0)
#define DOCTEST_FAST_CHECK_EQ(...) ((void)0)
#define DOCTEST_FAST_REQUIRE_EQ(...) ((void)0)
#define DOCTEST_FAST_WARN_NE(...) ((void)0)
#define DOCTEST_FAST_CHECK_NE(...) ((void)0)
#define DOCTEST_FAST_REQUIRE_NE(...) ((void)0)
#define DOCTEST_FAST_WARN_GT(...) ((void)0)
#define DOCTEST_FAST_CHECK_GT(...) ((void)0)
#define DOCTEST_FAST_REQUIRE_GT(...) ((void)0)
#define DOCTEST_FAST_WARN_LT(...) ((void)0)
#define DOCTEST_FAST_CHECK_LT(...) ((void)0)
#define DOCTEST_FAST_REQUIRE_LT(...) ((void)0)
#define DOCTEST_FAST_WARN_GE(...) ((void)0)
#define DOCTEST_FAST_CHECK_GE(...) ((void)0)
#define DOCTEST_FAST_REQUIRE_GE(...) ((void)0)
#define DOCTEST_FAST_WARN_LE(...) ((void)0)
#define DOCTEST_FAST_CHECK_LE(...) ((void)0)
#define DOCTEST_FAST_REQUIRE_LE(...) ((void)0)

#define DOCTEST_FAST_WARN_UNARY(...) ((void)0)
#define DOCTEST_FAST_CHECK_UNARY(...) ((void)0)
#define DOCTEST_FAST_REQUIRE_UNARY(...) ((void)0)
#define DOCTEST_FAST_WARN_UNARY_FALSE(...) ((void)0)
#define DOCTEST_FAST_CHECK_UNARY_FALSE(...) ((void)0)
#define DOCTEST_FAST_REQUIRE_UNARY_FALSE(...) ((void)0)

#else 

#define DOCTEST_WARN_EQ(lhs, rhs) ((void)0)
#define DOCTEST_CHECK_EQ(lhs, rhs) ((void)0)
#define DOCTEST_REQUIRE_EQ(lhs, rhs) ((void)0)
#define DOCTEST_WARN_NE(lhs, rhs) ((void)0)
#define DOCTEST_CHECK_NE(lhs, rhs) ((void)0)
#define DOCTEST_REQUIRE_NE(lhs, rhs) ((void)0)
#define DOCTEST_WARN_GT(lhs, rhs) ((void)0)
#define DOCTEST_CHECK_GT(lhs, rhs) ((void)0)
#define DOCTEST_REQUIRE_GT(lhs, rhs) ((void)0)
#define DOCTEST_WARN_LT(lhs, rhs) ((void)0)
#define DOCTEST_CHECK_LT(lhs, rhs) ((void)0)
#define DOCTEST_REQUIRE_LT(lhs, rhs) ((void)0)
#define DOCTEST_WARN_GE(lhs, rhs) ((void)0)
#define DOCTEST_CHECK_GE(lhs, rhs) ((void)0)
#define DOCTEST_REQUIRE_GE(lhs, rhs) ((void)0)
#define DOCTEST_WARN_LE(lhs, rhs) ((void)0)
#define DOCTEST_CHECK_LE(lhs, rhs) ((void)0)
#define DOCTEST_REQUIRE_LE(lhs, rhs) ((void)0)

#define DOCTEST_WARN_UNARY(val) ((void)0)
#define DOCTEST_CHECK_UNARY(val) ((void)0)
#define DOCTEST_REQUIRE_UNARY(val) ((void)0)
#define DOCTEST_WARN_UNARY_FALSE(val) ((void)0)
#define DOCTEST_CHECK_UNARY_FALSE(val) ((void)0)
#define DOCTEST_REQUIRE_UNARY_FALSE(val) ((void)0)

#define DOCTEST_FAST_WARN_EQ(lhs, rhs) ((void)0)
#define DOCTEST_FAST_CHECK_EQ(lhs, rhs) ((void)0)
#define DOCTEST_FAST_REQUIRE_EQ(lhs, rhs) ((void)0)
#define DOCTEST_FAST_WARN_NE(lhs, rhs) ((void)0)
#define DOCTEST_FAST_CHECK_NE(lhs, rhs) ((void)0)
#define DOCTEST_FAST_REQUIRE_NE(lhs, rhs) ((void)0)
#define DOCTEST_FAST_WARN_GT(lhs, rhs) ((void)0)
#define DOCTEST_FAST_CHECK_GT(lhs, rhs) ((void)0)
#define DOCTEST_FAST_REQUIRE_GT(lhs, rhs) ((void)0)
#define DOCTEST_FAST_WARN_LT(lhs, rhs) ((void)0)
#define DOCTEST_FAST_CHECK_LT(lhs, rhs) ((void)0)
#define DOCTEST_FAST_REQUIRE_LT(lhs, rhs) ((void)0)
#define DOCTEST_FAST_WARN_GE(lhs, rhs) ((void)0)
#define DOCTEST_FAST_CHECK_GE(lhs, rhs) ((void)0)
#define DOCTEST_FAST_REQUIRE_GE(lhs, rhs) ((void)0)
#define DOCTEST_FAST_WARN_LE(lhs, rhs) ((void)0)
#define DOCTEST_FAST_CHECK_LE(lhs, rhs) ((void)0)
#define DOCTEST_FAST_REQUIRE_LE(lhs, rhs) ((void)0)

#define DOCTEST_FAST_WARN_UNARY(val) ((void)0)
#define DOCTEST_FAST_CHECK_UNARY(val) ((void)0)
#define DOCTEST_FAST_REQUIRE_UNARY(val) ((void)0)
#define DOCTEST_FAST_WARN_UNARY_FALSE(val) ((void)0)
#define DOCTEST_FAST_CHECK_UNARY_FALSE(val) ((void)0)
#define DOCTEST_FAST_REQUIRE_UNARY_FALSE(val) ((void)0)

#endif 

#endif 

#define DOCTEST_SCENARIO(name)  TEST_CASE("  Scenario: " name)
#define DOCTEST_GIVEN(name)     SUBCASE("   Given: " name)
#define DOCTEST_WHEN(name)      SUBCASE("    When: " name)
#define DOCTEST_AND_WHEN(name)  SUBCASE("And when: " name)
#define DOCTEST_THEN(name)      SUBCASE("    Then: " name)
#define DOCTEST_AND_THEN(name)  SUBCASE("     And: " name)

#if !defined(DOCTEST_CONFIG_NO_SHORT_MACRO_NAMES)

#define TEST_CASE DOCTEST_TEST_CASE
#define TEST_CASE_FIXTURE DOCTEST_TEST_CASE_FIXTURE
#define TYPE_TO_STRING DOCTEST_TYPE_TO_STRING
#define TEST_CASE_TEMPLATE DOCTEST_TEST_CASE_TEMPLATE
#define TEST_CASE_TEMPLATE_DEFINE DOCTEST_TEST_CASE_TEMPLATE_DEFINE
#define TEST_CASE_TEMPLATE_INSTANTIATE DOCTEST_TEST_CASE_TEMPLATE_INSTANTIATE
#define SUBCASE DOCTEST_SUBCASE
#define TEST_SUITE DOCTEST_TEST_SUITE
#define TEST_SUITE_BEGIN DOCTEST_TEST_SUITE_BEGIN
#define TEST_SUITE_END DOCTEST_TEST_SUITE_END
#define REGISTER_EXCEPTION_TRANSLATOR DOCTEST_REGISTER_EXCEPTION_TRANSLATOR
#define INFO DOCTEST_INFO
#define CAPTURE DOCTEST_CAPTURE
#define ADD_MESSAGE_AT DOCTEST_ADD_MESSAGE_AT
#define ADD_FAIL_CHECK_AT DOCTEST_ADD_FAIL_CHECK_AT
#define ADD_FAIL_AT DOCTEST_ADD_FAIL_AT
#define MESSAGE DOCTEST_MESSAGE
#define FAIL_CHECK DOCTEST_FAIL_CHECK
#define FAIL DOCTEST_FAIL
#define TO_LVALUE DOCTEST_TO_LVALUE

#define WARN DOCTEST_WARN
#define WARN_FALSE DOCTEST_WARN_FALSE
#define WARN_THROWS DOCTEST_WARN_THROWS
#define WARN_THROWS_AS DOCTEST_WARN_THROWS_AS
#define WARN_NOTHROW DOCTEST_WARN_NOTHROW
#define CHECK DOCTEST_CHECK
#define CHECK_FALSE DOCTEST_CHECK_FALSE
#define CHECK_THROWS DOCTEST_CHECK_THROWS
#define CHECK_THROWS_AS DOCTEST_CHECK_THROWS_AS
#define CHECK_NOTHROW DOCTEST_CHECK_NOTHROW
#define REQUIRE DOCTEST_REQUIRE
#define REQUIRE_FALSE DOCTEST_REQUIRE_FALSE
#define REQUIRE_THROWS DOCTEST_REQUIRE_THROWS
#define REQUIRE_THROWS_AS DOCTEST_REQUIRE_THROWS_AS
#define REQUIRE_NOTHROW DOCTEST_REQUIRE_NOTHROW

#define WARN_MESSAGE DOCTEST_WARN_MESSAGE
#define WARN_FALSE_MESSAGE DOCTEST_WARN_FALSE_MESSAGE
#define WARN_THROWS_MESSAGE DOCTEST_WARN_THROWS_MESSAGE
#define WARN_THROWS_AS_MESSAGE DOCTEST_WARN_THROWS_AS_MESSAGE
#define WARN_NOTHROW_MESSAGE DOCTEST_WARN_NOTHROW_MESSAGE
#define CHECK_MESSAGE DOCTEST_CHECK_MESSAGE
#define CHECK_FALSE_MESSAGE DOCTEST_CHECK_FALSE_MESSAGE
#define CHECK_THROWS_MESSAGE DOCTEST_CHECK_THROWS_MESSAGE
#define CHECK_THROWS_AS_MESSAGE DOCTEST_CHECK_THROWS_AS_MESSAGE
#define CHECK_NOTHROW_MESSAGE DOCTEST_CHECK_NOTHROW_MESSAGE
#define REQUIRE_MESSAGE DOCTEST_REQUIRE_MESSAGE
#define REQUIRE_FALSE_MESSAGE DOCTEST_REQUIRE_FALSE_MESSAGE
#define REQUIRE_THROWS_MESSAGE DOCTEST_REQUIRE_THROWS_MESSAGE
#define REQUIRE_THROWS_AS_MESSAGE DOCTEST_REQUIRE_THROWS_AS_MESSAGE
#define REQUIRE_NOTHROW_MESSAGE DOCTEST_REQUIRE_NOTHROW_MESSAGE

#define SCENARIO DOCTEST_SCENARIO
#define GIVEN DOCTEST_GIVEN
#define WHEN DOCTEST_WHEN
#define AND_WHEN DOCTEST_AND_WHEN
#define THEN DOCTEST_THEN
#define AND_THEN DOCTEST_AND_THEN

#define WARN_EQ DOCTEST_WARN_EQ
#define CHECK_EQ DOCTEST_CHECK_EQ
#define REQUIRE_EQ DOCTEST_REQUIRE_EQ
#define WARN_NE DOCTEST_WARN_NE
#define CHECK_NE DOCTEST_CHECK_NE
#define REQUIRE_NE DOCTEST_REQUIRE_NE
#define WARN_GT DOCTEST_WARN_GT
#define CHECK_GT DOCTEST_CHECK_GT
#define REQUIRE_GT DOCTEST_REQUIRE_GT
#define WARN_LT DOCTEST_WARN_LT
#define CHECK_LT DOCTEST_CHECK_LT
#define REQUIRE_LT DOCTEST_REQUIRE_LT
#define WARN_GE DOCTEST_WARN_GE
#define CHECK_GE DOCTEST_CHECK_GE
#define REQUIRE_GE DOCTEST_REQUIRE_GE
#define WARN_LE DOCTEST_WARN_LE
#define CHECK_LE DOCTEST_CHECK_LE
#define REQUIRE_LE DOCTEST_REQUIRE_LE
#define WARN_UNARY DOCTEST_WARN_UNARY
#define CHECK_UNARY DOCTEST_CHECK_UNARY
#define REQUIRE_UNARY DOCTEST_REQUIRE_UNARY
#define WARN_UNARY_FALSE DOCTEST_WARN_UNARY_FALSE
#define CHECK_UNARY_FALSE DOCTEST_CHECK_UNARY_FALSE
#define REQUIRE_UNARY_FALSE DOCTEST_REQUIRE_UNARY_FALSE

#define FAST_WARN_EQ DOCTEST_FAST_WARN_EQ
#define FAST_CHECK_EQ DOCTEST_FAST_CHECK_EQ
#define FAST_REQUIRE_EQ DOCTEST_FAST_REQUIRE_EQ
#define FAST_WARN_NE DOCTEST_FAST_WARN_NE
#define FAST_CHECK_NE DOCTEST_FAST_CHECK_NE
#define FAST_REQUIRE_NE DOCTEST_FAST_REQUIRE_NE
#define FAST_WARN_GT DOCTEST_FAST_WARN_GT
#define FAST_CHECK_GT DOCTEST_FAST_CHECK_GT
#define FAST_REQUIRE_GT DOCTEST_FAST_REQUIRE_GT
#define FAST_WARN_LT DOCTEST_FAST_WARN_LT
#define FAST_CHECK_LT DOCTEST_FAST_CHECK_LT
#define FAST_REQUIRE_LT DOCTEST_FAST_REQUIRE_LT
#define FAST_WARN_GE DOCTEST_FAST_WARN_GE
#define FAST_CHECK_GE DOCTEST_FAST_CHECK_GE
#define FAST_REQUIRE_GE DOCTEST_FAST_REQUIRE_GE
#define FAST_WARN_LE DOCTEST_FAST_WARN_LE
#define FAST_CHECK_LE DOCTEST_FAST_CHECK_LE
#define FAST_REQUIRE_LE DOCTEST_FAST_REQUIRE_LE
#define FAST_WARN_UNARY DOCTEST_FAST_WARN_UNARY
#define FAST_CHECK_UNARY DOCTEST_FAST_CHECK_UNARY
#define FAST_REQUIRE_UNARY DOCTEST_FAST_REQUIRE_UNARY
#define FAST_WARN_UNARY_FALSE DOCTEST_FAST_WARN_UNARY_FALSE
#define FAST_CHECK_UNARY_FALSE DOCTEST_FAST_CHECK_UNARY_FALSE
#define FAST_REQUIRE_UNARY_FALSE DOCTEST_FAST_REQUIRE_UNARY_FALSE

#endif 

DOCTEST_TEST_SUITE_END();

namespace doctest
{
namespace detail
{
DOCTEST_TYPE_TO_STRING_IMPL(bool)
DOCTEST_TYPE_TO_STRING_IMPL(float)
DOCTEST_TYPE_TO_STRING_IMPL(double)
DOCTEST_TYPE_TO_STRING_IMPL(long double)
DOCTEST_TYPE_TO_STRING_IMPL(char)
DOCTEST_TYPE_TO_STRING_IMPL(signed char)
DOCTEST_TYPE_TO_STRING_IMPL(unsigned char)
DOCTEST_TYPE_TO_STRING_IMPL(wchar_t)
DOCTEST_TYPE_TO_STRING_IMPL(short int)
DOCTEST_TYPE_TO_STRING_IMPL(unsigned short int)
DOCTEST_TYPE_TO_STRING_IMPL(int)
DOCTEST_TYPE_TO_STRING_IMPL(unsigned int)
DOCTEST_TYPE_TO_STRING_IMPL(long int)
DOCTEST_TYPE_TO_STRING_IMPL(unsigned long int)
#ifdef DOCTEST_CONFIG_WITH_LONG_LONG
DOCTEST_TYPE_TO_STRING_IMPL(long long int)
DOCTEST_TYPE_TO_STRING_IMPL(unsigned long long int)
#endif 
} 
} 

#endif 

#if defined(__clang__)
#pragma clang diagnostic pop
#endif 

#if defined(__GNUC__) && !defined(__clang__)
#if __GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ > 6)
#pragma GCC diagnostic pop
#endif 
#endif 

#ifdef _MSC_VER
#pragma warning(pop)
#endif 

#ifndef DOCTEST_SINGLE_HEADER
#define DOCTEST_SINGLE_HEADER
#endif 

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-pragmas"
#pragma clang diagnostic ignored "-Wpadded"
#pragma clang diagnostic ignored "-Wglobal-constructors"
#pragma clang diagnostic ignored "-Wexit-time-destructors"
#pragma clang diagnostic ignored "-Wmissing-prototypes"
#pragma clang diagnostic ignored "-Wsign-conversion"
#pragma clang diagnostic ignored "-Wshorten-64-to-32"
#pragma clang diagnostic ignored "-Wmissing-variable-declarations"
#pragma clang diagnostic ignored "-Wswitch"
#pragma clang diagnostic ignored "-Wswitch-enum"
#pragma clang diagnostic ignored "-Wcovered-switch-default"
#pragma clang diagnostic ignored "-Wmissing-noreturn"
#pragma clang diagnostic ignored "-Wunused-local-typedef"
#pragma clang diagnostic ignored "-Wdisabled-macro-expansion"
#pragma clang diagnostic ignored "-Wmissing-braces"
#pragma clang diagnostic ignored "-Wmissing-field-initializers"
#pragma clang diagnostic ignored "-Wzero-as-null-pointer-constant"
#pragma clang diagnostic ignored "-Wc++11-long-long"
#endif 

#if defined(__GNUC__) && !defined(__clang__)
#if __GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ > 6)
#pragma GCC diagnostic push
#endif 
#pragma GCC diagnostic ignored "-Wunknown-pragmas"
#pragma GCC diagnostic ignored "-Wconversion"
#pragma GCC diagnostic ignored "-Weffc++"
#pragma GCC diagnostic ignored "-Wsign-conversion"
#pragma GCC diagnostic ignored "-Wstrict-overflow"
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#pragma GCC diagnostic ignored "-Wmissing-braces"
#pragma GCC diagnostic ignored "-Wmissing-declarations"
#pragma GCC diagnostic ignored "-Winline"
#pragma GCC diagnostic ignored "-Wswitch"
#pragma GCC diagnostic ignored "-Wswitch-enum"
#pragma GCC diagnostic ignored "-Wswitch-default"
#pragma GCC diagnostic ignored "-Wunsafe-loop-optimizations"
#pragma GCC diagnostic ignored "-Wlong-long"
#pragma GCC diagnostic ignored "-Wold-style-cast"
#if __GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ > 6)
#pragma GCC diagnostic ignored "-Wzero-as-null-pointer-constant"
#endif 
#if __GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ > 7)
#pragma GCC diagnostic ignored "-Wunused-local-typedefs"
#endif 
#if __GNUC__ > 5 || (__GNUC__ == 5 && __GNUC_MINOR__ > 3)
#pragma GCC diagnostic ignored "-Wuseless-cast"
#endif 
#endif 

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4996) 
#pragma warning(disable : 4267) 
#pragma warning(disable : 4706) 
#pragma warning(disable : 4512) 
#pragma warning(disable : 4127) 
#pragma warning(disable : 4530) 
#pragma warning(disable : 4577) 
#endif                          

#if defined(DOCTEST_CONFIG_IMPLEMENT) || !defined(DOCTEST_SINGLE_HEADER)
#ifndef DOCTEST_LIBRARY_IMPLEMENTATION
#define DOCTEST_LIBRARY_IMPLEMENTATION

#ifndef DOCTEST_SINGLE_HEADER
#include "doctest_fwd.h"
#endif 

#if defined(__clang__) && defined(DOCTEST_NO_CPP11_COMPAT)
#pragma clang diagnostic ignored "-Wc++98-compat"
#pragma clang diagnostic ignored "-Wc++98-compat-pedantic"
#endif 

#ifdef _MSC_VER
#define DOCTEST_SNPRINTF _snprintf
#else
#define DOCTEST_SNPRINTF std::snprintf
#endif

#define DOCTEST_LOG_START()                                                                        \
do {                                                                                           \
if(!contextState->hasLoggedCurrentTestStart) {                                             \
doctest::detail::logTestStart(*contextState->currentTest);                             \
contextState->hasLoggedCurrentTestStart = true;                                        \
}                                                                                          \
} while(false)

#include <ctime>
#include <cmath>
#ifdef __BORLANDC__
#include <math.h>
#endif 
#include <new>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <utility>
#include <sstream>
#include <iomanip>
#include <vector>
#include <set>
#include <exception>
#include <stdexcept>
#include <csignal>
#include <cfloat>
#ifndef _MSC_VER
#include <stdint.h>
#endif 

namespace doctest
{
namespace detail
{
char tolower(const char c) { return (c >= 'A' && c <= 'Z') ? static_cast<char>(c + 32) : c; }

template <typename T>
T my_max(const T& lhs, const T& rhs) {
return lhs > rhs ? lhs : rhs;
}

int stricmp(char const* a, char const* b) {
for(;; a++, b++) {
int d = tolower(*a) - tolower(*b);
if(d != 0 || !*a)
return d;
}
}

void my_memcpy(void* dest, const void* src, unsigned num) {
const char* csrc  = static_cast<const char*>(src);
char*       cdest = static_cast<char*>(dest);
for(unsigned i = 0; i < num; ++i)
cdest[i]   = csrc[i];
}

unsigned my_strlen(const char* in) {
const char* temp = in;
while(temp && *temp)
++temp;
return unsigned(temp - in);
}

template <typename T>
String fpToString(T value, int precision) {
std::ostringstream oss;
oss << std::setprecision(precision) << std::fixed << value;
std::string d = oss.str();
size_t      i = d.find_last_not_of('0');
if(i != std::string::npos && i != d.size() - 1) {
if(d[i] == '.')
i++;
d = d.substr(0, i + 1);
}
return d.c_str();
}

struct Endianness
{
enum Arch
{
Big,
Little
};

static Arch which() {
union _
{
int  asInt;
char asChar[sizeof(int)];
} u;

u.asInt = 1;                                            
return (u.asChar[sizeof(int) - 1] == 1) ? Big : Little; 
}
};

String rawMemoryToString(const void* object, unsigned size) {
int i = 0, end = static_cast<int>(size), inc = 1;
if(Endianness::which() == Endianness::Little) {
i   = end - 1;
end = inc = -1;
}

unsigned char const* bytes = static_cast<unsigned char const*>(object);
std::ostringstream   os;
os << "0x" << std::setfill('0') << std::hex;
for(; i != end; i += inc)
os << std::setw(2) << static_cast<unsigned>(bytes[i]);
return os.str().c_str();
}

std::ostream* createStream() { return new std::ostringstream(); }
String getStreamResult(std::ostream* in) {
return static_cast<std::ostringstream*>(in)->str().c_str(); 
}
void freeStream(std::ostream* in) { delete in; }

#ifndef DOCTEST_CONFIG_DISABLE

struct ContextState : TestAccessibleContextState 
{

std::vector<std::vector<String> > filters;

String   order_by;  
unsigned rand_seed; 

unsigned first; 
unsigned last;  

int  abort_after;           
int  subcase_filter_levels; 
bool case_sensitive;        
bool exit;         
bool duration;     
bool no_exitcode;  
bool no_run;       
bool no_version;   
bool no_colors;    
bool force_colors; 
bool no_breaks;    
bool no_skip;      
bool no_path_in_filenames; 
bool no_line_numbers;      
bool no_skipped_summary;   

bool help;             
bool version;          
bool count;            
bool list_test_cases;  
bool list_test_suites; 


unsigned        numTestsPassingFilters;
unsigned        numTestSuitesPassingFilters;
unsigned        numFailed;
const TestCase* currentTest;
bool            hasLoggedCurrentTestStart;
int             numAssertionsForCurrentTestcase;
int             numAssertions;
int             numFailedAssertionsForCurrentTestcase;
int             numFailedAssertions;
bool            hasCurrentTestFailed;

std::vector<IContextScope*> contexts;            
std::vector<std::string>    exceptionalContexts; 

std::set<SubcaseSignature> subcasesPassed;
std::set<int>              subcasesEnteredLevels;
std::vector<Subcase>       subcasesStack;
int                        subcasesCurrentLevel;
bool                       subcasesHasSkipped;

void resetRunData() {
numTestsPassingFilters                = 0;
numTestSuitesPassingFilters           = 0;
numFailed                             = 0;
numAssertions                         = 0;
numFailedAssertions                   = 0;
numFailedAssertionsForCurrentTestcase = 0;
}

ContextState()
: filters(8) 
{
resetRunData();
}
};

ContextState* contextState = 0;
#endif 
} 

void String::copy(const String& other) {
if(other.isOnStack()) {
detail::my_memcpy(buf, other.buf, len);
} else {
setOnHeap();
data.size     = other.data.size;
data.capacity = data.size + 1;
data.ptr      = new char[data.capacity];
detail::my_memcpy(data.ptr, other.data.ptr, data.size + 1);
}
}

String::String(const char* in) {
unsigned in_len = detail::my_strlen(in);
if(in_len <= last) {
detail::my_memcpy(buf, in, in_len + 1);
setLast(last - in_len);
} else {
setOnHeap();
data.size     = in_len;
data.capacity = data.size + 1;
data.ptr      = new char[data.capacity];
detail::my_memcpy(data.ptr, in, in_len + 1);
}
}

String& String::operator+=(const String& other) {
unsigned my_old_size = size();
unsigned other_size  = other.size();
unsigned total_size  = my_old_size + other_size;
if(isOnStack()) {
if(total_size < len) {
detail::my_memcpy(buf + my_old_size, other.c_str(), other_size + 1);
setLast(last - total_size);
} else {
char* temp = new char[total_size + 1];
detail::my_memcpy(temp, buf, my_old_size); 
setOnHeap();
data.size     = total_size;
data.capacity = data.size + 1;
data.ptr      = temp;
detail::my_memcpy(data.ptr + my_old_size, other.c_str(), other_size + 1);
}
} else {
if(data.capacity > total_size) {
data.size = total_size;
detail::my_memcpy(data.ptr + my_old_size, other.c_str(), other_size + 1);
} else {
data.capacity *= 2;
if(data.capacity <= total_size)
data.capacity = total_size + 1;
char* temp = new char[data.capacity];
detail::my_memcpy(temp, data.ptr, my_old_size); 
delete[] data.ptr;
data.size = total_size;
data.ptr  = temp;
detail::my_memcpy(data.ptr + my_old_size, other.c_str(), other_size + 1);
}
}

return *this;
}

#ifdef DOCTEST_CONFIG_WITH_RVALUE_REFERENCES
String::String(String&& other) {
detail::my_memcpy(buf, other.buf, len);
other.buf[0] = '\0';
other.setLast();
}

String& String::operator=(String&& other) {
if(!isOnStack())
delete[] data.ptr;
detail::my_memcpy(buf, other.buf, len);
other.buf[0] = '\0';
other.setLast();
return *this;
}
#endif 

int String::compare(const char* other, bool no_case) const {
if(no_case)
return detail::stricmp(c_str(), other);
return std::strcmp(c_str(), other);
}

int String::compare(const String& other, bool no_case) const {
return compare(other.c_str(), no_case);
}

std::ostream& operator<<(std::ostream& stream, const String& in) {
stream << in.c_str();
return stream;
}

Approx::Approx(double value)
: m_epsilon(static_cast<double>(std::numeric_limits<float>::epsilon()) * 100)
, m_scale(1.0)
, m_value(value) {}

bool operator==(double lhs, Approx const& rhs) {
return std::fabs(lhs - rhs.m_value) <
rhs.m_epsilon * (rhs.m_scale + detail::my_max(std::fabs(lhs), std::fabs(rhs.m_value)));
}

String Approx::toString() const { return String("Approx( ") + doctest::toString(m_value) + " )"; }

#ifdef DOCTEST_CONFIG_TREAT_CHAR_STAR_AS_STRING
String toString(char* in) { return toString(static_cast<const char*>(in)); }
String toString(const char* in) { return String("\"") + (in ? in : "{null string}") + "\""; }
#endif 
String toString(bool in) { return in ? "true" : "false"; }
String toString(float in) { return detail::fpToString(in, 5) + "f"; }
String toString(double in) { return detail::fpToString(in, 10); }
String toString(double long in) { return detail::fpToString(in, 15); }

String toString(char in) {
char buf[64];
std::sprintf(buf, "%d", in);
return buf;
}

String toString(char signed in) {
char buf[64];
std::sprintf(buf, "%d", in);
return buf;
}

String toString(char unsigned in) {
char buf[64];
std::sprintf(buf, "%ud", in);
return buf;
}

String toString(int short in) {
char buf[64];
std::sprintf(buf, "%d", in);
return buf;
}

String toString(int short unsigned in) {
char buf[64];
std::sprintf(buf, "%u", in);
return buf;
}

String toString(int in) {
char buf[64];
std::sprintf(buf, "%d", in);
return buf;
}

String toString(int unsigned in) {
char buf[64];
std::sprintf(buf, "%u", in);
return buf;
}

String toString(int long in) {
char buf[64];
std::sprintf(buf, "%ld", in);
return buf;
}

String toString(int long unsigned in) {
char buf[64];
std::sprintf(buf, "%lu", in);
return buf;
}

#ifdef DOCTEST_CONFIG_WITH_LONG_LONG
String toString(int long long in) {
char buf[64];
std::sprintf(buf, "%lld", in);
return buf;
}
String toString(int long long unsigned in) {
char buf[64];
std::sprintf(buf, "%llu", in);
return buf;
}
#endif 

#ifdef DOCTEST_CONFIG_WITH_NULLPTR
String toString(std::nullptr_t) { return "nullptr"; }
#endif 

} 

#ifdef DOCTEST_CONFIG_DISABLE
namespace doctest
{
bool isRunningInTest() { return false; }
Context::Context(int, const char* const*) {}
Context::~Context() {}
void Context::applyCommandLine(int, const char* const*) {}
void Context::addFilter(const char*, const char*) {}
void Context::clearFilters() {}
void Context::setOption(const char*, int) {}
void Context::setOption(const char*, const char*) {}
bool Context::shouldExit() { return false; }
int  Context::run() { return 0; }
} 
#else 

#if !defined(DOCTEST_CONFIG_COLORS_NONE)
#if !defined(DOCTEST_CONFIG_COLORS_WINDOWS) && !defined(DOCTEST_CONFIG_COLORS_ANSI)
#ifdef DOCTEST_PLATFORM_WINDOWS
#define DOCTEST_CONFIG_COLORS_WINDOWS
#else 
#define DOCTEST_CONFIG_COLORS_ANSI
#endif 
#endif 
#endif 

#define DOCTEST_PRINTF_COLORED(buffer, color)                                                      \
do {                                                                                           \
doctest::detail::Color col(color);                                                         \
std::printf("%s", buffer);                                                                 \
} while((void)0, 0)

#if !defined(DOCTEST_SNPRINTF_BUFFER_LENGTH)
#define DOCTEST_SNPRINTF_BUFFER_LENGTH 1024
#endif 

#if defined(_MSC_VER) || defined(__MINGW32__)
#if defined(_MSC_VER) && _MSC_VER >= 1700
#define DOCTEST_WINDOWS_SAL_IN_OPT _In_opt_
#else 
#define DOCTEST_WINDOWS_SAL_IN_OPT
#endif 
extern "C" __declspec(dllimport) void __stdcall OutputDebugStringA(
DOCTEST_WINDOWS_SAL_IN_OPT const char*);
extern "C" __declspec(dllimport) int __stdcall IsDebuggerPresent();
#endif 

#ifdef DOCTEST_CONFIG_COLORS_ANSI
#include <unistd.h>
#endif 

#ifdef _WIN32

#ifndef WIN32_MEAN_AND_LEAN
#define WIN32_MEAN_AND_LEAN
#endif 
#ifndef VC_EXTRA_LEAN
#define VC_EXTRA_LEAN
#endif 
#ifndef NOMINMAX
#define NOMINMAX
#endif 

#ifdef __AFXDLL
#include <AfxWin.h>
#else
#include <windows.h>
#endif
#include <io.h>

#else 

#include <sys/time.h>

#endif 

namespace doctest_detail_test_suite_ns
{
doctest::detail::TestSuite& getCurrentTestSuite() {
static doctest::detail::TestSuite data;
return data;
}
} 

namespace doctest
{
namespace detail
{
TestCase::TestCase(funcType test, const char* file, unsigned line, const TestSuite& test_suite,
const char* type, int template_id)
: m_test(test)
, m_name(0)
, m_type(type)
, m_test_suite(test_suite.m_test_suite)
, m_description(test_suite.m_description)
, m_skip(test_suite.m_skip)
, m_may_fail(test_suite.m_may_fail)
, m_should_fail(test_suite.m_should_fail)
, m_expected_failures(test_suite.m_expected_failures)
, m_timeout(test_suite.m_timeout)
, m_file(file)
, m_line(line)
, m_template_id(template_id) {}

TestCase& TestCase::operator*(const char* in) {
m_name = in;
if(m_template_id != -1) {
m_full_name = String(m_name) + m_type;
m_name = m_full_name.c_str();
}
return *this;
}

TestCase& TestCase::operator=(const TestCase& other) {
m_test              = other.m_test;
m_full_name         = other.m_full_name;
m_name              = other.m_name;
m_type              = other.m_type;
m_test_suite        = other.m_test_suite;
m_description       = other.m_description;
m_skip              = other.m_skip;
m_may_fail          = other.m_may_fail;
m_should_fail       = other.m_should_fail;
m_expected_failures = other.m_expected_failures;
m_timeout           = other.m_timeout;
m_file              = other.m_file;
m_line              = other.m_line;
m_template_id       = other.m_template_id;

if(m_template_id != -1)
m_name = m_full_name.c_str();
return *this;
}

bool TestCase::operator<(const TestCase& other) const {
if(m_line != other.m_line)
return m_line < other.m_line;
int file_cmp = std::strcmp(m_file, other.m_file);
if(file_cmp != 0)
return file_cmp < 0;
return m_template_id < other.m_template_id;
}

const char* getAssertString(assertType::Enum val) {
switch(val) { 
case assertType::DT_WARN                    : return "WARN";
case assertType::DT_CHECK                   : return "CHECK";
case assertType::DT_REQUIRE                 : return "REQUIRE";

case assertType::DT_WARN_FALSE              : return "WARN_FALSE";
case assertType::DT_CHECK_FALSE             : return "CHECK_FALSE";
case assertType::DT_REQUIRE_FALSE           : return "REQUIRE_FALSE";

case assertType::DT_WARN_THROWS             : return "WARN_THROWS";
case assertType::DT_CHECK_THROWS            : return "CHECK_THROWS";
case assertType::DT_REQUIRE_THROWS          : return "REQUIRE_THROWS";

case assertType::DT_WARN_THROWS_AS          : return "WARN_THROWS_AS";
case assertType::DT_CHECK_THROWS_AS         : return "CHECK_THROWS_AS";
case assertType::DT_REQUIRE_THROWS_AS       : return "REQUIRE_THROWS_AS";

case assertType::DT_WARN_NOTHROW            : return "WARN_NOTHROW";
case assertType::DT_CHECK_NOTHROW           : return "CHECK_NOTHROW";
case assertType::DT_REQUIRE_NOTHROW         : return "REQUIRE_NOTHROW";

case assertType::DT_WARN_EQ                 : return "WARN_EQ";
case assertType::DT_CHECK_EQ                : return "CHECK_EQ";
case assertType::DT_REQUIRE_EQ              : return "REQUIRE_EQ";
case assertType::DT_WARN_NE                 : return "WARN_NE";
case assertType::DT_CHECK_NE                : return "CHECK_NE";
case assertType::DT_REQUIRE_NE              : return "REQUIRE_NE";
case assertType::DT_WARN_GT                 : return "WARN_GT";
case assertType::DT_CHECK_GT                : return "CHECK_GT";
case assertType::DT_REQUIRE_GT              : return "REQUIRE_GT";
case assertType::DT_WARN_LT                 : return "WARN_LT";
case assertType::DT_CHECK_LT                : return "CHECK_LT";
case assertType::DT_REQUIRE_LT              : return "REQUIRE_LT";
case assertType::DT_WARN_GE                 : return "WARN_GE";
case assertType::DT_CHECK_GE                : return "CHECK_GE";
case assertType::DT_REQUIRE_GE              : return "REQUIRE_GE";
case assertType::DT_WARN_LE                 : return "WARN_LE";
case assertType::DT_CHECK_LE                : return "CHECK_LE";
case assertType::DT_REQUIRE_LE              : return "REQUIRE_LE";

case assertType::DT_WARN_UNARY              : return "WARN_UNARY";
case assertType::DT_CHECK_UNARY             : return "CHECK_UNARY";
case assertType::DT_REQUIRE_UNARY           : return "REQUIRE_UNARY";
case assertType::DT_WARN_UNARY_FALSE        : return "WARN_UNARY_FALSE";
case assertType::DT_CHECK_UNARY_FALSE       : return "CHECK_UNARY_FALSE";
case assertType::DT_REQUIRE_UNARY_FALSE     : return "REQUIRE_UNARY_FALSE";

case assertType::DT_FAST_WARN_EQ            : return "FAST_WARN_EQ";
case assertType::DT_FAST_CHECK_EQ           : return "FAST_CHECK_EQ";
case assertType::DT_FAST_REQUIRE_EQ         : return "FAST_REQUIRE_EQ";
case assertType::DT_FAST_WARN_NE            : return "FAST_WARN_NE";
case assertType::DT_FAST_CHECK_NE           : return "FAST_CHECK_NE";
case assertType::DT_FAST_REQUIRE_NE         : return "FAST_REQUIRE_NE";
case assertType::DT_FAST_WARN_GT            : return "FAST_WARN_GT";
case assertType::DT_FAST_CHECK_GT           : return "FAST_CHECK_GT";
case assertType::DT_FAST_REQUIRE_GT         : return "FAST_REQUIRE_GT";
case assertType::DT_FAST_WARN_LT            : return "FAST_WARN_LT";
case assertType::DT_FAST_CHECK_LT           : return "FAST_CHECK_LT";
case assertType::DT_FAST_REQUIRE_LT         : return "FAST_REQUIRE_LT";
case assertType::DT_FAST_WARN_GE            : return "FAST_WARN_GE";
case assertType::DT_FAST_CHECK_GE           : return "FAST_CHECK_GE";
case assertType::DT_FAST_REQUIRE_GE         : return "FAST_REQUIRE_GE";
case assertType::DT_FAST_WARN_LE            : return "FAST_WARN_LE";
case assertType::DT_FAST_CHECK_LE           : return "FAST_CHECK_LE";
case assertType::DT_FAST_REQUIRE_LE         : return "FAST_REQUIRE_LE";

case assertType::DT_FAST_WARN_UNARY         : return "FAST_WARN_UNARY";
case assertType::DT_FAST_CHECK_UNARY        : return "FAST_CHECK_UNARY";
case assertType::DT_FAST_REQUIRE_UNARY      : return "FAST_REQUIRE_UNARY";
case assertType::DT_FAST_WARN_UNARY_FALSE   : return "FAST_WARN_UNARY_FALSE";
case assertType::DT_FAST_CHECK_UNARY_FALSE  : return "FAST_CHECK_UNARY_FALSE";
case assertType::DT_FAST_REQUIRE_UNARY_FALSE: return "FAST_REQUIRE_UNARY_FALSE";
}
return "";
}

bool checkIfShouldThrow(assertType::Enum assert_type) {
if(assert_type & assertType::is_require) 
return true;

if((assert_type & assertType::is_check) 
&& contextState->abort_after > 0 &&
contextState->numFailedAssertions >= contextState->abort_after)
return true;

return false;
}
void fastAssertThrowIfFlagSet(int flags) {
if(flags & assertAction::shouldthrow) 
throwException();
}
void throwException() {
#ifndef DOCTEST_CONFIG_NO_EXCEPTIONS
throw TestFailureException();
#endif 
}

int wildcmp(const char* str, const char* wild, bool caseSensitive) {
const char* cp = 0;
const char* mp = 0;

while((*str) && (*wild != '*')) {
if((caseSensitive ? (*wild != *str) : (tolower(*wild) != tolower(*str))) &&
(*wild != '?')) {
return 0;
}
wild++;
str++;
}

while(*str) {
if(*wild == '*') {
if(!*++wild) {
return 1;
}
mp = wild;
cp = str + 1;
} else if((caseSensitive ? (*wild == *str) : (tolower(*wild) == tolower(*str))) ||
(*wild == '?')) {
wild++;
str++;
} else {
wild = mp;   
str  = cp++; 
}
}

while(*wild == '*') {
wild++;
}
return !*wild;
}


bool matchesAny(const char* name, const std::vector<String>& filters, int matchEmpty,
bool caseSensitive) {
if(filters.empty() && matchEmpty)
return true;
for(unsigned i = 0; i < filters.size(); ++i)
if(wildcmp(name, filters[i].c_str(), caseSensitive))
return true;
return false;
}

#ifdef _WIN32

typedef unsigned long long UInt64;

UInt64 getCurrentTicks() {
static UInt64 hz = 0, hzo = 0;
if(!hz) {
QueryPerformanceFrequency(reinterpret_cast<LARGE_INTEGER*>(&hz));
QueryPerformanceCounter(reinterpret_cast<LARGE_INTEGER*>(&hzo));
}
UInt64 t;
QueryPerformanceCounter(reinterpret_cast<LARGE_INTEGER*>(&t));
return ((t - hzo) * 1000000) / hz;
}
#else  

typedef uint64_t UInt64;

UInt64 getCurrentTicks() {
timeval t;
gettimeofday(&t, 0);
return static_cast<UInt64>(t.tv_sec) * 1000000 + static_cast<UInt64>(t.tv_usec);
}
#endif 

class Timer
{
public:
Timer()
: m_ticks(0) {}
void         start() { m_ticks = getCurrentTicks(); }
unsigned int getElapsedMicroseconds() const {
return static_cast<unsigned int>(getCurrentTicks() - m_ticks);
}
unsigned int getElapsedMilliseconds() const {
return static_cast<unsigned int>(getElapsedMicroseconds() / 1000);
}
double getElapsedSeconds() const { return getElapsedMicroseconds() / 1000000.0; }

private:
UInt64 m_ticks;
};

TestAccessibleContextState* getTestsContextState() { return contextState; }

bool SubcaseSignature::operator<(const SubcaseSignature& other) const {
if(m_line != other.m_line)
return m_line < other.m_line;
if(std::strcmp(m_file, other.m_file) != 0)
return std::strcmp(m_file, other.m_file) < 0;
return std::strcmp(m_name, other.m_name) < 0;
}

Subcase::Subcase(const char* name, const char* file, int line)
: m_signature(name, file, line)
, m_entered(false) {
ContextState* s = contextState;

if(s->subcasesPassed.count(m_signature) != 0)
return;

if(s->subcasesCurrentLevel < s->subcase_filter_levels) {
if(!matchesAny(m_signature.m_name, s->filters[6], 1, s->case_sensitive))
return;
if(matchesAny(m_signature.m_name, s->filters[7], 0, s->case_sensitive))
return;
}

if(s->subcasesEnteredLevels.count(s->subcasesCurrentLevel) != 0) {
s->subcasesHasSkipped = true;
return;
}

s->subcasesStack.push_back(*this);
if(s->hasLoggedCurrentTestStart)
logTestEnd();
s->hasLoggedCurrentTestStart = false;

s->subcasesEnteredLevels.insert(s->subcasesCurrentLevel++);
m_entered = true;
}

Subcase::Subcase(const Subcase& other)
: m_signature(other.m_signature.m_name, other.m_signature.m_file,
other.m_signature.m_line)
, m_entered(other.m_entered) {}

Subcase::~Subcase() {
if(m_entered) {
ContextState* s = contextState;

s->subcasesCurrentLevel--;
if(s->subcasesHasSkipped == false)
s->subcasesPassed.insert(m_signature);

if(!s->subcasesStack.empty())
s->subcasesStack.pop_back();
if(s->hasLoggedCurrentTestStart)
logTestEnd();
s->hasLoggedCurrentTestStart = false;
}
}

Result::~Result() {}

Result& Result::operator=(const Result& other) {
m_passed        = other.m_passed;
m_decomposition = other.m_decomposition;

return *this;
}

int fileOrderComparator(const void* a, const void* b) {
const TestCase* lhs = *static_cast<TestCase* const*>(a);
const TestCase* rhs = *static_cast<TestCase* const*>(b);
#ifdef _MSC_VER
int res = stricmp(lhs->m_file, rhs->m_file);
#else  
int res = std::strcmp(lhs->m_file, rhs->m_file);
#endif 
if(res != 0)
return res;
return static_cast<int>(lhs->m_line - rhs->m_line);
}

int suiteOrderComparator(const void* a, const void* b) {
const TestCase* lhs = *static_cast<TestCase* const*>(a);
const TestCase* rhs = *static_cast<TestCase* const*>(b);

int res = std::strcmp(lhs->m_test_suite, rhs->m_test_suite);
if(res != 0)
return res;
return fileOrderComparator(a, b);
}

int nameOrderComparator(const void* a, const void* b) {
const TestCase* lhs = *static_cast<TestCase* const*>(a);
const TestCase* rhs = *static_cast<TestCase* const*>(b);

int res_name = std::strcmp(lhs->m_name, rhs->m_name);
if(res_name != 0)
return res_name;
return suiteOrderComparator(a, b);
}

int setTestSuite(const TestSuite& ts) {
doctest_detail_test_suite_ns::getCurrentTestSuite() = ts;
return 0;
}

std::set<TestCase>& getRegisteredTests() {
static std::set<TestCase> data;
return data;
}

int regTest(const TestCase& tc) {
getRegisteredTests().insert(tc);
return 0;
}

struct Color
{
enum Code
{
None = 0,
White,
Red,
Green,
Blue,
Cyan,
Yellow,
Grey,

Bright = 0x10,

BrightRed   = Bright | Red,
BrightGreen = Bright | Green,
LightGrey   = Bright | Grey,
BrightWhite = Bright | White
};
explicit Color(Code code) { use(code); }
~Color() { use(None); }

static void use(Code code);
static void init();
};

#ifdef DOCTEST_CONFIG_COLORS_WINDOWS
HANDLE g_stdoutHandle;
WORD   g_originalForegroundAttributes;
WORD   g_originalBackgroundAttributes;
bool   g_attrsInitted = false;
#endif 

void Color::init() {
#ifdef DOCTEST_CONFIG_COLORS_WINDOWS
if(!g_attrsInitted) {
g_stdoutHandle = GetStdHandle(STD_OUTPUT_HANDLE);
g_attrsInitted = true;
CONSOLE_SCREEN_BUFFER_INFO csbiInfo;
GetConsoleScreenBufferInfo(g_stdoutHandle, &csbiInfo);
g_originalForegroundAttributes =
csbiInfo.wAttributes &
~(BACKGROUND_GREEN | BACKGROUND_RED | BACKGROUND_BLUE | BACKGROUND_INTENSITY);
g_originalBackgroundAttributes =
csbiInfo.wAttributes &
~(FOREGROUND_GREEN | FOREGROUND_RED | FOREGROUND_BLUE | FOREGROUND_INTENSITY);
}
#endif 
}

void Color::use(Code
#ifndef DOCTEST_CONFIG_COLORS_NONE
code
#endif 
) {
const ContextState* p = contextState;
if(p->no_colors)
return;
#ifdef DOCTEST_CONFIG_COLORS_ANSI
if(isatty(STDOUT_FILENO) == false && p->force_colors == false)
return;

const char* col = "";
switch(code) { 
case Color::Red:         col = "[0;31m"; break;
case Color::Green:       col = "[0;32m"; break;
case Color::Blue:        col = "[0;34m"; break;
case Color::Cyan:        col = "[0;36m"; break;
case Color::Yellow:      col = "[0;33m"; break;
case Color::Grey:        col = "[1;30m"; break;
case Color::LightGrey:   col = "[0;37m"; break;
case Color::BrightRed:   col = "[1;31m"; break;
case Color::BrightGreen: col = "[1;32m"; break;
case Color::BrightWhite: col = "[1;37m"; break;
case Color::Bright: 
case Color::None:
case Color::White:
default:                 col = "[0m";
}
std::printf("\033%s", col);
#endif 

#ifdef DOCTEST_CONFIG_COLORS_WINDOWS
if(isatty(fileno(stdout)) == false && p->force_colors == false)
return;

#define DOCTEST_SET_ATTR(x)                                                                        \
SetConsoleTextAttribute(g_stdoutHandle, x | g_originalBackgroundAttributes)

switch (code) {
case Color::White:       DOCTEST_SET_ATTR(FOREGROUND_GREEN | FOREGROUND_RED | FOREGROUND_BLUE); break;
case Color::Red:         DOCTEST_SET_ATTR(FOREGROUND_RED);                                      break;
case Color::Green:       DOCTEST_SET_ATTR(FOREGROUND_GREEN);                                    break;
case Color::Blue:        DOCTEST_SET_ATTR(FOREGROUND_BLUE);                                     break;
case Color::Cyan:        DOCTEST_SET_ATTR(FOREGROUND_BLUE | FOREGROUND_GREEN);                  break;
case Color::Yellow:      DOCTEST_SET_ATTR(FOREGROUND_RED | FOREGROUND_GREEN);                   break;
case Color::Grey:        DOCTEST_SET_ATTR(0);                                                   break;
case Color::LightGrey:   DOCTEST_SET_ATTR(FOREGROUND_INTENSITY);                                break;
case Color::BrightRed:   DOCTEST_SET_ATTR(FOREGROUND_INTENSITY | FOREGROUND_RED);               break;
case Color::BrightGreen: DOCTEST_SET_ATTR(FOREGROUND_INTENSITY | FOREGROUND_GREEN);             break;
case Color::BrightWhite: DOCTEST_SET_ATTR(FOREGROUND_INTENSITY | FOREGROUND_GREEN | FOREGROUND_RED | FOREGROUND_BLUE); break;
case Color::None:
case Color::Bright: 
default:                 DOCTEST_SET_ATTR(g_originalForegroundAttributes);
}
#undef DOCTEST_SET_ATTR
#endif 
}

IExceptionTranslator::~IExceptionTranslator() {}

std::vector<const IExceptionTranslator*>& getExceptionTranslators() {
static std::vector<const IExceptionTranslator*> data;
return data;
}

void registerExceptionTranslatorImpl(const IExceptionTranslator* translateFunction) {
getExceptionTranslators().push_back(translateFunction);
}

String translateActiveException() {
#ifndef DOCTEST_CONFIG_NO_EXCEPTIONS
String                                    res;
std::vector<const IExceptionTranslator*>& translators = getExceptionTranslators();
for(size_t i = 0; i < translators.size(); ++i)
if(translators[i]->translate(res))
return res;
try {
throw;
} catch(std::exception& ex) {
return ex.what();
} catch(std::string& msg) {
return msg.c_str();
} catch(const char* msg) {
return msg;
} catch(...) {
return "unknown exception";
}
#else  
return "";
#endif 
}

void writeStringToStream(std::ostream* stream, const String& str) { *stream << str; }

#ifdef DOCTEST_CONFIG_TREAT_CHAR_STAR_AS_STRING
void toStream(std::ostream* stream, char* in) { *stream << in; }
void toStream(std::ostream* stream, const char* in) { *stream << in; }
#endif 
void toStream(std::ostream* stream, bool in) {
*stream << std::boolalpha << in << std::noboolalpha;
}
void toStream(std::ostream* stream, float in) { *stream << in; }
void toStream(std::ostream* stream, double in) { *stream << in; }
void toStream(std::ostream* stream, double long in) { *stream << in; }

void toStream(std::ostream* stream, char in) { *stream << in; }
void toStream(std::ostream* stream, char signed in) { *stream << in; }
void toStream(std::ostream* stream, char unsigned in) { *stream << in; }
void toStream(std::ostream* stream, int short in) { *stream << in; }
void toStream(std::ostream* stream, int short unsigned in) { *stream << in; }
void toStream(std::ostream* stream, int in) { *stream << in; }
void toStream(std::ostream* stream, int unsigned in) { *stream << in; }
void toStream(std::ostream* stream, int long in) { *stream << in; }
void toStream(std::ostream* stream, int long unsigned in) { *stream << in; }

#ifdef DOCTEST_CONFIG_WITH_LONG_LONG
void toStream(std::ostream* stream, int long long in) { *stream << in; }
void toStream(std::ostream* stream, int long long unsigned in) { *stream << in; }
#endif 

void addToContexts(IContextScope* ptr) { contextState->contexts.push_back(ptr); }
void                              popFromContexts() { contextState->contexts.pop_back(); }
void useContextIfExceptionOccurred(IContextScope* ptr) {
if(std::uncaught_exception()) {
std::ostringstream stream;
ptr->build(&stream);
contextState->exceptionalContexts.push_back(stream.str());
}
}

void printSummary();

#if !defined(DOCTEST_CONFIG_POSIX_SIGNALS) && !defined(DOCTEST_CONFIG_WINDOWS_SEH)
void reportFatal(const std::string&) {}
struct FatalConditionHandler
{
void reset() {}
};
#else 

void reportFatal(const std::string& message) {
DOCTEST_LOG_START();

contextState->numAssertions += contextState->numAssertionsForCurrentTestcase;
logTestException(message.c_str(), true);
logTestEnd();
contextState->numFailed++;

printSummary();
}

#ifdef DOCTEST_PLATFORM_WINDOWS

struct SignalDefs
{
DWORD       id;
const char* name;
};
SignalDefs signalDefs[] = {
{EXCEPTION_ILLEGAL_INSTRUCTION, "SIGILL - Illegal instruction signal"},
{EXCEPTION_STACK_OVERFLOW, "SIGSEGV - Stack overflow"},
{EXCEPTION_ACCESS_VIOLATION, "SIGSEGV - Segmentation violation signal"},
{EXCEPTION_INT_DIVIDE_BY_ZERO, "Divide by zero error"},
};

struct FatalConditionHandler
{
static LONG CALLBACK handleVectoredException(PEXCEPTION_POINTERS ExceptionInfo) {
for(size_t i = 0; i < sizeof(signalDefs) / sizeof(SignalDefs); ++i) {
if(ExceptionInfo->ExceptionRecord->ExceptionCode == signalDefs[i].id) {
reportFatal(signalDefs[i].name);
}
}
return EXCEPTION_CONTINUE_SEARCH;
}

FatalConditionHandler() {
isSet = true;
guaranteeSize          = 32 * 1024;
exceptionHandlerHandle = 0;
exceptionHandlerHandle = AddVectoredExceptionHandler(1, handleVectoredException);
SetThreadStackGuarantee(&guaranteeSize);
}

static void reset() {
if(isSet) {
RemoveVectoredExceptionHandler(exceptionHandlerHandle);
SetThreadStackGuarantee(&guaranteeSize);
exceptionHandlerHandle = 0;
isSet                  = false;
}
}

~FatalConditionHandler() { reset(); }

private:
static bool  isSet;
static ULONG guaranteeSize;
static PVOID exceptionHandlerHandle;
};

bool  FatalConditionHandler::isSet                  = false;
ULONG FatalConditionHandler::guaranteeSize          = 0;
PVOID FatalConditionHandler::exceptionHandlerHandle = 0;

#else 

struct SignalDefs
{
int         id;
const char* name;
};
SignalDefs signalDefs[] = {{SIGINT, "SIGINT - Terminal interrupt signal"},
{SIGILL, "SIGILL - Illegal instruction signal"},
{SIGFPE, "SIGFPE - Floating point error signal"},
{SIGSEGV, "SIGSEGV - Segmentation violation signal"},
{SIGTERM, "SIGTERM - Termination request signal"},
{SIGABRT, "SIGABRT - Abort (abnormal termination) signal"}};

struct FatalConditionHandler
{
static bool             isSet;
static struct sigaction oldSigActions[sizeof(signalDefs) / sizeof(SignalDefs)];
static stack_t          oldSigStack;
static char             altStackMem[SIGSTKSZ];

static void handleSignal(int sig) {
std::string name = "<unknown signal>";
for(std::size_t i = 0; i < sizeof(signalDefs) / sizeof(SignalDefs); ++i) {
SignalDefs& def = signalDefs[i];
if(sig == def.id) {
name = def.name;
break;
}
}
reset();
reportFatal(name);
raise(sig);
}

FatalConditionHandler() {
isSet = true;
stack_t sigStack;
sigStack.ss_sp    = altStackMem;
sigStack.ss_size  = SIGSTKSZ;
sigStack.ss_flags = 0;
sigaltstack(&sigStack, &oldSigStack);
struct sigaction sa = {0};

sa.sa_handler = handleSignal; 
sa.sa_flags   = SA_ONSTACK;
for(std::size_t i = 0; i < sizeof(signalDefs) / sizeof(SignalDefs); ++i) {
sigaction(signalDefs[i].id, &sa, &oldSigActions[i]);
}
}

~FatalConditionHandler() { reset(); }
static void reset() {
if(isSet) {
for(std::size_t i = 0; i < sizeof(signalDefs) / sizeof(SignalDefs); ++i) {
sigaction(signalDefs[i].id, &oldSigActions[i], 0);
}
sigaltstack(&oldSigStack, 0);
isSet = false;
}
}
};

bool             FatalConditionHandler::isSet = false;
struct sigaction FatalConditionHandler::oldSigActions[sizeof(signalDefs) / sizeof(SignalDefs)] =
{};
stack_t FatalConditionHandler::oldSigStack           = {};
char    FatalConditionHandler::altStackMem[SIGSTKSZ] = {};

#endif 
#endif 

const char* fileForOutput(const char* file) {
if(contextState->no_path_in_filenames) {
const char* back    = std::strrchr(file, '\\');
const char* forward = std::strrchr(file, '/');
if(back || forward) {
if(back > forward)
forward = back;
return forward + 1;
}
}
return file;
}

int lineForOutput(int line) {
if(contextState->no_line_numbers)
return 0;
return line;
}

#ifdef DOCTEST_PLATFORM_MAC
#include <sys/types.h>
#include <unistd.h>
#include <sys/sysctl.h>
bool isDebuggerActive() {
int        mib[4];
kinfo_proc info;
size_t     size;
info.kp_proc.p_flag = 0;
mib[0] = CTL_KERN;
mib[1] = KERN_PROC;
mib[2] = KERN_PROC_PID;
mib[3] = getpid();
size = sizeof(info);
if(sysctl(mib, sizeof(mib) / sizeof(*mib), &info, &size, 0, 0) != 0) {
fprintf(stderr, "\n** Call to sysctl failed - unable to determine if debugger is "
"active **\n\n");
return false;
}
return ((info.kp_proc.p_flag & P_TRACED) != 0);
}
#elif defined(_MSC_VER) || defined(__MINGW32__)
bool  isDebuggerActive() { return ::IsDebuggerPresent() != 0; }
#else
bool isDebuggerActive() { return false; }
#endif 

#ifdef DOCTEST_PLATFORM_WINDOWS
void myOutputDebugString(const String& text) { ::OutputDebugStringA(text.c_str()); }
#else
void myOutputDebugString(const String&) {}
#endif 

const char* getSeparator() {
return "===============================================================================\n";
}

void printToDebugConsole(const String& text) {
if(isDebuggerActive())
myOutputDebugString(text.c_str());
}

void addFailedAssert(assertType::Enum assert_type) {
if((assert_type & assertType::is_warn) == 0) { 
contextState->numFailedAssertions++;
contextState->numFailedAssertionsForCurrentTestcase++;
contextState->hasCurrentTestFailed = true;
}
}

void logTestStart(const TestCase& tc) {
char loc[DOCTEST_SNPRINTF_BUFFER_LENGTH];
DOCTEST_SNPRINTF(loc, DOCTEST_COUNTOF(loc), "%s(%d)\n", fileForOutput(tc.m_file),
lineForOutput(tc.m_line));

char ts1[DOCTEST_SNPRINTF_BUFFER_LENGTH];
DOCTEST_SNPRINTF(ts1, DOCTEST_COUNTOF(ts1), "TEST SUITE: ");
char ts2[DOCTEST_SNPRINTF_BUFFER_LENGTH];
DOCTEST_SNPRINTF(ts2, DOCTEST_COUNTOF(ts2), "%s\n", tc.m_test_suite);
char n1[DOCTEST_SNPRINTF_BUFFER_LENGTH];
DOCTEST_SNPRINTF(n1, DOCTEST_COUNTOF(n1), "TEST CASE:  ");
char n2[DOCTEST_SNPRINTF_BUFFER_LENGTH];
DOCTEST_SNPRINTF(n2, DOCTEST_COUNTOF(n2), "%s\n", tc.m_name);
char d1[DOCTEST_SNPRINTF_BUFFER_LENGTH];
DOCTEST_SNPRINTF(d1, DOCTEST_COUNTOF(d1), "DESCRIPTION: ");
char d2[DOCTEST_SNPRINTF_BUFFER_LENGTH];
DOCTEST_SNPRINTF(d2, DOCTEST_COUNTOF(d2), "%s\n", tc.m_description);

char scenario[] = "  Scenario:";
if(std::string(tc.m_name).substr(0, DOCTEST_COUNTOF(scenario) - 1) == scenario)
n1[0] = '\0';

DOCTEST_PRINTF_COLORED(getSeparator(), Color::Yellow);
DOCTEST_PRINTF_COLORED(loc, Color::LightGrey);

String forDebugConsole;
if(tc.m_description) {
DOCTEST_PRINTF_COLORED(d1, Color::Yellow);
DOCTEST_PRINTF_COLORED(d2, Color::None);
forDebugConsole += d1;
forDebugConsole += d2;
}
if(tc.m_test_suite[0] != '\0') {
DOCTEST_PRINTF_COLORED(ts1, Color::Yellow);
DOCTEST_PRINTF_COLORED(ts2, Color::None);
forDebugConsole += ts1;
forDebugConsole += ts2;
}
DOCTEST_PRINTF_COLORED(n1, Color::Yellow);
DOCTEST_PRINTF_COLORED(n2, Color::None);

String                subcaseStuff;
std::vector<Subcase>& subcasesStack = contextState->subcasesStack;
for(unsigned i = 0; i < subcasesStack.size(); ++i) {
if(subcasesStack[i].m_signature.m_name[0] != '\0') {
char subcase[DOCTEST_SNPRINTF_BUFFER_LENGTH];
DOCTEST_SNPRINTF(subcase, DOCTEST_COUNTOF(loc), "  %s\n",
subcasesStack[i].m_signature.m_name);
DOCTEST_PRINTF_COLORED(subcase, Color::None);
subcaseStuff += subcase;
}
}

DOCTEST_PRINTF_COLORED("\n", Color::None);

printToDebugConsole(String(getSeparator()) + loc + forDebugConsole.c_str() + n1 + n2 +
subcaseStuff.c_str() + "\n");
}

void logTestEnd() {}

void logTestException(const String& what, bool crash) {
char msg[DOCTEST_SNPRINTF_BUFFER_LENGTH];

DOCTEST_SNPRINTF(msg, DOCTEST_COUNTOF(msg), "TEST CASE FAILED!\n");

char info1[DOCTEST_SNPRINTF_BUFFER_LENGTH];
char info2[DOCTEST_SNPRINTF_BUFFER_LENGTH];
info1[0] = 0;
info2[0] = 0;
DOCTEST_SNPRINTF(info1, DOCTEST_COUNTOF(info1),
crash ? "crashed:\n" : "threw exception:\n");
DOCTEST_SNPRINTF(info2, DOCTEST_COUNTOF(info2), "  %s\n", what.c_str());

std::string contextStr;

if(!contextState->exceptionalContexts.empty()) {
contextStr += "with context:\n";
for(size_t i = contextState->exceptionalContexts.size(); i > 0; --i) {
contextStr += "  ";
contextStr += contextState->exceptionalContexts[i - 1];
contextStr += "\n";
}
}

DOCTEST_PRINTF_COLORED(msg, Color::Red);
DOCTEST_PRINTF_COLORED(info1, Color::None);
DOCTEST_PRINTF_COLORED(info2, Color::Cyan);
DOCTEST_PRINTF_COLORED(contextStr.c_str(), Color::None);
DOCTEST_PRINTF_COLORED("\n", Color::None);

printToDebugConsole(String(msg) + info1 + info2 + contextStr.c_str() + "\n");
}

String logContext() {
std::ostringstream           stream;
std::vector<IContextScope*>& contexts = contextState->contexts;
if(!contexts.empty())
stream << "with context:\n";
for(size_t i = 0; i < contexts.size(); ++i) {
stream << "  ";
contexts[i]->build(&stream);
stream << "\n";
}
return stream.str().c_str();
}

const char* getFailString(assertType::Enum assert_type) {
if(assert_type & assertType::is_warn) 
return "WARNING";
if(assert_type & assertType::is_check) 
return "ERROR";
if(assert_type & assertType::is_require) 
return "FATAL ERROR";
return "";
}

void logAssert(bool passed, const char* decomposition, bool threw, const String& exception,
const char* expr, assertType::Enum assert_type, const char* file, int line) {
char loc[DOCTEST_SNPRINTF_BUFFER_LENGTH];
DOCTEST_SNPRINTF(loc, DOCTEST_COUNTOF(loc), "%s(%d)", fileForOutput(file),
lineForOutput(line));

char msg[DOCTEST_SNPRINTF_BUFFER_LENGTH];
DOCTEST_SNPRINTF(msg, DOCTEST_COUNTOF(msg), " %s!\n",
passed ? "PASSED" : getFailString(assert_type));

char info1[DOCTEST_SNPRINTF_BUFFER_LENGTH];
DOCTEST_SNPRINTF(info1, DOCTEST_COUNTOF(info1), "  %s( %s )\n",
getAssertString(assert_type), expr);

char info2[DOCTEST_SNPRINTF_BUFFER_LENGTH];
char info3[DOCTEST_SNPRINTF_BUFFER_LENGTH];
info2[0] = 0;
info3[0] = 0;
if(threw) {
DOCTEST_SNPRINTF(info2, DOCTEST_COUNTOF(info2), "threw exception:\n");
DOCTEST_SNPRINTF(info3, DOCTEST_COUNTOF(info3), "  %s\n", exception.c_str());
} else {
DOCTEST_SNPRINTF(info2, DOCTEST_COUNTOF(info2), "with expansion:\n");
DOCTEST_SNPRINTF(info3, DOCTEST_COUNTOF(info3), "  %s( %s )\n",
getAssertString(assert_type), decomposition);
}

bool isWarn = assert_type & assertType::is_warn;
DOCTEST_PRINTF_COLORED(loc, Color::LightGrey);
DOCTEST_PRINTF_COLORED(msg,
passed ? Color::BrightGreen : isWarn ? Color::Yellow : Color::Red);
DOCTEST_PRINTF_COLORED(info1, Color::Cyan);
DOCTEST_PRINTF_COLORED(info2, Color::None);
DOCTEST_PRINTF_COLORED(info3, Color::Cyan);
String context = logContext();
DOCTEST_PRINTF_COLORED(context.c_str(), Color::None);
DOCTEST_PRINTF_COLORED("\n", Color::None);

printToDebugConsole(String(loc) + msg + info1 + info2 + info3 + context.c_str() + "\n");
}

void logAssertThrows(bool threw, const char* expr, assertType::Enum assert_type,
const char* file, int line) {
char loc[DOCTEST_SNPRINTF_BUFFER_LENGTH];
DOCTEST_SNPRINTF(loc, DOCTEST_COUNTOF(loc), "%s(%d)", fileForOutput(file),
lineForOutput(line));

char msg[DOCTEST_SNPRINTF_BUFFER_LENGTH];
DOCTEST_SNPRINTF(msg, DOCTEST_COUNTOF(msg), " %s!\n",
threw ? "PASSED" : getFailString(assert_type));

char info1[DOCTEST_SNPRINTF_BUFFER_LENGTH];
DOCTEST_SNPRINTF(info1, DOCTEST_COUNTOF(info1), "  %s( %s )\n",
getAssertString(assert_type), expr);

char info2[DOCTEST_SNPRINTF_BUFFER_LENGTH];
info2[0] = 0;

if(!threw)
DOCTEST_SNPRINTF(info2, DOCTEST_COUNTOF(info2), "didn't throw at all\n");

bool isWarn = assert_type & assertType::is_warn;
DOCTEST_PRINTF_COLORED(loc, Color::LightGrey);
DOCTEST_PRINTF_COLORED(msg,
threw ? Color::BrightGreen : isWarn ? Color::Yellow : Color::Red);
DOCTEST_PRINTF_COLORED(info1, Color::Cyan);
DOCTEST_PRINTF_COLORED(info2, Color::None);
String context = logContext();
DOCTEST_PRINTF_COLORED(context.c_str(), Color::None);
DOCTEST_PRINTF_COLORED("\n", Color::None);

printToDebugConsole(String(loc) + msg + info1 + info2 + context.c_str() + "\n");
}

void logAssertThrowsAs(bool threw, bool threw_as, const char* as, const String& exception,
const char* expr, assertType::Enum assert_type, const char* file,
int line) {
char loc[DOCTEST_SNPRINTF_BUFFER_LENGTH];
DOCTEST_SNPRINTF(loc, DOCTEST_COUNTOF(loc), "%s(%d)", fileForOutput(file),
lineForOutput(line));

char msg[DOCTEST_SNPRINTF_BUFFER_LENGTH];
DOCTEST_SNPRINTF(msg, DOCTEST_COUNTOF(msg), " %s!\n",
threw_as ? "PASSED" : getFailString(assert_type));

char info1[DOCTEST_SNPRINTF_BUFFER_LENGTH];
DOCTEST_SNPRINTF(info1, DOCTEST_COUNTOF(info1), "  %s( %s, %s )\n",
getAssertString(assert_type), expr, as);

char info2[DOCTEST_SNPRINTF_BUFFER_LENGTH];
char info3[DOCTEST_SNPRINTF_BUFFER_LENGTH];
info2[0] = 0;
info3[0] = 0;

if(!threw) { 
DOCTEST_SNPRINTF(info2, DOCTEST_COUNTOF(info2), "didn't throw at all\n");
} else if(!threw_as) {
DOCTEST_SNPRINTF(info2, DOCTEST_COUNTOF(info2), "threw a different exception:\n");
DOCTEST_SNPRINTF(info3, DOCTEST_COUNTOF(info3), "  %s\n", exception.c_str());
}

bool isWarn = assert_type & assertType::is_warn;
DOCTEST_PRINTF_COLORED(loc, Color::LightGrey);
DOCTEST_PRINTF_COLORED(msg,
threw_as ? Color::BrightGreen : isWarn ? Color::Yellow : Color::Red);
DOCTEST_PRINTF_COLORED(info1, Color::Cyan);
DOCTEST_PRINTF_COLORED(info2, Color::None);
DOCTEST_PRINTF_COLORED(info3, Color::Cyan);
String context = logContext();
DOCTEST_PRINTF_COLORED(context.c_str(), Color::None);
DOCTEST_PRINTF_COLORED("\n", Color::None);

printToDebugConsole(String(loc) + msg + info1 + info2 + info3 + context.c_str() + "\n");
}

void logAssertNothrow(bool threw, const String& exception, const char* expr,
assertType::Enum assert_type, const char* file, int line) {
char loc[DOCTEST_SNPRINTF_BUFFER_LENGTH];
DOCTEST_SNPRINTF(loc, DOCTEST_COUNTOF(loc), "%s(%d)", fileForOutput(file),
lineForOutput(line));

char msg[DOCTEST_SNPRINTF_BUFFER_LENGTH];
DOCTEST_SNPRINTF(msg, DOCTEST_COUNTOF(msg), " %s!\n",
threw ? getFailString(assert_type) : "PASSED");

char info1[DOCTEST_SNPRINTF_BUFFER_LENGTH];
DOCTEST_SNPRINTF(info1, DOCTEST_COUNTOF(info1), "  %s( %s )\n",
getAssertString(assert_type), expr);

char info2[DOCTEST_SNPRINTF_BUFFER_LENGTH];
char info3[DOCTEST_SNPRINTF_BUFFER_LENGTH];
info2[0] = 0;
info3[0] = 0;
if(threw) {
DOCTEST_SNPRINTF(info2, DOCTEST_COUNTOF(info2), "threw exception:\n");
DOCTEST_SNPRINTF(info3, DOCTEST_COUNTOF(info3), "  %s\n", exception.c_str());
}

bool isWarn = assert_type & assertType::is_warn;
DOCTEST_PRINTF_COLORED(loc, Color::LightGrey);
DOCTEST_PRINTF_COLORED(msg,
threw ? isWarn ? Color::Yellow : Color::Red : Color::BrightGreen);
DOCTEST_PRINTF_COLORED(info1, Color::Cyan);
DOCTEST_PRINTF_COLORED(info2, Color::None);
DOCTEST_PRINTF_COLORED(info3, Color::Cyan);
String context = logContext();
DOCTEST_PRINTF_COLORED(context.c_str(), Color::None);
DOCTEST_PRINTF_COLORED("\n", Color::None);

printToDebugConsole(String(loc) + msg + info1 + info2 + info3 + context.c_str() + "\n");
}

ResultBuilder::ResultBuilder(assertType::Enum assert_type, const char* file, int line,
const char* expr, const char* exception_type)
: m_assert_type(assert_type)
, m_file(file)
, m_line(line)
, m_expr(expr)
, m_exception_type(exception_type)
, m_threw(false)
, m_threw_as(false)
, m_failed(false) {
#ifdef _MSC_VER
if(m_expr[0] == ' ') 
++m_expr;
#endif 
}

ResultBuilder::~ResultBuilder() {}

void ResultBuilder::unexpectedExceptionOccurred() {
m_threw = true;

m_exception = translateActiveException();
}

bool ResultBuilder::log() {
if((m_assert_type & assertType::is_warn) == 0) 
contextState->numAssertionsForCurrentTestcase++;

if(m_assert_type & assertType::is_throws) { 
m_failed = !m_threw;
} else if(m_assert_type & 
assertType::is_throws_as) {
m_failed = !m_threw_as;
} else if(m_assert_type & 
assertType::is_nothrow) {
m_failed = m_threw;
} else {
m_failed = m_result;
}

if(m_failed || contextState->success) {
DOCTEST_LOG_START();

if(m_assert_type & assertType::is_throws) { 
logAssertThrows(m_threw, m_expr, m_assert_type, m_file, m_line);
} else if(m_assert_type & 
assertType::is_throws_as) {
logAssertThrowsAs(m_threw, m_threw_as, m_exception_type, m_exception, m_expr,
m_assert_type, m_file, m_line);
} else if(m_assert_type & 
assertType::is_nothrow) {
logAssertNothrow(m_threw, m_exception, m_expr, m_assert_type, m_file, m_line);
} else {
logAssert(m_result.m_passed, m_result.m_decomposition.c_str(), m_threw, m_exception,
m_expr, m_assert_type, m_file, m_line);
}
}

if(m_failed)
addFailedAssert(m_assert_type);

return m_failed && isDebuggerActive() && !contextState->no_breaks; 
}

void ResultBuilder::react() const {
if(m_failed && checkIfShouldThrow(m_assert_type))
throwException();
}

MessageBuilder::MessageBuilder(const char* file, int line,
doctest::detail::assertType::Enum severity)
: m_stream(createStream())
, m_file(file)
, m_line(line)
, m_severity(severity) {}

bool MessageBuilder::log() {
DOCTEST_LOG_START();

bool is_warn = m_severity & doctest::detail::assertType::is_warn;

if(!is_warn) {
contextState->numAssertionsForCurrentTestcase++;
addFailedAssert(m_severity);
}

char loc[DOCTEST_SNPRINTF_BUFFER_LENGTH];
DOCTEST_SNPRINTF(loc, DOCTEST_COUNTOF(loc), "%s(%d)", fileForOutput(m_file),
lineForOutput(m_line));
char msg[DOCTEST_SNPRINTF_BUFFER_LENGTH];
DOCTEST_SNPRINTF(msg, DOCTEST_COUNTOF(msg), " %s!\n",
is_warn ? "MESSAGE" : getFailString(m_severity));

DOCTEST_PRINTF_COLORED(loc, Color::LightGrey);
DOCTEST_PRINTF_COLORED(msg, is_warn ? Color::Yellow : Color::Red);

String info = getStreamResult(m_stream);
if(info.size()) {
DOCTEST_PRINTF_COLORED("  ", Color::None);
DOCTEST_PRINTF_COLORED(info.c_str(), Color::None);
DOCTEST_PRINTF_COLORED("\n", Color::None);
}
String context = logContext();
DOCTEST_PRINTF_COLORED(context.c_str(), Color::None);
DOCTEST_PRINTF_COLORED("\n", Color::None);

printToDebugConsole(String(loc) + msg + "  " + info.c_str() + "\n" + context.c_str() +
"\n");

return isDebuggerActive() && !contextState->no_breaks && !is_warn; 
}

void MessageBuilder::react() {
if(m_severity & assertType::is_require) 
throwException();
}

MessageBuilder::~MessageBuilder() { freeStream(m_stream); }

bool parseFlagImpl(int argc, const char* const* argv, const char* pattern) {
for(int i = argc - 1; i >= 0; --i) {
const char* temp = std::strstr(argv[i], pattern);
if(temp && my_strlen(temp) == my_strlen(pattern)) {
bool noBadCharsFound = true; 
while(temp != argv[i]) {
if(*--temp != '-') {
noBadCharsFound = false;
break;
}
}
if(noBadCharsFound && argv[i][0] == '-')
return true;
}
}
return false;
}

bool parseFlag(int argc, const char* const* argv, const char* pattern) {
#ifndef DOCTEST_CONFIG_NO_UNPREFIXED_OPTIONS
if(!parseFlagImpl(argc, argv, pattern))
return parseFlagImpl(argc, argv, pattern + 3); 
return true;
#else  
return parseFlagImpl(argc, argv, pattern);
#endif 
}

bool parseOptionImpl(int argc, const char* const* argv, const char* pattern, String& res) {
for(int i = argc - 1; i >= 0; --i) {
const char* temp = std::strstr(argv[i], pattern);
if(temp) { 
bool        noBadCharsFound = true;
const char* curr            = argv[i];
while(curr != temp) {
if(*curr++ != '-') {
noBadCharsFound = false;
break;
}
}
if(noBadCharsFound && argv[i][0] == '-') {
temp += my_strlen(pattern);
unsigned len = my_strlen(temp);
if(len) {
res = temp;
return true;
}
}
}
}
return false;
}

bool parseOption(int argc, const char* const* argv, const char* pattern, String& res,
const String& defaultVal = String()) {
res = defaultVal;
#ifndef DOCTEST_CONFIG_NO_UNPREFIXED_OPTIONS
if(!parseOptionImpl(argc, argv, pattern, res))
return parseOptionImpl(argc, argv, pattern + 3, res); 
return true;
#else 
return parseOptionImpl(argc, argv, pattern, res);
#endif 
}

bool parseCommaSepArgs(int argc, const char* const* argv, const char* pattern,
std::vector<String>& res) {
String filtersString;
if(parseOption(argc, argv, pattern, filtersString)) {
char* pch = std::strtok(filtersString.c_str(), ","); 
while(pch != 0) {
if(my_strlen(pch))
res.push_back(pch);
pch = std::strtok(0, ",");
}
return true;
}
return false;
}

enum optionType
{
option_bool,
option_int
};

bool parseIntOption(int argc, const char* const* argv, const char* pattern, optionType type,
int& res) {
String parsedValue;
if(!parseOption(argc, argv, pattern, parsedValue))
return false;

if(type == 0) {
const char positive[][5] = {"1", "true", "on", "yes"};  
const char negative[][6] = {"0", "false", "off", "no"}; 

for(unsigned i = 0; i < 4; i++) {
if(parsedValue.compare(positive[i], true) == 0) {
res = 1; 
return true;
}
if(parsedValue.compare(negative[i], true) == 0) {
res = 0; 
return true;
}
}
} else {
int theInt = std::atoi(parsedValue.c_str()); 
if(theInt != 0) {
res = theInt; 
return true;
}
}
return false;
}

void printVersion() {
if(contextState->no_version == false) {
DOCTEST_PRINTF_COLORED("[doctest] ", Color::Cyan);
std::printf("doctest version is \"%s\"\n", DOCTEST_VERSION_STR);
}
}

void printHelp() {
printVersion();
DOCTEST_PRINTF_COLORED("[doctest]\n", Color::Cyan);
DOCTEST_PRINTF_COLORED("[doctest] ", Color::Cyan);
std::printf("boolean values: \"1/on/yes/true\" or \"0/off/no/false\"\n");
DOCTEST_PRINTF_COLORED("[doctest] ", Color::Cyan);
std::printf("filter  values: \"str1,str2,str3\" (comma separated strings)\n");
DOCTEST_PRINTF_COLORED("[doctest]\n", Color::Cyan);
DOCTEST_PRINTF_COLORED("[doctest] ", Color::Cyan);
std::printf("filters use wildcards for matching strings\n");
DOCTEST_PRINTF_COLORED("[doctest] ", Color::Cyan);
std::printf("something passes a filter if any of the strings in a filter matches\n");
DOCTEST_PRINTF_COLORED("[doctest]\n", Color::Cyan);
DOCTEST_PRINTF_COLORED("[doctest] ", Color::Cyan);
std::printf("ALL FLAGS, OPTIONS AND FILTERS ALSO AVAILABLE WITH A \"dt-\" PREFIX!!!\n");
DOCTEST_PRINTF_COLORED("[doctest]\n", Color::Cyan);
DOCTEST_PRINTF_COLORED("[doctest] ", Color::Cyan);
std::printf("Query flags - the program quits after them. Available:\n\n");
std::printf(" -?,   --help, -h                      prints this message\n");
std::printf(" -v,   --version                       prints the version\n");
std::printf(" -c,   --count                         prints the number of matching tests\n");
std::printf(" -ltc, --list-test-cases               lists all matching tests by name\n");
std::printf(" -lts, --list-test-suites              lists all matching test suites\n\n");
DOCTEST_PRINTF_COLORED("[doctest] ", Color::Cyan);
std::printf("The available <int>/<string> options/filters are:\n\n");
std::printf(" -tc,  --test-case=<filters>           filters     tests by their name\n");
std::printf(" -tce, --test-case-exclude=<filters>   filters OUT tests by their name\n");
std::printf(" -sf,  --source-file=<filters>         filters     tests by their file\n");
std::printf(" -sfe, --source-file-exclude=<filters> filters OUT tests by their file\n");
std::printf(" -ts,  --test-suite=<filters>          filters     tests by their test suite\n");
std::printf(" -tse, --test-suite-exclude=<filters>  filters OUT tests by their test suite\n");
std::printf(" -sc,  --subcase=<filters>             filters     subcases by their name\n");
std::printf(" -sce, --subcase-exclude=<filters>     filters OUT subcases by their name\n");
std::printf(" -ob,  --order-by=<string>             how the tests should be ordered\n");
std::printf("                                       <string> - by [file/suite/name/rand]\n");
std::printf(" -rs,  --rand-seed=<int>               seed for random ordering\n");
std::printf(" -f,   --first=<int>                   the first test passing the filters to\n");
std::printf("                                       execute - for range-based execution\n");
std::printf(" -l,   --last=<int>                    the last test passing the filters to\n");
std::printf("                                       execute - for range-based execution\n");
std::printf(" -aa,  --abort-after=<int>             stop after <int> failed assertions\n");
std::printf(" -scfl,--subcase-filter-levels=<int>   apply filters for the first <int> levels\n");
DOCTEST_PRINTF_COLORED("\n[doctest] ", Color::Cyan);
std::printf("Bool options - can be used like flags and true is assumed. Available:\n\n");
std::printf(" -s,   --success=<bool>                include successful assertions in output\n");
std::printf(" -cs,  --case-sensitive=<bool>         filters being treated as case sensitive\n");
std::printf(" -e,   --exit=<bool>                   exits after the tests finish\n");
std::printf(" -d,   --duration=<bool>               prints the time duration of each test\n");
std::printf(" -nt,  --no-throw=<bool>               skips exceptions-related assert checks\n");
std::printf(" -ne,  --no-exitcode=<bool>            returns (or exits) always with success\n");
std::printf(" -nr,  --no-run=<bool>                 skips all runtime doctest operations\n");
std::printf(" -nv,  --no-version=<bool>             omit the framework version in the output\n");
std::printf(" -nc,  --no-colors=<bool>              disables colors in output\n");
std::printf(" -fc,  --force-colors=<bool>           use colors even when not in a tty\n");
std::printf(" -nb,  --no-breaks=<bool>              disables breakpoints in debuggers\n");
std::printf(" -ns,  --no-skip=<bool>                don't skip test cases marked as skip\n");
std::printf(" -npf, --no-path-filenames=<bool>      only filenames and no paths in output\n");
std::printf(" -nln, --no-line-numbers=<bool>        0 instead of real line numbers in output\n");

DOCTEST_PRINTF_COLORED("\n[doctest] ", Color::Cyan);
std::printf("for more information visit the project documentation\n\n");
}

void printSummary() {
const ContextState* p = contextState;

DOCTEST_PRINTF_COLORED(getSeparator(), Color::Yellow);
if(p->count || p->list_test_cases) {
DOCTEST_PRINTF_COLORED("[doctest] ", Color::Cyan);
std::printf("unskipped test cases passing the current filters: %u\n",
p->numTestsPassingFilters);
} else if(p->list_test_suites) {
DOCTEST_PRINTF_COLORED("[doctest] ", Color::Cyan);
std::printf("unskipped test cases passing the current filters: %u\n",
p->numTestsPassingFilters);
DOCTEST_PRINTF_COLORED("[doctest] ", Color::Cyan);
std::printf("test suites with unskipped test cases passing the current filters: %u\n",
p->numTestSuitesPassingFilters);
} else {
bool anythingFailed = p->numFailed > 0 || p->numFailedAssertions > 0;

char buff[DOCTEST_SNPRINTF_BUFFER_LENGTH];

DOCTEST_PRINTF_COLORED("[doctest] ", Color::Cyan);

DOCTEST_SNPRINTF(buff, DOCTEST_COUNTOF(buff), "test cases: %6u",
p->numTestsPassingFilters);
DOCTEST_PRINTF_COLORED(buff, Color::None);
DOCTEST_SNPRINTF(buff, DOCTEST_COUNTOF(buff), " | ");
DOCTEST_PRINTF_COLORED(buff, Color::None);
DOCTEST_SNPRINTF(buff, DOCTEST_COUNTOF(buff), "%6d passed",
p->numTestsPassingFilters - p->numFailed);
DOCTEST_PRINTF_COLORED(buff,
(p->numTestsPassingFilters == 0 || anythingFailed) ?
Color::None :
Color::Green);
DOCTEST_SNPRINTF(buff, DOCTEST_COUNTOF(buff), " | ");
DOCTEST_PRINTF_COLORED(buff, Color::None);
DOCTEST_SNPRINTF(buff, DOCTEST_COUNTOF(buff), "%6u failed", p->numFailed);
DOCTEST_PRINTF_COLORED(buff, p->numFailed > 0 ? Color::Red : Color::None);

DOCTEST_SNPRINTF(buff, DOCTEST_COUNTOF(buff), " | ");
DOCTEST_PRINTF_COLORED(buff, Color::None);
if(p->no_skipped_summary == false) {
int numSkipped = static_cast<unsigned>(getRegisteredTests().size()) -
p->numTestsPassingFilters;
DOCTEST_SNPRINTF(buff, DOCTEST_COUNTOF(buff), "%6d skipped", numSkipped);
DOCTEST_PRINTF_COLORED(buff, numSkipped == 0 ? Color::None : Color::Yellow);
}
DOCTEST_PRINTF_COLORED("\n", Color::None);

DOCTEST_PRINTF_COLORED("[doctest] ", Color::Cyan);

DOCTEST_SNPRINTF(buff, DOCTEST_COUNTOF(buff), "assertions: %6d", p->numAssertions);
DOCTEST_PRINTF_COLORED(buff, Color::None);
DOCTEST_SNPRINTF(buff, DOCTEST_COUNTOF(buff), " | ");
DOCTEST_PRINTF_COLORED(buff, Color::None);
DOCTEST_SNPRINTF(buff, DOCTEST_COUNTOF(buff), "%6d passed",
p->numAssertions - p->numFailedAssertions);
DOCTEST_PRINTF_COLORED(
buff, (p->numAssertions == 0 || anythingFailed) ? Color::None : Color::Green);
DOCTEST_SNPRINTF(buff, DOCTEST_COUNTOF(buff), " | ");
DOCTEST_PRINTF_COLORED(buff, Color::None);
DOCTEST_SNPRINTF(buff, DOCTEST_COUNTOF(buff), "%6d failed", p->numFailedAssertions);
DOCTEST_PRINTF_COLORED(buff, p->numFailedAssertions > 0 ? Color::Red : Color::None);

DOCTEST_SNPRINTF(buff, DOCTEST_COUNTOF(buff), " |\n");
DOCTEST_PRINTF_COLORED(buff, Color::None);

DOCTEST_PRINTF_COLORED("[doctest] ", Color::Cyan);
DOCTEST_PRINTF_COLORED("Status: ", Color::None);
const char* result = (p->numFailed > 0) ? "FAILURE!\n" : "SUCCESS!\n";
DOCTEST_PRINTF_COLORED(result, p->numFailed > 0 ? Color::Red : Color::Green);
}

DOCTEST_PRINTF_COLORED("", Color::None);
}
} 

bool isRunningInTest() { return detail::contextState != 0; }

Context::Context(int argc, const char* const* argv)
: p(new detail::ContextState) {
parseArgs(argc, argv, true);
}

Context::~Context() { delete p; }

void Context::applyCommandLine(int argc, const char* const* argv) { parseArgs(argc, argv); }

void Context::parseArgs(int argc, const char* const* argv, bool withDefaults) {
using namespace detail;

parseCommaSepArgs(argc, argv, "dt-source-file=",        p->filters[0]);
parseCommaSepArgs(argc, argv, "dt-sf=",                 p->filters[0]);
parseCommaSepArgs(argc, argv, "dt-source-file-exclude=",p->filters[1]);
parseCommaSepArgs(argc, argv, "dt-sfe=",                p->filters[1]);
parseCommaSepArgs(argc, argv, "dt-test-suite=",         p->filters[2]);
parseCommaSepArgs(argc, argv, "dt-ts=",                 p->filters[2]);
parseCommaSepArgs(argc, argv, "dt-test-suite-exclude=", p->filters[3]);
parseCommaSepArgs(argc, argv, "dt-tse=",                p->filters[3]);
parseCommaSepArgs(argc, argv, "dt-test-case=",          p->filters[4]);
parseCommaSepArgs(argc, argv, "dt-tc=",                 p->filters[4]);
parseCommaSepArgs(argc, argv, "dt-test-case-exclude=",  p->filters[5]);
parseCommaSepArgs(argc, argv, "dt-tce=",                p->filters[5]);
parseCommaSepArgs(argc, argv, "dt-subcase=",            p->filters[6]);
parseCommaSepArgs(argc, argv, "dt-sc=",                 p->filters[6]);
parseCommaSepArgs(argc, argv, "dt-subcase-exclude=",    p->filters[7]);
parseCommaSepArgs(argc, argv, "dt-sce=",                p->filters[7]);

int    intRes = 0;
String strRes;

#define DOCTEST_PARSE_AS_BOOL_OR_FLAG(name, sname, var, default)                                   \
if(parseIntOption(argc, argv, DOCTEST_STR_CONCAT_TOSTR(name, =), option_bool, intRes) ||       \
parseIntOption(argc, argv, DOCTEST_STR_CONCAT_TOSTR(sname, =), option_bool, intRes))        \
p->var = !!intRes;                                                                         \
else if(parseFlag(argc, argv, #name) || parseFlag(argc, argv, #sname))                         \
p->var = true;                                                                             \
else if(withDefaults)                                                                          \
p->var = default

#define DOCTEST_PARSE_INT_OPTION(name, sname, var, default)                                        \
if(parseIntOption(argc, argv, DOCTEST_STR_CONCAT_TOSTR(name, =), option_int, intRes) ||        \
parseIntOption(argc, argv, DOCTEST_STR_CONCAT_TOSTR(sname, =), option_int, intRes))         \
p->var = intRes;                                                                           \
else if(withDefaults)                                                                          \
p->var = default

#define DOCTEST_PARSE_STR_OPTION(name, sname, var, default)                                        \
if(parseOption(argc, argv, DOCTEST_STR_CONCAT_TOSTR(name, =), strRes, default) ||              \
parseOption(argc, argv, DOCTEST_STR_CONCAT_TOSTR(sname, =), strRes, default) ||             \
withDefaults)                                                                               \
p->var = strRes

DOCTEST_PARSE_STR_OPTION(dt-order-by, dt-ob, order_by, "file");
DOCTEST_PARSE_INT_OPTION(dt-rand-seed, dt-rs, rand_seed, 0);

DOCTEST_PARSE_INT_OPTION(dt-first, dt-f, first, 1);
DOCTEST_PARSE_INT_OPTION(dt-last, dt-l, last, 0);

DOCTEST_PARSE_INT_OPTION(dt-abort-after, dt-aa, abort_after, 0);
DOCTEST_PARSE_INT_OPTION(dt-subcase-filter-levels, dt-scfl, subcase_filter_levels, 2000000000);

DOCTEST_PARSE_AS_BOOL_OR_FLAG(dt-success, dt-s, success, false);
DOCTEST_PARSE_AS_BOOL_OR_FLAG(dt-case-sensitive, dt-cs, case_sensitive, false);
DOCTEST_PARSE_AS_BOOL_OR_FLAG(dt-exit, dt-e, exit, false);
DOCTEST_PARSE_AS_BOOL_OR_FLAG(dt-duration, dt-d, duration, false);
DOCTEST_PARSE_AS_BOOL_OR_FLAG(dt-no-throw, dt-nt, no_throw, false);
DOCTEST_PARSE_AS_BOOL_OR_FLAG(dt-no-exitcode, dt-ne, no_exitcode, false);
DOCTEST_PARSE_AS_BOOL_OR_FLAG(dt-no-run, dt-nr, no_run, false);
DOCTEST_PARSE_AS_BOOL_OR_FLAG(dt-no-version, dt-nv, no_version, false);
DOCTEST_PARSE_AS_BOOL_OR_FLAG(dt-no-colors, dt-nc, no_colors, false);
DOCTEST_PARSE_AS_BOOL_OR_FLAG(dt-force-colors, dt-fc, force_colors, false);
DOCTEST_PARSE_AS_BOOL_OR_FLAG(dt-no-breaks, dt-nb, no_breaks, false);
DOCTEST_PARSE_AS_BOOL_OR_FLAG(dt-no-skip, dt-ns, no_skip, false);
DOCTEST_PARSE_AS_BOOL_OR_FLAG(dt-no-path-filenames, dt-npf, no_path_in_filenames, false);
DOCTEST_PARSE_AS_BOOL_OR_FLAG(dt-no-line-numbers, dt-nln, no_line_numbers, false);
DOCTEST_PARSE_AS_BOOL_OR_FLAG(dt-no-skipped-summary, dt-nss, no_skipped_summary, false);

#undef DOCTEST_PARSE_STR_OPTION
#undef DOCTEST_PARSE_INT_OPTION
#undef DOCTEST_PARSE_AS_BOOL_OR_FLAG

if(withDefaults) {
p->help             = false;
p->version          = false;
p->count            = false;
p->list_test_cases  = false;
p->list_test_suites = false;
}
if(parseFlag(argc, argv, "dt-help") || parseFlag(argc, argv, "dt-h") ||
parseFlag(argc, argv, "dt-?")) {
p->help = true;
p->exit = true;
}
if(parseFlag(argc, argv, "dt-version") || parseFlag(argc, argv, "dt-v")) {
p->version = true;
p->exit    = true;
}
if(parseFlag(argc, argv, "dt-count") || parseFlag(argc, argv, "dt-c")) {
p->count = true;
p->exit  = true;
}
if(parseFlag(argc, argv, "dt-list-test-cases") || parseFlag(argc, argv, "dt-ltc")) {
p->list_test_cases = true;
p->exit            = true;
}
if(parseFlag(argc, argv, "dt-list-test-suites") || parseFlag(argc, argv, "dt-lts")) {
p->list_test_suites = true;
p->exit             = true;
}
}

void Context::addFilter(const char* filter, const char* value) { setOption(filter, value); }

void Context::clearFilters() {
for(unsigned i = 0; i < p->filters.size(); ++i)
p->filters[i].clear();
}

void Context::setOption(const char* option, int value) {
setOption(option, toString(value).c_str());
}

void Context::setOption(const char* option, const char* value) {
String      argv   = String("-") + option + "=" + value;
const char* lvalue = argv.c_str();
parseArgs(1, &lvalue);
}

bool Context::shouldExit() { return p->exit; }

int Context::run() {
using namespace detail;

Color::init();

contextState = p;
p->resetRunData();

if(p->no_run || p->version || p->help) {
if(p->version)
printVersion();
if(p->help)
printHelp();

contextState = 0;

return EXIT_SUCCESS;
}

printVersion();
DOCTEST_PRINTF_COLORED("[doctest] ", Color::Cyan);
std::printf("run with \"--help\" for options\n");

unsigned i = 0; 

std::set<TestCase>& registeredTests = getRegisteredTests();

std::vector<const TestCase*> testArray;
for(std::set<TestCase>::iterator it = registeredTests.begin(); it != registeredTests.end();
++it)
testArray.push_back(&(*it));

if(!testArray.empty()) {
if(p->order_by.compare("file", true) == 0) {
std::qsort(&testArray[0], testArray.size(), sizeof(TestCase*), fileOrderComparator);
} else if(p->order_by.compare("suite", true) == 0) {
std::qsort(&testArray[0], testArray.size(), sizeof(TestCase*), suiteOrderComparator);
} else if(p->order_by.compare("name", true) == 0) {
std::qsort(&testArray[0], testArray.size(), sizeof(TestCase*), nameOrderComparator);
} else if(p->order_by.compare("rand", true) == 0) {
std::srand(p->rand_seed);

const TestCase** first = &testArray[0];
for(i = testArray.size() - 1; i > 0; --i) {
int idxToSwap = std::rand() % (i + 1); 

const TestCase* temp = first[i];

first[i]         = first[idxToSwap];
first[idxToSwap] = temp;
}
}
}

if(p->list_test_cases) {
DOCTEST_PRINTF_COLORED("[doctest] ", Color::Cyan);
std::printf("listing all test case names\n");
DOCTEST_PRINTF_COLORED(getSeparator(), Color::Yellow);
}

std::set<String> testSuitesPassingFilters;
if(p->list_test_suites) {
DOCTEST_PRINTF_COLORED("[doctest] ", Color::Cyan);
std::printf("listing all test suites\n");
DOCTEST_PRINTF_COLORED(getSeparator(), Color::Yellow);
}

for(i = 0; i < testArray.size(); i++) {
const TestCase& data = *testArray[i];

if(data.m_skip && !p->no_skip)
continue;

if(!matchesAny(data.m_file, p->filters[0], 1, p->case_sensitive))
continue;
if(matchesAny(data.m_file, p->filters[1], 0, p->case_sensitive))
continue;
if(!matchesAny(data.m_test_suite, p->filters[2], 1, p->case_sensitive))
continue;
if(matchesAny(data.m_test_suite, p->filters[3], 0, p->case_sensitive))
continue;
if(!matchesAny(data.m_name, p->filters[4], 1, p->case_sensitive))
continue;
if(matchesAny(data.m_name, p->filters[5], 0, p->case_sensitive))
continue;

p->numTestsPassingFilters++;

if(p->count)
continue;

if(p->list_test_cases) {
std::printf("%s\n", data.m_name);
continue;
}

if(p->list_test_suites) {
if((testSuitesPassingFilters.count(data.m_test_suite) == 0) &&
data.m_test_suite[0] != '\0') {
std::printf("%s\n", data.m_test_suite);
testSuitesPassingFilters.insert(data.m_test_suite);
p->numTestSuitesPassingFilters++;
}
continue;
}

if((p->last < p->numTestsPassingFilters && p->first <= p->last) ||
(p->first > p->numTestsPassingFilters))
continue;

{
p->currentTest = &data;

bool failed                              = false;
p->hasLoggedCurrentTestStart             = false;
p->numFailedAssertionsForCurrentTestcase = 0;
p->subcasesPassed.clear();
double duration = 0;
Timer  timer;
timer.start();
do {
if(p->hasLoggedCurrentTestStart)
logTestEnd();
p->hasLoggedCurrentTestStart = false;

if(p->success)
DOCTEST_LOG_START();

p->numAssertionsForCurrentTestcase = 0;
p->hasCurrentTestFailed            = false;

p->subcasesHasSkipped   = false;
p->subcasesCurrentLevel = 0;
p->subcasesEnteredLevels.clear();

p->exceptionalContexts.clear();

#ifndef DOCTEST_CONFIG_NO_EXCEPTIONS
try {
#endif 
FatalConditionHandler fatalConditionHandler; 
data.m_test();
fatalConditionHandler.reset();
if(contextState->hasCurrentTestFailed)
failed = true;
#ifndef DOCTEST_CONFIG_NO_EXCEPTIONS
} catch(const TestFailureException&) { failed = true; } catch(...) {
DOCTEST_LOG_START();
logTestException(translateActiveException());
failed = true;
}
#endif 

p->numAssertions += p->numAssertionsForCurrentTestcase;

if(p->abort_after > 0 && p->numFailedAssertions >= p->abort_after) {
p->subcasesHasSkipped = false;
DOCTEST_PRINTF_COLORED("Aborting - too many failed asserts!\n", Color::Red);
}

} while(p->subcasesHasSkipped == true);

duration = timer.getElapsedSeconds();

if(Approx(p->currentTest->m_timeout).epsilon(DBL_EPSILON) != 0 &&
Approx(duration).epsilon(DBL_EPSILON) > p->currentTest->m_timeout) {
failed = true;
DOCTEST_LOG_START();
char msg[DOCTEST_SNPRINTF_BUFFER_LENGTH];
DOCTEST_SNPRINTF(msg, DOCTEST_COUNTOF(msg),
"Test case exceeded time limit of %.6f!\n",
p->currentTest->m_timeout);
DOCTEST_PRINTF_COLORED(msg, Color::Red);
}

if(p->duration) {
char msg[DOCTEST_SNPRINTF_BUFFER_LENGTH];
DOCTEST_SNPRINTF(msg, DOCTEST_COUNTOF(msg), "%.6f s: %s\n", duration,
p->currentTest->m_name);
DOCTEST_PRINTF_COLORED(msg, Color::None);
}

if(data.m_should_fail) {
DOCTEST_LOG_START();
if(failed) {
failed = false;
DOCTEST_PRINTF_COLORED("Failed as expected so marking it as not failed\n",
Color::Yellow);
} else {
failed = true;
DOCTEST_PRINTF_COLORED("Should have failed but didn't! Marking it as failed!\n",
Color::Red);
}
} else if(failed && data.m_may_fail) {
DOCTEST_LOG_START();
failed = false;
DOCTEST_PRINTF_COLORED("Allowed to fail so marking it as not failed\n",
Color::Yellow);
} else if(data.m_expected_failures > 0) {
DOCTEST_LOG_START();
char msg[DOCTEST_SNPRINTF_BUFFER_LENGTH];
if(p->numFailedAssertionsForCurrentTestcase == data.m_expected_failures) {
failed = false;
DOCTEST_SNPRINTF(
msg, DOCTEST_COUNTOF(msg),
"Failed exactly %d times as expected so marking it as not failed!\n",
data.m_expected_failures);
DOCTEST_PRINTF_COLORED(msg, Color::Yellow);
} else {
failed = true;
DOCTEST_SNPRINTF(msg, DOCTEST_COUNTOF(msg),
"Didn't fail exactly %d times so marking it as failed!\n",
data.m_expected_failures);
DOCTEST_PRINTF_COLORED(msg, Color::Red);
}
}

if(p->hasLoggedCurrentTestStart)
logTestEnd();

if(failed) 
p->numFailed++;

if(p->abort_after > 0 && p->numFailedAssertions >= p->abort_after)
break;
}
}

printSummary();

contextState = 0;

if(p->numFailed && !p->no_exitcode)
return EXIT_FAILURE;
return EXIT_SUCCESS;
}
} 

#endif 

#ifdef DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
int main(int argc, char** argv) { return doctest::Context(argc, argv).run(); }
#endif 

#endif 
#endif 

#if defined(__clang__)
#pragma clang diagnostic pop
#endif 

#if defined(__GNUC__) && !defined(__clang__)
#if __GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ > 6)
#pragma GCC diagnostic pop
#endif 
#endif 

#ifdef _MSC_VER
#pragma warning(pop)
#endif 