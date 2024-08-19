

#ifndef CATCH_AMALGAMATED_HPP_INCLUDED
#define CATCH_AMALGAMATED_HPP_INCLUDED




#ifndef CATCH_ALL_HPP_INCLUDED
#define CATCH_ALL_HPP_INCLUDED





#ifndef CATCH_BENCHMARK_ALL_HPP_INCLUDED
#define CATCH_BENCHMARK_ALL_HPP_INCLUDED




#ifndef CATCH_BENCHMARK_HPP_INCLUDED
#define CATCH_BENCHMARK_HPP_INCLUDED



#ifndef CATCH_INTERFACES_CONFIG_HPP_INCLUDED
#define CATCH_INTERFACES_CONFIG_HPP_INCLUDED



#ifndef CATCH_NONCOPYABLE_HPP_INCLUDED
#define CATCH_NONCOPYABLE_HPP_INCLUDED

namespace Catch {
namespace Detail {

class NonCopyable {
NonCopyable( NonCopyable const& ) = delete;
NonCopyable( NonCopyable&& ) = delete;
NonCopyable& operator=( NonCopyable const& ) = delete;
NonCopyable& operator=( NonCopyable&& ) = delete;

protected:
NonCopyable() noexcept = default;
};

} 
} 

#endif 


#ifndef CATCH_STRINGREF_HPP_INCLUDED
#define CATCH_STRINGREF_HPP_INCLUDED

#include <cstddef>
#include <string>
#include <iosfwd>
#include <cassert>

namespace Catch {

class StringRef {
public:
using size_type = std::size_t;
using const_iterator = const char*;

private:
static constexpr char const* const s_empty = "";

char const* m_start = s_empty;
size_type m_size = 0;

public: 
constexpr StringRef() noexcept = default;

StringRef( char const* rawChars ) noexcept;

constexpr StringRef( char const* rawChars, size_type size ) noexcept
:   m_start( rawChars ),
m_size( size )
{}

StringRef( std::string const& stdString ) noexcept
:   m_start( stdString.c_str() ),
m_size( stdString.size() )
{}

explicit operator std::string() const {
return std::string(m_start, m_size);
}

public: 
auto operator == ( StringRef other ) const noexcept -> bool;
auto operator != (StringRef other) const noexcept -> bool {
return !(*this == other);
}

constexpr auto operator[] ( size_type index ) const noexcept -> char {
assert(index < m_size);
return m_start[index];
}

bool operator<(StringRef rhs) const noexcept;

public: 
constexpr auto empty() const noexcept -> bool {
return m_size == 0;
}
constexpr auto size() const noexcept -> size_type {
return m_size;
}

constexpr StringRef substr(size_type start, size_type length) const noexcept {
if (start < m_size) {
const auto shortened_size = m_size - start;
return StringRef(m_start + start, (shortened_size < length) ? shortened_size : length);
} else {
return StringRef();
}
}

constexpr char const* data() const noexcept {
return m_start;
}

constexpr const_iterator begin() const { return m_start; }
constexpr const_iterator end() const { return m_start + m_size; }


friend std::string& operator += (std::string& lhs, StringRef sr);
friend std::ostream& operator << (std::ostream& os, StringRef sr);
friend std::string operator+(StringRef lhs, StringRef rhs);


int compare( StringRef rhs ) const;
};


constexpr auto operator ""_sr( char const* rawChars, std::size_t size ) noexcept -> StringRef {
return StringRef( rawChars, size );
}
} 

constexpr auto operator ""_catch_sr( char const* rawChars, std::size_t size ) noexcept -> Catch::StringRef {
return Catch::StringRef( rawChars, size );
}

#endif 

#include <chrono>
#include <iosfwd>
#include <string>
#include <vector>

namespace Catch {

enum class Verbosity {
Quiet = 0,
Normal,
High
};

struct WarnAbout { enum What {
Nothing = 0x00,
NoAssertions = 0x01,
UnmatchedTestSpec = 0x02,
}; };

enum class ShowDurations {
DefaultForReporter,
Always,
Never
};
enum class TestRunOrder {
Declared,
LexicographicallySorted,
Randomized
};
enum class ColourMode : std::uint8_t {
PlatformDefault,
ANSI,
Win32,
None
};
struct WaitForKeypress { enum When {
Never,
BeforeStart = 1,
BeforeExit = 2,
BeforeStartAndExit = BeforeStart | BeforeExit
}; };

class TestSpec;
class IStream;

class IConfig : public Detail::NonCopyable {
public:
virtual ~IConfig();

virtual bool allowThrows() const = 0;
virtual StringRef name() const = 0;
virtual bool includeSuccessfulResults() const = 0;
virtual bool shouldDebugBreak() const = 0;
virtual bool warnAboutMissingAssertions() const = 0;
virtual bool warnAboutUnmatchedTestSpecs() const = 0;
virtual bool zeroTestsCountAsSuccess() const = 0;
virtual int abortAfter() const = 0;
virtual bool showInvisibles() const = 0;
virtual ShowDurations showDurations() const = 0;
virtual double minDuration() const = 0;
virtual TestSpec const& testSpec() const = 0;
virtual bool hasTestFilters() const = 0;
virtual std::vector<std::string> const& getTestsOrTags() const = 0;
virtual TestRunOrder runOrder() const = 0;
virtual uint32_t rngSeed() const = 0;
virtual unsigned int shardCount() const = 0;
virtual unsigned int shardIndex() const = 0;
virtual ColourMode defaultColourMode() const = 0;
virtual std::vector<std::string> const& getSectionsToRun() const = 0;
virtual Verbosity verbosity() const = 0;

virtual bool skipBenchmarks() const = 0;
virtual bool benchmarkNoAnalysis() const = 0;
virtual unsigned int benchmarkSamples() const = 0;
virtual double benchmarkConfidenceInterval() const = 0;
virtual unsigned int benchmarkResamples() const = 0;
virtual std::chrono::milliseconds benchmarkWarmupTime() const = 0;
};
}

#endif 


#ifndef CATCH_COMPILER_CAPABILITIES_HPP_INCLUDED
#define CATCH_COMPILER_CAPABILITIES_HPP_INCLUDED





#ifndef CATCH_PLATFORM_HPP_INCLUDED
#define CATCH_PLATFORM_HPP_INCLUDED

#ifdef __APPLE__
#  include <TargetConditionals.h>
#  if (defined(TARGET_OS_OSX) && TARGET_OS_OSX == 1) || \
(defined(TARGET_OS_MAC) && TARGET_OS_MAC == 1)
#    define CATCH_PLATFORM_MAC
#  elif (defined(TARGET_OS_IPHONE) && TARGET_OS_IPHONE == 1)
#    define CATCH_PLATFORM_IPHONE
#  endif

#elif defined(linux) || defined(__linux) || defined(__linux__)
#  define CATCH_PLATFORM_LINUX

#elif defined(WIN32) || defined(__WIN32__) || defined(_WIN32) || defined(_MSC_VER) || defined(__MINGW32__)
#  define CATCH_PLATFORM_WINDOWS

#  if defined( WINAPI_FAMILY ) && ( WINAPI_FAMILY == WINAPI_FAMILY_APP )
#      define CATCH_PLATFORM_WINDOWS_UWP
#  endif

#elif defined(__ORBIS__) || defined(__PROSPERO__)
#  define CATCH_PLATFORM_PLAYSTATION

#endif

#endif 

#ifdef __cplusplus

#  if (__cplusplus >= 201402L) || (defined(_MSVC_LANG) && _MSVC_LANG >= 201402L)
#    define CATCH_CPP14_OR_GREATER
#  endif

#  if (__cplusplus >= 201703L) || (defined(_MSVC_LANG) && _MSVC_LANG >= 201703L)
#    define CATCH_CPP17_OR_GREATER
#  endif

#endif

#if defined(__GNUC__) && !defined(__clang__) && !defined(__ICC) && !defined(__CUDACC__) && !defined(__LCC__)
#    define CATCH_INTERNAL_START_WARNINGS_SUPPRESSION _Pragma( "GCC diagnostic push" )
#    define CATCH_INTERNAL_STOP_WARNINGS_SUPPRESSION  _Pragma( "GCC diagnostic pop" )

#    define CATCH_INTERNAL_SUPPRESS_PARENTHESES_WARNINGS \
_Pragma( "GCC diagnostic ignored \"-Wparentheses\"" )

#    define CATCH_INTERNAL_SUPPRESS_UNUSED_VARIABLE_WARNINGS \
_Pragma( "GCC diagnostic ignored \"-Wunused-variable\"" )

#    define CATCH_INTERNAL_SUPPRESS_USELESS_CAST_WARNINGS \
_Pragma( "GCC diagnostic ignored \"-Wuseless-cast\"" )

#    define CATCH_INTERNAL_IGNORE_BUT_WARN(...) (void)__builtin_constant_p(__VA_ARGS__)

#endif

#if defined(__CUDACC__) && !defined(__clang__)
#    define CATCH_INTERNAL_START_WARNINGS_SUPPRESSION _Pragma( "nv_diagnostic push" )
#    define CATCH_INTERNAL_STOP_WARNINGS_SUPPRESSION  _Pragma( "nv_diagnostic pop" )
#    define CATCH_INTERNAL_SUPPRESS_UNUSED_VARIABLE_WARNINGS _Pragma( "nv_diag_suppress 177" )
#endif

#if defined(__clang__) && !defined(_MSC_VER)

#    define CATCH_INTERNAL_START_WARNINGS_SUPPRESSION _Pragma( "clang diagnostic push" )
#    define CATCH_INTERNAL_STOP_WARNINGS_SUPPRESSION  _Pragma( "clang diagnostic pop" )

#endif 

#if defined(__clang__)

#  if !defined(__ibmxl__) && !defined(__CUDACC__) && !defined( __NVCOMPILER )
#    define CATCH_INTERNAL_IGNORE_BUT_WARN(...) (void)__builtin_constant_p(__VA_ARGS__) 
#  endif


#    define CATCH_INTERNAL_SUPPRESS_GLOBALS_WARNINGS \
_Pragma( "clang diagnostic ignored \"-Wexit-time-destructors\"" ) \
_Pragma( "clang diagnostic ignored \"-Wglobal-constructors\"")

#    define CATCH_INTERNAL_SUPPRESS_PARENTHESES_WARNINGS \
_Pragma( "clang diagnostic ignored \"-Wparentheses\"" )

#    define CATCH_INTERNAL_SUPPRESS_UNUSED_VARIABLE_WARNINGS \
_Pragma( "clang diagnostic ignored \"-Wunused-variable\"" )

#    define CATCH_INTERNAL_SUPPRESS_ZERO_VARIADIC_WARNINGS \
_Pragma( "clang diagnostic ignored \"-Wgnu-zero-variadic-macro-arguments\"" )

#    define CATCH_INTERNAL_SUPPRESS_UNUSED_TEMPLATE_WARNINGS \
_Pragma( "clang diagnostic ignored \"-Wunused-template\"" )

#    define CATCH_INTERNAL_SUPPRESS_COMMA_WARNINGS \
_Pragma( "clang diagnostic ignored \"-Wcomma\"" )

#endif 


#if defined( CATCH_PLATFORM_WINDOWS ) ||                                       \
defined( CATCH_PLATFORM_PLAYSTATION ) ||                                   \
defined( __CYGWIN__ ) ||                                                   \
defined( __QNX__ ) ||                                                      \
defined( __EMSCRIPTEN__ ) ||                                               \
defined( __DJGPP__ ) ||                                                    \
defined( __OS400__ )
#    define CATCH_INTERNAL_CONFIG_NO_POSIX_SIGNALS
#else
#    define CATCH_INTERNAL_CONFIG_POSIX_SIGNALS
#endif

#if defined(CATCH_PLATFORM_WINDOWS_UWP) || defined(CATCH_PLATFORM_PLAYSTATION)
#    define CATCH_INTERNAL_CONFIG_NO_GETENV
#else
#    define CATCH_INTERNAL_CONFIG_GETENV
#endif

#if defined(__ANDROID__)
#    define CATCH_INTERNAL_CONFIG_NO_CPP11_TO_STRING
#endif

#if defined(__MINGW32__)
#    define CATCH_INTERNAL_CONFIG_NO_WINDOWS_SEH
#endif

#if defined(__ORBIS__)
#    define CATCH_INTERNAL_CONFIG_NO_NEW_CAPTURE
#endif

#ifdef __CYGWIN__

#   define _BSD_SOURCE
# if !((__cplusplus >= 201103L) && defined(_GLIBCXX_USE_C99) \
&& !defined(_GLIBCXX_HAVE_BROKEN_VSWPRINTF))

#    define CATCH_INTERNAL_CONFIG_NO_CPP11_TO_STRING

# endif
#endif 

#if defined(_MSC_VER)

#  define CATCH_INTERNAL_START_WARNINGS_SUPPRESSION __pragma( warning(push) )
#  define CATCH_INTERNAL_STOP_WARNINGS_SUPPRESSION  __pragma( warning(pop) )

#  if defined(CATCH_PLATFORM_WINDOWS_UWP)
#    define CATCH_INTERNAL_CONFIG_NO_COLOUR_WIN32
#  else
#    define CATCH_INTERNAL_CONFIG_WINDOWS_SEH
#  endif

#  if !defined(__clang__) 
#    if !defined(_MSVC_TRADITIONAL) || (defined(_MSVC_TRADITIONAL) && _MSVC_TRADITIONAL)
#      define CATCH_INTERNAL_CONFIG_TRADITIONAL_MSVC_PREPROCESSOR
#    endif 
#  endif 

#endif 

#if defined(_REENTRANT) || defined(_MSC_VER)
# define CATCH_INTERNAL_CONFIG_USE_ASYNC
#endif 

#if defined(__EXCEPTIONS) || defined(__cpp_exceptions) || defined(_CPPUNWIND)
#  define CATCH_INTERNAL_CONFIG_EXCEPTIONS_ENABLED
#endif


#if defined(__BORLANDC__)
#define CATCH_INTERNAL_CONFIG_POLYFILL_ISNAN
#endif


#if defined(UNDER_RTSS) || defined(RTX64_BUILD)
#define CATCH_INTERNAL_CONFIG_NO_WINDOWS_SEH
#define CATCH_INTERNAL_CONFIG_NO_ASYNC
#define CATCH_INTERNAL_CONFIG_NO_COLOUR_WIN32
#endif

#if !defined(_GLIBCXX_USE_C99_MATH_TR1)
#define CATCH_INTERNAL_CONFIG_GLOBAL_NEXTAFTER
#endif

#if defined(__has_include)
#if __has_include(<string_view>) && defined(CATCH_CPP17_OR_GREATER)
#    define CATCH_INTERNAL_CONFIG_CPP17_STRING_VIEW
#endif

#  if __has_include(<optional>) && defined(CATCH_CPP17_OR_GREATER)
#    define CATCH_INTERNAL_CONFIG_CPP17_OPTIONAL
#  endif 

#  if __has_include(<cstddef>) && defined(CATCH_CPP17_OR_GREATER)
#    include <cstddef>
#    if defined(__cpp_lib_byte) && (__cpp_lib_byte > 0)
#      define CATCH_INTERNAL_CONFIG_CPP17_BYTE
#    endif
#  endif 

#  if __has_include(<variant>) && defined(CATCH_CPP17_OR_GREATER)
#    if defined(__clang__) && (__clang_major__ < 8)
#      include <ciso646>
#      if defined(__GLIBCXX__) && defined(_GLIBCXX_RELEASE) && (_GLIBCXX_RELEASE < 9)
#        define CATCH_CONFIG_NO_CPP17_VARIANT
#      else
#        define CATCH_INTERNAL_CONFIG_CPP17_VARIANT
#      endif 
#    else
#      define CATCH_INTERNAL_CONFIG_CPP17_VARIANT
#    endif 
#  endif 
#endif 


#if defined(CATCH_INTERNAL_CONFIG_WINDOWS_SEH) && !defined(CATCH_CONFIG_NO_WINDOWS_SEH) && !defined(CATCH_CONFIG_WINDOWS_SEH) && !defined(CATCH_INTERNAL_CONFIG_NO_WINDOWS_SEH)
#   define CATCH_CONFIG_WINDOWS_SEH
#endif
#if defined(CATCH_INTERNAL_CONFIG_POSIX_SIGNALS) && !defined(CATCH_INTERNAL_CONFIG_NO_POSIX_SIGNALS) && !defined(CATCH_CONFIG_NO_POSIX_SIGNALS) && !defined(CATCH_CONFIG_POSIX_SIGNALS)
#   define CATCH_CONFIG_POSIX_SIGNALS
#endif

#if defined(CATCH_INTERNAL_CONFIG_GETENV) && !defined(CATCH_INTERNAL_CONFIG_NO_GETENV) && !defined(CATCH_CONFIG_NO_GETENV) && !defined(CATCH_CONFIG_GETENV)
#   define CATCH_CONFIG_GETENV
#endif

#if !defined(CATCH_INTERNAL_CONFIG_NO_CPP11_TO_STRING) && !defined(CATCH_CONFIG_NO_CPP11_TO_STRING) && !defined(CATCH_CONFIG_CPP11_TO_STRING)
#    define CATCH_CONFIG_CPP11_TO_STRING
#endif

#if defined(CATCH_INTERNAL_CONFIG_CPP17_OPTIONAL) && !defined(CATCH_CONFIG_NO_CPP17_OPTIONAL) && !defined(CATCH_CONFIG_CPP17_OPTIONAL)
#  define CATCH_CONFIG_CPP17_OPTIONAL
#endif

#if defined(CATCH_INTERNAL_CONFIG_CPP17_STRING_VIEW) && !defined(CATCH_CONFIG_NO_CPP17_STRING_VIEW) && !defined(CATCH_CONFIG_CPP17_STRING_VIEW)
#  define CATCH_CONFIG_CPP17_STRING_VIEW
#endif

#if defined(CATCH_INTERNAL_CONFIG_CPP17_VARIANT) && !defined(CATCH_CONFIG_NO_CPP17_VARIANT) && !defined(CATCH_CONFIG_CPP17_VARIANT)
#  define CATCH_CONFIG_CPP17_VARIANT
#endif

#if defined(CATCH_INTERNAL_CONFIG_CPP17_BYTE) && !defined(CATCH_CONFIG_NO_CPP17_BYTE) && !defined(CATCH_CONFIG_CPP17_BYTE)
#  define CATCH_CONFIG_CPP17_BYTE
#endif


#if defined(CATCH_CONFIG_EXPERIMENTAL_REDIRECT)
#  define CATCH_INTERNAL_CONFIG_NEW_CAPTURE
#endif

#if defined(CATCH_INTERNAL_CONFIG_NEW_CAPTURE) && !defined(CATCH_INTERNAL_CONFIG_NO_NEW_CAPTURE) && !defined(CATCH_CONFIG_NO_NEW_CAPTURE) && !defined(CATCH_CONFIG_NEW_CAPTURE)
#  define CATCH_CONFIG_NEW_CAPTURE
#endif

#if !defined( CATCH_INTERNAL_CONFIG_EXCEPTIONS_ENABLED ) && \
!defined( CATCH_CONFIG_DISABLE_EXCEPTIONS ) &&          \
!defined( CATCH_CONFIG_NO_DISABLE_EXCEPTIONS )
#  define CATCH_CONFIG_DISABLE_EXCEPTIONS
#endif

#if defined(CATCH_INTERNAL_CONFIG_POLYFILL_ISNAN) && !defined(CATCH_CONFIG_NO_POLYFILL_ISNAN) && !defined(CATCH_CONFIG_POLYFILL_ISNAN)
#  define CATCH_CONFIG_POLYFILL_ISNAN
#endif

#if defined(CATCH_INTERNAL_CONFIG_USE_ASYNC)  && !defined(CATCH_INTERNAL_CONFIG_NO_ASYNC) && !defined(CATCH_CONFIG_NO_USE_ASYNC) && !defined(CATCH_CONFIG_USE_ASYNC)
#  define CATCH_CONFIG_USE_ASYNC
#endif

#if defined(CATCH_INTERNAL_CONFIG_GLOBAL_NEXTAFTER) && !defined(CATCH_CONFIG_NO_GLOBAL_NEXTAFTER) && !defined(CATCH_CONFIG_GLOBAL_NEXTAFTER)
#  define CATCH_CONFIG_GLOBAL_NEXTAFTER
#endif


#if !defined(CATCH_INTERNAL_START_WARNINGS_SUPPRESSION)
#   define CATCH_INTERNAL_START_WARNINGS_SUPPRESSION
#endif
#if !defined(CATCH_INTERNAL_STOP_WARNINGS_SUPPRESSION)
#   define CATCH_INTERNAL_STOP_WARNINGS_SUPPRESSION
#endif
#if !defined(CATCH_INTERNAL_SUPPRESS_PARENTHESES_WARNINGS)
#   define CATCH_INTERNAL_SUPPRESS_PARENTHESES_WARNINGS
#endif
#if !defined(CATCH_INTERNAL_SUPPRESS_GLOBALS_WARNINGS)
#   define CATCH_INTERNAL_SUPPRESS_GLOBALS_WARNINGS
#endif
#if !defined(CATCH_INTERNAL_SUPPRESS_UNUSED_VARIABLE_WARNINGS)
#   define CATCH_INTERNAL_SUPPRESS_UNUSED_VARIABLE_WARNINGS
#endif
#if !defined(CATCH_INTERNAL_SUPPRESS_USELESS_CAST_WARNINGS)
#   define CATCH_INTERNAL_SUPPRESS_USELESS_CAST_WARNINGS
#endif
#if !defined(CATCH_INTERNAL_SUPPRESS_ZERO_VARIADIC_WARNINGS)
#   define CATCH_INTERNAL_SUPPRESS_ZERO_VARIADIC_WARNINGS
#endif

#if !defined(CATCH_INTERNAL_IGNORE_BUT_WARN)
#   define CATCH_INTERNAL_IGNORE_BUT_WARN(...)
#endif

#if defined(__APPLE__) && defined(__apple_build_version__) && (__clang_major__ < 10)
#   undef CATCH_INTERNAL_SUPPRESS_UNUSED_TEMPLATE_WARNINGS
#elif defined(__clang__) && (__clang_major__ < 5)
#   undef CATCH_INTERNAL_SUPPRESS_UNUSED_TEMPLATE_WARNINGS
#endif

#if !defined(CATCH_INTERNAL_SUPPRESS_UNUSED_TEMPLATE_WARNINGS)
#   define CATCH_INTERNAL_SUPPRESS_UNUSED_TEMPLATE_WARNINGS
#endif

#if !defined(CATCH_INTERNAL_SUPPRESS_COMMA_WARNINGS)
#   define CATCH_INTERNAL_SUPPRESS_COMMA_WARNINGS
#endif

#if defined(CATCH_CONFIG_DISABLE_EXCEPTIONS)
#define CATCH_TRY if ((true))
#define CATCH_CATCH_ALL if ((false))
#define CATCH_CATCH_ANON(type) if ((false))
#else
#define CATCH_TRY try
#define CATCH_CATCH_ALL catch (...)
#define CATCH_CATCH_ANON(type) catch (type)
#endif

#if defined(CATCH_INTERNAL_CONFIG_TRADITIONAL_MSVC_PREPROCESSOR) && !defined(CATCH_CONFIG_NO_TRADITIONAL_MSVC_PREPROCESSOR) && !defined(CATCH_CONFIG_TRADITIONAL_MSVC_PREPROCESSOR)
#define CATCH_CONFIG_TRADITIONAL_MSVC_PREPROCESSOR
#endif

#if defined( CATCH_PLATFORM_WINDOWS ) &&       \
!defined( CATCH_CONFIG_COLOUR_WIN32 ) && \
!defined( CATCH_CONFIG_NO_COLOUR_WIN32 ) && \
!defined( CATCH_INTERNAL_CONFIG_NO_COLOUR_WIN32 )
#    define CATCH_CONFIG_COLOUR_WIN32
#endif

#if defined( CATCH_CONFIG_SHARED_LIBRARY ) && defined( _MSC_VER ) && \
!defined( CATCH_CONFIG_STATIC )
#    ifdef Catch2_EXPORTS
#        define CATCH_EXPORT 
#    else
#        define CATCH_EXPORT __declspec( dllimport )
#    endif
#else
#    define CATCH_EXPORT
#endif

#endif 


#ifndef CATCH_CONTEXT_HPP_INCLUDED
#define CATCH_CONTEXT_HPP_INCLUDED


namespace Catch {

class IResultCapture;
class IConfig;

class IContext {
public:
virtual ~IContext(); 

virtual IResultCapture* getResultCapture() = 0;
virtual IConfig const* getConfig() const = 0;
};

class IMutableContext : public IContext {
public:
~IMutableContext() override; 
virtual void setResultCapture( IResultCapture* resultCapture ) = 0;
virtual void setConfig( IConfig const* config ) = 0;

private:
CATCH_EXPORT static IMutableContext* currentContext;
friend IMutableContext& getCurrentMutableContext();
friend void cleanUpContext();
static void createContext();
};

inline IMutableContext& getCurrentMutableContext()
{
if( !IMutableContext::currentContext )
IMutableContext::createContext();
return *IMutableContext::currentContext;
}

inline IContext& getCurrentContext()
{
return getCurrentMutableContext();
}

void cleanUpContext();

class SimplePcg32;
SimplePcg32& sharedRng();
}

#endif 


#ifndef CATCH_INTERFACES_REPORTER_HPP_INCLUDED
#define CATCH_INTERFACES_REPORTER_HPP_INCLUDED



#ifndef CATCH_SECTION_INFO_HPP_INCLUDED
#define CATCH_SECTION_INFO_HPP_INCLUDED



#ifndef CATCH_MOVE_AND_FORWARD_HPP_INCLUDED
#define CATCH_MOVE_AND_FORWARD_HPP_INCLUDED

#include <type_traits>

#define CATCH_MOVE(...) static_cast<std::remove_reference_t<decltype(__VA_ARGS__)>&&>(__VA_ARGS__)

#define CATCH_FORWARD(...) static_cast<decltype(__VA_ARGS__)&&>(__VA_ARGS__)

#endif 


#ifndef CATCH_SOURCE_LINE_INFO_HPP_INCLUDED
#define CATCH_SOURCE_LINE_INFO_HPP_INCLUDED

#include <cstddef>
#include <iosfwd>

namespace Catch {

struct SourceLineInfo {

SourceLineInfo() = delete;
constexpr SourceLineInfo( char const* _file, std::size_t _line ) noexcept:
file( _file ),
line( _line )
{}

bool operator == ( SourceLineInfo const& other ) const noexcept;
bool operator < ( SourceLineInfo const& other ) const noexcept;

char const* file;
std::size_t line;

friend std::ostream& operator << (std::ostream& os, SourceLineInfo const& info);
};
}

#define CATCH_INTERNAL_LINEINFO \
::Catch::SourceLineInfo( __FILE__, static_cast<std::size_t>( __LINE__ ) )

#endif 


#ifndef CATCH_TOTALS_HPP_INCLUDED
#define CATCH_TOTALS_HPP_INCLUDED

#include <cstdint>

namespace Catch {

struct Counts {
Counts operator - ( Counts const& other ) const;
Counts& operator += ( Counts const& other );

std::uint64_t total() const;
bool allPassed() const;
bool allOk() const;

std::uint64_t passed = 0;
std::uint64_t failed = 0;
std::uint64_t failedButOk = 0;
};

struct Totals {

Totals operator - ( Totals const& other ) const;
Totals& operator += ( Totals const& other );

Totals delta( Totals const& prevTotals ) const;

Counts assertions;
Counts testCases;
};
}

#endif 

#include <string>

namespace Catch {

struct SectionInfo {
SectionInfo( SourceLineInfo const& _lineInfo, std::string _name,
const char* const = nullptr ):
name(CATCH_MOVE(_name)),
lineInfo(_lineInfo)
{}

std::string name;
SourceLineInfo lineInfo;
};

struct SectionEndInfo {
SectionInfo sectionInfo;
Counts prevAssertions;
double durationInSeconds;
};

} 

#endif 


#ifndef CATCH_ASSERTION_RESULT_HPP_INCLUDED
#define CATCH_ASSERTION_RESULT_HPP_INCLUDED



#ifndef CATCH_ASSERTION_INFO_HPP_INCLUDED
#define CATCH_ASSERTION_INFO_HPP_INCLUDED



#ifndef CATCH_RESULT_TYPE_HPP_INCLUDED
#define CATCH_RESULT_TYPE_HPP_INCLUDED

namespace Catch {

struct ResultWas { enum OfType {
Unknown = -1,
Ok = 0,
Info = 1,
Warning = 2,

FailureBit = 0x10,

ExpressionFailed = FailureBit | 1,
ExplicitFailure = FailureBit | 2,

Exception = 0x100 | FailureBit,

ThrewException = Exception | 1,
DidntThrowException = Exception | 2,

FatalErrorCondition = 0x200 | FailureBit

}; };

bool isOk( ResultWas::OfType resultType );
bool isJustInfo( int flags );


struct ResultDisposition { enum Flags {
Normal = 0x01,

ContinueOnFailure = 0x02,   
FalseTest = 0x04,           
SuppressFail = 0x08         
}; };

ResultDisposition::Flags operator | ( ResultDisposition::Flags lhs, ResultDisposition::Flags rhs );

bool shouldContinueOnFailure( int flags );
inline bool isFalseTest( int flags ) { return ( flags & ResultDisposition::FalseTest ) != 0; }
bool shouldSuppressFailure( int flags );

} 

#endif 

namespace Catch {

struct AssertionInfo {

StringRef macroName;
SourceLineInfo lineInfo;
StringRef capturedExpression;
ResultDisposition::Flags resultDisposition;
};

} 

#endif 


#ifndef CATCH_LAZY_EXPR_HPP_INCLUDED
#define CATCH_LAZY_EXPR_HPP_INCLUDED

#include <iosfwd>

namespace Catch {

class ITransientExpression;

class LazyExpression {
friend class AssertionHandler;
friend struct AssertionStats;
friend class RunContext;

ITransientExpression const* m_transientExpression = nullptr;
bool m_isNegated;
public:
LazyExpression( bool isNegated ):
m_isNegated(isNegated)
{}
LazyExpression(LazyExpression const& other) = default;
LazyExpression& operator = ( LazyExpression const& ) = delete;

explicit operator bool() const {
return m_transientExpression != nullptr;
}

friend auto operator << ( std::ostream& os, LazyExpression const& lazyExpr ) -> std::ostream&;
};

} 

#endif 

#include <string>

namespace Catch {

struct AssertionResultData
{
AssertionResultData() = delete;

AssertionResultData( ResultWas::OfType _resultType, LazyExpression const& _lazyExpression );

std::string message;
mutable std::string reconstructedExpression;
LazyExpression lazyExpression;
ResultWas::OfType resultType;

std::string reconstructExpression() const;
};

class AssertionResult {
public:
AssertionResult() = delete;
AssertionResult( AssertionInfo const& info, AssertionResultData const& data );

bool isOk() const;
bool succeeded() const;
ResultWas::OfType getResultType() const;
bool hasExpression() const;
bool hasMessage() const;
std::string getExpression() const;
std::string getExpressionInMacro() const;
bool hasExpandedExpression() const;
std::string getExpandedExpression() const;
StringRef getMessage() const;
SourceLineInfo getSourceInfo() const;
StringRef getTestMacroName() const;

AssertionInfo m_info;
AssertionResultData m_resultData;
};

} 

#endif 


#ifndef CATCH_MESSAGE_INFO_HPP_INCLUDED
#define CATCH_MESSAGE_INFO_HPP_INCLUDED



#ifndef CATCH_INTERFACES_CAPTURE_HPP_INCLUDED
#define CATCH_INTERFACES_CAPTURE_HPP_INCLUDED

#include <string>
#include <chrono>


namespace Catch {

class AssertionResult;
struct AssertionInfo;
struct SectionInfo;
struct SectionEndInfo;
struct MessageInfo;
struct MessageBuilder;
struct Counts;
struct AssertionReaction;
struct SourceLineInfo;

class ITransientExpression;
class IGeneratorTracker;

struct BenchmarkInfo;
template <typename Duration = std::chrono::duration<double, std::nano>>
struct BenchmarkStats;

class IResultCapture {
public:
virtual ~IResultCapture();

virtual bool sectionStarted(    SectionInfo const& sectionInfo,
Counts& assertions ) = 0;
virtual void sectionEnded( SectionEndInfo const& endInfo ) = 0;
virtual void sectionEndedEarly( SectionEndInfo const& endInfo ) = 0;

virtual auto acquireGeneratorTracker( StringRef generatorName, SourceLineInfo const& lineInfo ) -> IGeneratorTracker& = 0;

virtual void benchmarkPreparing( StringRef name ) = 0;
virtual void benchmarkStarting( BenchmarkInfo const& info ) = 0;
virtual void benchmarkEnded( BenchmarkStats<> const& stats ) = 0;
virtual void benchmarkFailed( StringRef error ) = 0;

virtual void pushScopedMessage( MessageInfo const& message ) = 0;
virtual void popScopedMessage( MessageInfo const& message ) = 0;

virtual void emplaceUnscopedMessage( MessageBuilder const& builder ) = 0;

virtual void handleFatalErrorCondition( StringRef message ) = 0;

virtual void handleExpr
(   AssertionInfo const& info,
ITransientExpression const& expr,
AssertionReaction& reaction ) = 0;
virtual void handleMessage
(   AssertionInfo const& info,
ResultWas::OfType resultType,
StringRef message,
AssertionReaction& reaction ) = 0;
virtual void handleUnexpectedExceptionNotThrown
(   AssertionInfo const& info,
AssertionReaction& reaction ) = 0;
virtual void handleUnexpectedInflightException
(   AssertionInfo const& info,
std::string const& message,
AssertionReaction& reaction ) = 0;
virtual void handleIncomplete
(   AssertionInfo const& info ) = 0;
virtual void handleNonExpr
(   AssertionInfo const &info,
ResultWas::OfType resultType,
AssertionReaction &reaction ) = 0;



virtual bool lastAssertionPassed() = 0;
virtual void assertionPassed() = 0;

virtual std::string getCurrentTestName() const = 0;
virtual const AssertionResult* getLastResult() const = 0;
virtual void exceptionEarlyReported() = 0;
};

IResultCapture& getResultCapture();
}

#endif 

#include <string>

namespace Catch {

struct MessageInfo {
MessageInfo(    StringRef _macroName,
SourceLineInfo const& _lineInfo,
ResultWas::OfType _type );

StringRef macroName;
std::string message;
SourceLineInfo lineInfo;
ResultWas::OfType type;
unsigned int sequence;

bool operator == (MessageInfo const& other) const {
return sequence == other.sequence;
}
bool operator < (MessageInfo const& other) const {
return sequence < other.sequence;
}
private:
static unsigned int globalCount;
};

} 

#endif 


#ifndef CATCH_UNIQUE_PTR_HPP_INCLUDED
#define CATCH_UNIQUE_PTR_HPP_INCLUDED

#include <cassert>
#include <type_traits>


namespace Catch {
namespace Detail {

template <typename T>
class unique_ptr {
T* m_ptr;
public:
constexpr unique_ptr(std::nullptr_t = nullptr):
m_ptr{}
{}
explicit constexpr unique_ptr(T* ptr):
m_ptr(ptr)
{}

template <typename U, typename = std::enable_if_t<std::is_base_of<T, U>::value>>
unique_ptr(unique_ptr<U>&& from):
m_ptr(from.release())
{}

template <typename U, typename = std::enable_if_t<std::is_base_of<T, U>::value>>
unique_ptr& operator=(unique_ptr<U>&& from) {
reset(from.release());

return *this;
}

unique_ptr(unique_ptr const&) = delete;
unique_ptr& operator=(unique_ptr const&) = delete;

unique_ptr(unique_ptr&& rhs) noexcept:
m_ptr(rhs.m_ptr) {
rhs.m_ptr = nullptr;
}
unique_ptr& operator=(unique_ptr&& rhs) noexcept {
reset(rhs.release());

return *this;
}

~unique_ptr() {
delete m_ptr;
}

T& operator*() {
assert(m_ptr);
return *m_ptr;
}
T const& operator*() const {
assert(m_ptr);
return *m_ptr;
}
T* operator->() noexcept {
assert(m_ptr);
return m_ptr;
}
T const* operator->() const noexcept {
assert(m_ptr);
return m_ptr;
}

T* get() { return m_ptr; }
T const* get() const { return m_ptr; }

void reset(T* ptr = nullptr) {
delete m_ptr;
m_ptr = ptr;
}

T* release() {
auto temp = m_ptr;
m_ptr = nullptr;
return temp;
}

explicit operator bool() const {
return m_ptr;
}

friend void swap(unique_ptr& lhs, unique_ptr& rhs) {
auto temp = lhs.m_ptr;
lhs.m_ptr = rhs.m_ptr;
rhs.m_ptr = temp;
}
};

template <typename T>
class unique_ptr<T[]>;

template <typename T, typename... Args>
unique_ptr<T> make_unique(Args&&... args) {
return unique_ptr<T>(new T(CATCH_FORWARD(args)...));
}


} 
} 

#endif 



#ifndef CATCH_ESTIMATE_HPP_INCLUDED
#define CATCH_ESTIMATE_HPP_INCLUDED

namespace Catch {
namespace Benchmark {
template <typename Duration>
struct Estimate {
Duration point;
Duration lower_bound;
Duration upper_bound;
double confidence_interval;

template <typename Duration2>
operator Estimate<Duration2>() const {
return { point, lower_bound, upper_bound, confidence_interval };
}
};
} 
} 

#endif 



#ifndef CATCH_OUTLIER_CLASSIFICATION_HPP_INCLUDED
#define CATCH_OUTLIER_CLASSIFICATION_HPP_INCLUDED

namespace Catch {
namespace Benchmark {
struct OutlierClassification {
int samples_seen = 0;
int low_severe = 0;     
int low_mild = 0;       
int high_mild = 0;      
int high_severe = 0;    

int total() const {
return low_severe + low_mild + high_mild + high_severe;
}
};
} 
} 

#endif 


#include <map>
#include <string>
#include <vector>
#include <iosfwd>

namespace Catch {

struct ReporterDescription;
struct ListenerDescription;
struct TagInfo;
struct TestCaseInfo;
class TestCaseHandle;
class IConfig;
class IStream;
enum class ColourMode : std::uint8_t;

struct ReporterConfig {
ReporterConfig( IConfig const* _fullConfig,
Detail::unique_ptr<IStream> _stream,
ColourMode colourMode,
std::map<std::string, std::string> customOptions );

ReporterConfig( ReporterConfig&& ) = default;
ReporterConfig& operator=( ReporterConfig&& ) = default;
~ReporterConfig(); 

Detail::unique_ptr<IStream> takeStream() &&;
IConfig const* fullConfig() const;
ColourMode colourMode() const;
std::map<std::string, std::string> const& customOptions() const;

private:
Detail::unique_ptr<IStream> m_stream;
IConfig const* m_fullConfig;
ColourMode m_colourMode;
std::map<std::string, std::string> m_customOptions;
};

struct TestRunInfo {
constexpr TestRunInfo(StringRef _name) : name(_name) {}
StringRef name;
};

struct AssertionStats {
AssertionStats( AssertionResult const& _assertionResult,
std::vector<MessageInfo> const& _infoMessages,
Totals const& _totals );

AssertionStats( AssertionStats const& )              = default;
AssertionStats( AssertionStats && )                  = default;
AssertionStats& operator = ( AssertionStats const& ) = delete;
AssertionStats& operator = ( AssertionStats && )     = delete;

AssertionResult assertionResult;
std::vector<MessageInfo> infoMessages;
Totals totals;
};

struct SectionStats {
SectionStats(   SectionInfo const& _sectionInfo,
Counts const& _assertions,
double _durationInSeconds,
bool _missingAssertions );

SectionInfo sectionInfo;
Counts assertions;
double durationInSeconds;
bool missingAssertions;
};

struct TestCaseStats {
TestCaseStats(  TestCaseInfo const& _testInfo,
Totals const& _totals,
std::string const& _stdOut,
std::string const& _stdErr,
bool _aborting );

TestCaseInfo const * testInfo;
Totals totals;
std::string stdOut;
std::string stdErr;
bool aborting;
};

struct TestRunStats {
TestRunStats(   TestRunInfo const& _runInfo,
Totals const& _totals,
bool _aborting );

TestRunInfo runInfo;
Totals totals;
bool aborting;
};


struct BenchmarkInfo {
std::string name;
double estimatedDuration;
int iterations;
unsigned int samples;
unsigned int resamples;
double clockResolution;
double clockCost;
};

template <class Duration>
struct BenchmarkStats {
BenchmarkInfo info;

std::vector<Duration> samples;
Benchmark::Estimate<Duration> mean;
Benchmark::Estimate<Duration> standardDeviation;
Benchmark::OutlierClassification outliers;
double outlierVariance;

template <typename Duration2>
operator BenchmarkStats<Duration2>() const {
std::vector<Duration2> samples2;
samples2.reserve(samples.size());
for (auto const& sample : samples) {
samples2.push_back(Duration2(sample));
}
return {
info,
CATCH_MOVE(samples2),
mean,
standardDeviation,
outliers,
outlierVariance,
};
}
};

struct ReporterPreferences {
bool shouldRedirectStdOut = false;
bool shouldReportAllAssertions = false;
};


class IEventListener {
protected:
ReporterPreferences m_preferences;
IConfig const* m_config;

public:
IEventListener( IConfig const* config ): m_config( config ) {}

virtual ~IEventListener(); 


ReporterPreferences const& getPreferences() const {
return m_preferences;
}

virtual void noMatchingTestCases( StringRef unmatchedSpec ) = 0;
virtual void reportInvalidTestSpec( StringRef invalidArgument ) = 0;


virtual void testRunStarting( TestRunInfo const& testRunInfo ) = 0;

virtual void testCaseStarting( TestCaseInfo const& testInfo ) = 0;
virtual void testCasePartialStarting( TestCaseInfo const& testInfo, uint64_t partNumber ) = 0;
virtual void sectionStarting( SectionInfo const& sectionInfo ) = 0;

virtual void benchmarkPreparing( StringRef benchmarkName ) = 0;
virtual void benchmarkStarting( BenchmarkInfo const& benchmarkInfo ) = 0;
virtual void benchmarkEnded( BenchmarkStats<> const& benchmarkStats ) = 0;
virtual void benchmarkFailed( StringRef benchmarkName ) = 0;

virtual void assertionStarting( AssertionInfo const& assertionInfo ) = 0;

virtual void assertionEnded( AssertionStats const& assertionStats ) = 0;

virtual void sectionEnded( SectionStats const& sectionStats ) = 0;
virtual void testCasePartialEnded(TestCaseStats const& testCaseStats, uint64_t partNumber ) = 0;
virtual void testCaseEnded( TestCaseStats const& testCaseStats ) = 0;

virtual void testRunEnded( TestRunStats const& testRunStats ) = 0;

virtual void skipTest( TestCaseInfo const& testInfo ) = 0;

virtual void fatalErrorEncountered( StringRef error ) = 0;

virtual void listReporters(std::vector<ReporterDescription> const& descriptions) = 0;
virtual void listListeners(std::vector<ListenerDescription> const& descriptions) = 0;
virtual void listTests(std::vector<TestCaseHandle> const& tests) = 0;
virtual void listTags(std::vector<TagInfo> const& tags) = 0;
};
using IEventListenerPtr = Detail::unique_ptr<IEventListener>;

} 

#endif 


#ifndef CATCH_UNIQUE_NAME_HPP_INCLUDED
#define CATCH_UNIQUE_NAME_HPP_INCLUDED






#ifndef CATCH_CONFIG_COUNTER_HPP_INCLUDED
#define CATCH_CONFIG_COUNTER_HPP_INCLUDED

#if ( !defined(__JETBRAINS_IDE__) || __JETBRAINS_IDE__ >= 20170300L )
#define CATCH_INTERNAL_CONFIG_COUNTER
#endif

#if defined( CATCH_INTERNAL_CONFIG_COUNTER ) && \
!defined( CATCH_CONFIG_NO_COUNTER ) && \
!defined( CATCH_CONFIG_COUNTER )
#    define CATCH_CONFIG_COUNTER
#endif


#endif 
#define INTERNAL_CATCH_UNIQUE_NAME_LINE2( name, line ) name##line
#define INTERNAL_CATCH_UNIQUE_NAME_LINE( name, line ) INTERNAL_CATCH_UNIQUE_NAME_LINE2( name, line )
#ifdef CATCH_CONFIG_COUNTER
#  define INTERNAL_CATCH_UNIQUE_NAME( name ) INTERNAL_CATCH_UNIQUE_NAME_LINE( name, __COUNTER__ )
#else
#  define INTERNAL_CATCH_UNIQUE_NAME( name ) INTERNAL_CATCH_UNIQUE_NAME_LINE( name, __LINE__ )
#endif

#endif 



#ifndef CATCH_CHRONOMETER_HPP_INCLUDED
#define CATCH_CHRONOMETER_HPP_INCLUDED




#ifndef CATCH_CLOCK_HPP_INCLUDED
#define CATCH_CLOCK_HPP_INCLUDED

#include <chrono>
#include <ratio>

namespace Catch {
namespace Benchmark {
template <typename Clock>
using ClockDuration = typename Clock::duration;
template <typename Clock>
using FloatDuration = std::chrono::duration<double, typename Clock::period>;

template <typename Clock>
using TimePoint = typename Clock::time_point;

using default_clock = std::chrono::steady_clock;

template <typename Clock>
struct now {
TimePoint<Clock> operator()() const {
return Clock::now();
}
};

using fp_seconds = std::chrono::duration<double, std::ratio<1>>;
} 
} 

#endif 



#ifndef CATCH_OPTIMIZER_HPP_INCLUDED
#define CATCH_OPTIMIZER_HPP_INCLUDED

#if defined(_MSC_VER)
#   include <atomic> 
#endif


#include <type_traits>

namespace Catch {
namespace Benchmark {
#if defined(__GNUC__) || defined(__clang__)
template <typename T>
inline void keep_memory(T* p) {
asm volatile("" : : "g"(p) : "memory");
}
inline void keep_memory() {
asm volatile("" : : : "memory");
}

namespace Detail {
inline void optimizer_barrier() { keep_memory(); }
} 
#elif defined(_MSC_VER)

#pragma optimize("", off)
template <typename T>
inline void keep_memory(T* p) {
*reinterpret_cast<char volatile*>(p) = *reinterpret_cast<char const volatile*>(p);
}
#pragma optimize("", on)

namespace Detail {
inline void optimizer_barrier() {
std::atomic_thread_fence(std::memory_order_seq_cst);
}
} 

#endif

template <typename T>
inline void deoptimize_value(T&& x) {
keep_memory(&x);
}

template <typename Fn, typename... Args>
inline auto invoke_deoptimized(Fn&& fn, Args&&... args) -> std::enable_if_t<!std::is_same<void, decltype(fn(args...))>::value> {
deoptimize_value(CATCH_FORWARD(fn) (CATCH_FORWARD(args)...));
}

template <typename Fn, typename... Args>
inline auto invoke_deoptimized(Fn&& fn, Args&&... args) -> std::enable_if_t<std::is_same<void, decltype(fn(args...))>::value> {
CATCH_FORWARD(fn) (CATCH_FORWARD(args)...);
}
} 
} 

#endif 



#ifndef CATCH_COMPLETE_INVOKE_HPP_INCLUDED
#define CATCH_COMPLETE_INVOKE_HPP_INCLUDED



#ifndef CATCH_TEST_FAILURE_EXCEPTION_HPP_INCLUDED
#define CATCH_TEST_FAILURE_EXCEPTION_HPP_INCLUDED

namespace Catch {

struct TestFailureException{};


[[noreturn]] void throw_test_failure_exception();

} 

#endif 


#ifndef CATCH_META_HPP_INCLUDED
#define CATCH_META_HPP_INCLUDED

#include <type_traits>

namespace Catch {
template <typename>
struct true_given : std::true_type {};

struct is_callable_tester {
template <typename Fun, typename... Args>
static true_given<decltype(std::declval<Fun>()(std::declval<Args>()...))> test(int);
template <typename...>
static std::false_type test(...);
};

template <typename T>
struct is_callable;

template <typename Fun, typename... Args>
struct is_callable<Fun(Args...)> : decltype(is_callable_tester::test<Fun, Args...>(0)) {};


#if defined(__cpp_lib_is_invocable) && __cpp_lib_is_invocable >= 201703
template <typename Func, typename... U>
using FunctionReturnType = std::remove_reference_t<std::remove_cv_t<std::invoke_result_t<Func, U...>>>;
#else
template <typename Func, typename... U>
using FunctionReturnType = std::remove_reference_t<std::remove_cv_t<std::result_of_t<Func(U...)>>>;
#endif

} 

namespace mpl_{
struct na;
}

#endif 


#ifndef CATCH_INTERFACES_REGISTRY_HUB_HPP_INCLUDED
#define CATCH_INTERFACES_REGISTRY_HUB_HPP_INCLUDED


#include <string>

namespace Catch {

class TestCaseHandle;
struct TestCaseInfo;
class ITestCaseRegistry;
class IExceptionTranslatorRegistry;
class IExceptionTranslator;
class IReporterRegistry;
class IReporterFactory;
class ITagAliasRegistry;
class ITestInvoker;
class IMutableEnumValuesRegistry;
struct SourceLineInfo;

class StartupExceptionRegistry;
class EventListenerFactory;

using IReporterFactoryPtr = Detail::unique_ptr<IReporterFactory>;

class IRegistryHub {
public:
virtual ~IRegistryHub(); 

virtual IReporterRegistry const& getReporterRegistry() const = 0;
virtual ITestCaseRegistry const& getTestCaseRegistry() const = 0;
virtual ITagAliasRegistry const& getTagAliasRegistry() const = 0;
virtual IExceptionTranslatorRegistry const& getExceptionTranslatorRegistry() const = 0;


virtual StartupExceptionRegistry const& getStartupExceptionRegistry() const = 0;
};

class IMutableRegistryHub {
public:
virtual ~IMutableRegistryHub(); 
virtual void registerReporter( std::string const& name, IReporterFactoryPtr factory ) = 0;
virtual void registerListener( Detail::unique_ptr<EventListenerFactory> factory ) = 0;
virtual void registerTest(Detail::unique_ptr<TestCaseInfo>&& testInfo, Detail::unique_ptr<ITestInvoker>&& invoker) = 0;
virtual void registerTranslator( Detail::unique_ptr<IExceptionTranslator>&& translator ) = 0;
virtual void registerTagAlias( std::string const& alias, std::string const& tag, SourceLineInfo const& lineInfo ) = 0;
virtual void registerStartupException() noexcept = 0;
virtual IMutableEnumValuesRegistry& getMutableEnumValuesRegistry() = 0;
};

IRegistryHub const& getRegistryHub();
IMutableRegistryHub& getMutableRegistryHub();
void cleanUp();
std::string translateActiveException();

}

#endif 

#include <type_traits>

namespace Catch {
namespace Benchmark {
namespace Detail {
template <typename T>
struct CompleteType { using type = T; };
template <>
struct CompleteType<void> { struct type {}; };

template <typename T>
using CompleteType_t = typename CompleteType<T>::type;

template <typename Result>
struct CompleteInvoker {
template <typename Fun, typename... Args>
static Result invoke(Fun&& fun, Args&&... args) {
return CATCH_FORWARD(fun)(CATCH_FORWARD(args)...);
}
};
template <>
struct CompleteInvoker<void> {
template <typename Fun, typename... Args>
static CompleteType_t<void> invoke(Fun&& fun, Args&&... args) {
CATCH_FORWARD(fun)(CATCH_FORWARD(args)...);
return {};
}
};

template <typename Fun, typename... Args>
CompleteType_t<FunctionReturnType<Fun, Args...>> complete_invoke(Fun&& fun, Args&&... args) {
return CompleteInvoker<FunctionReturnType<Fun, Args...>>::invoke(CATCH_FORWARD(fun), CATCH_FORWARD(args)...);
}

} 

template <typename Fun>
Detail::CompleteType_t<FunctionReturnType<Fun>> user_code(Fun&& fun) {
return Detail::complete_invoke(CATCH_FORWARD(fun));
}
} 
} 

#endif 

namespace Catch {
namespace Benchmark {
namespace Detail {
struct ChronometerConcept {
virtual void start() = 0;
virtual void finish() = 0;
virtual ~ChronometerConcept(); 

ChronometerConcept() = default;
ChronometerConcept(ChronometerConcept const&) = default;
ChronometerConcept& operator=(ChronometerConcept const&) = default;
};
template <typename Clock>
struct ChronometerModel final : public ChronometerConcept {
void start() override { started = Clock::now(); }
void finish() override { finished = Clock::now(); }

ClockDuration<Clock> elapsed() const { return finished - started; }

TimePoint<Clock> started;
TimePoint<Clock> finished;
};
} 

struct Chronometer {
public:
template <typename Fun>
void measure(Fun&& fun) { measure(CATCH_FORWARD(fun), is_callable<Fun(int)>()); }

int runs() const { return repeats; }

Chronometer(Detail::ChronometerConcept& meter, int repeats_)
: impl(&meter)
, repeats(repeats_) {}

private:
template <typename Fun>
void measure(Fun&& fun, std::false_type) {
measure([&fun](int) { return fun(); }, std::true_type());
}

template <typename Fun>
void measure(Fun&& fun, std::true_type) {
Detail::optimizer_barrier();
impl->start();
for (int i = 0; i < repeats; ++i) invoke_deoptimized(fun, i);
impl->finish();
Detail::optimizer_barrier();
}

Detail::ChronometerConcept* impl;
int repeats;
};
} 
} 

#endif 



#ifndef CATCH_ENVIRONMENT_HPP_INCLUDED
#define CATCH_ENVIRONMENT_HPP_INCLUDED


namespace Catch {
namespace Benchmark {
template <typename Duration>
struct EnvironmentEstimate {
Duration mean;
OutlierClassification outliers;

template <typename Duration2>
operator EnvironmentEstimate<Duration2>() const {
return { mean, outliers };
}
};
template <typename Clock>
struct Environment {
using clock_type = Clock;
EnvironmentEstimate<FloatDuration<Clock>> clock_resolution;
EnvironmentEstimate<FloatDuration<Clock>> clock_cost;
};
} 
} 

#endif 



#ifndef CATCH_EXECUTION_PLAN_HPP_INCLUDED
#define CATCH_EXECUTION_PLAN_HPP_INCLUDED




#ifndef CATCH_BENCHMARK_FUNCTION_HPP_INCLUDED
#define CATCH_BENCHMARK_FUNCTION_HPP_INCLUDED


#include <type_traits>

namespace Catch {
namespace Benchmark {
namespace Detail {
template <typename T, typename U>
struct is_related
: std::is_same<std::decay_t<T>, std::decay_t<U>> {};

struct BenchmarkFunction {
private:
struct callable {
virtual void call(Chronometer meter) const = 0;
virtual Catch::Detail::unique_ptr<callable> clone() const = 0;
virtual ~callable(); 

callable() = default;
callable(callable const&) = default;
callable& operator=(callable const&) = default;
};
template <typename Fun>
struct model : public callable {
model(Fun&& fun_) : fun(CATCH_MOVE(fun_)) {}
model(Fun const& fun_) : fun(fun_) {}

Catch::Detail::unique_ptr<callable> clone() const override {
return Catch::Detail::make_unique<model<Fun>>( *this );
}

void call(Chronometer meter) const override {
call(meter, is_callable<Fun(Chronometer)>());
}
void call(Chronometer meter, std::true_type) const {
fun(meter);
}
void call(Chronometer meter, std::false_type) const {
meter.measure(fun);
}

Fun fun;
};

struct do_nothing { void operator()() const {} };

template <typename T>
BenchmarkFunction(model<T>* c) : f(c) {}

public:
BenchmarkFunction()
: f(new model<do_nothing>{ {} }) {}

template <typename Fun,
std::enable_if_t<!is_related<Fun, BenchmarkFunction>::value, int> = 0>
BenchmarkFunction(Fun&& fun)
: f(new model<std::decay_t<Fun>>(CATCH_FORWARD(fun))) {}

BenchmarkFunction( BenchmarkFunction&& that ) noexcept:
f( CATCH_MOVE( that.f ) ) {}

BenchmarkFunction(BenchmarkFunction const& that)
: f(that.f->clone()) {}

BenchmarkFunction&
operator=( BenchmarkFunction&& that ) noexcept {
f = CATCH_MOVE( that.f );
return *this;
}

BenchmarkFunction& operator=(BenchmarkFunction const& that) {
f = that.f->clone();
return *this;
}

void operator()(Chronometer meter) const { f->call(meter); }

private:
Catch::Detail::unique_ptr<callable> f;
};
} 
} 
} 

#endif 



#ifndef CATCH_REPEAT_HPP_INCLUDED
#define CATCH_REPEAT_HPP_INCLUDED

#include <type_traits>

namespace Catch {
namespace Benchmark {
namespace Detail {
template <typename Fun>
struct repeater {
void operator()(int k) const {
for (int i = 0; i < k; ++i) {
fun();
}
}
Fun fun;
};
template <typename Fun>
repeater<std::decay_t<Fun>> repeat(Fun&& fun) {
return { CATCH_FORWARD(fun) };
}
} 
} 
} 

#endif 



#ifndef CATCH_RUN_FOR_AT_LEAST_HPP_INCLUDED
#define CATCH_RUN_FOR_AT_LEAST_HPP_INCLUDED




#ifndef CATCH_MEASURE_HPP_INCLUDED
#define CATCH_MEASURE_HPP_INCLUDED




#ifndef CATCH_TIMING_HPP_INCLUDED
#define CATCH_TIMING_HPP_INCLUDED


#include <type_traits>

namespace Catch {
namespace Benchmark {
template <typename Duration, typename Result>
struct Timing {
Duration elapsed;
Result result;
int iterations;
};
template <typename Clock, typename Func, typename... Args>
using TimingOf = Timing<ClockDuration<Clock>, Detail::CompleteType_t<FunctionReturnType<Func, Args...>>>;
} 
} 

#endif 

namespace Catch {
namespace Benchmark {
namespace Detail {
template <typename Clock, typename Fun, typename... Args>
TimingOf<Clock, Fun, Args...> measure(Fun&& fun, Args&&... args) {
auto start = Clock::now();
auto&& r = Detail::complete_invoke(fun, CATCH_FORWARD(args)...);
auto end = Clock::now();
auto delta = end - start;
return { delta, CATCH_FORWARD(r), 1 };
}
} 
} 
} 

#endif 

#include <type_traits>

namespace Catch {
namespace Benchmark {
namespace Detail {
template <typename Clock, typename Fun>
TimingOf<Clock, Fun, int> measure_one(Fun&& fun, int iters, std::false_type) {
return Detail::measure<Clock>(fun, iters);
}
template <typename Clock, typename Fun>
TimingOf<Clock, Fun, Chronometer> measure_one(Fun&& fun, int iters, std::true_type) {
Detail::ChronometerModel<Clock> meter;
auto&& result = Detail::complete_invoke(fun, Chronometer(meter, iters));

return { meter.elapsed(), CATCH_MOVE(result), iters };
}

template <typename Clock, typename Fun>
using run_for_at_least_argument_t = std::conditional_t<is_callable<Fun(Chronometer)>::value, Chronometer, int>;


[[noreturn]]
void throw_optimized_away_error();

template <typename Clock, typename Fun>
TimingOf<Clock, Fun, run_for_at_least_argument_t<Clock, Fun>>
run_for_at_least(ClockDuration<Clock> how_long,
const int initial_iterations,
Fun&& fun) {
auto iters = initial_iterations;
while (iters < (1 << 30)) {
auto&& Timing = measure_one<Clock>(fun, iters, is_callable<Fun(Chronometer)>());

if (Timing.elapsed >= how_long) {
return { Timing.elapsed, CATCH_MOVE(Timing.result), iters };
}
iters *= 2;
}
throw_optimized_away_error();
}
} 
} 
} 

#endif 

#include <algorithm>
#include <iterator>

namespace Catch {
namespace Benchmark {
template <typename Duration>
struct ExecutionPlan {
int iterations_per_sample;
Duration estimated_duration;
Detail::BenchmarkFunction benchmark;
Duration warmup_time;
int warmup_iterations;

template <typename Duration2>
operator ExecutionPlan<Duration2>() const {
return { iterations_per_sample, estimated_duration, benchmark, warmup_time, warmup_iterations };
}

template <typename Clock>
std::vector<FloatDuration<Clock>> run(const IConfig &cfg, Environment<FloatDuration<Clock>> env) const {
Detail::run_for_at_least<Clock>(std::chrono::duration_cast<ClockDuration<Clock>>(warmup_time), warmup_iterations, Detail::repeat(now<Clock>{}));

std::vector<FloatDuration<Clock>> times;
times.reserve(cfg.benchmarkSamples());
std::generate_n(std::back_inserter(times), cfg.benchmarkSamples(), [this, env] {
Detail::ChronometerModel<Clock> model;
this->benchmark(Chronometer(model, iterations_per_sample));
auto sample_time = model.elapsed() - env.clock_cost.mean;
if (sample_time < FloatDuration<Clock>::zero()) sample_time = FloatDuration<Clock>::zero();
return sample_time / iterations_per_sample;
});
return times;
}
};
} 
} 

#endif 



#ifndef CATCH_ESTIMATE_CLOCK_HPP_INCLUDED
#define CATCH_ESTIMATE_CLOCK_HPP_INCLUDED




#ifndef CATCH_STATS_HPP_INCLUDED
#define CATCH_STATS_HPP_INCLUDED


#include <algorithm>
#include <vector>
#include <numeric>
#include <tuple>
#include <cmath>

namespace Catch {
namespace Benchmark {
namespace Detail {
using sample = std::vector<double>;

bool directCompare( double lhs, double rhs );

double weighted_average_quantile(int k, int q, std::vector<double>::iterator first, std::vector<double>::iterator last);

template <typename Iterator>
OutlierClassification classify_outliers(Iterator first, Iterator last) {
std::vector<double> copy(first, last);

auto q1 = weighted_average_quantile(1, 4, copy.begin(), copy.end());
auto q3 = weighted_average_quantile(3, 4, copy.begin(), copy.end());
auto iqr = q3 - q1;
auto los = q1 - (iqr * 3.);
auto lom = q1 - (iqr * 1.5);
auto him = q3 + (iqr * 1.5);
auto his = q3 + (iqr * 3.);

OutlierClassification o;
for (; first != last; ++first) {
auto&& t = *first;
if (t < los) ++o.low_severe;
else if (t < lom) ++o.low_mild;
else if (t > his) ++o.high_severe;
else if (t > him) ++o.high_mild;
++o.samples_seen;
}
return o;
}

template <typename Iterator>
double mean(Iterator first, Iterator last) {
auto count = last - first;
double sum = std::accumulate(first, last, 0.);
return sum / static_cast<double>(count);
}

template <typename Estimator, typename Iterator>
sample jackknife(Estimator&& estimator, Iterator first, Iterator last) {
auto n = static_cast<size_t>(last - first);
auto second = first;
++second;
sample results;
results.reserve(n);

for (auto it = first; it != last; ++it) {
std::iter_swap(it, first);
results.push_back(estimator(second, last));
}

return results;
}

inline double normal_cdf(double x) {
return std::erfc(-x / std::sqrt(2.0)) / 2.0;
}

double erfc_inv(double x);

double normal_quantile(double p);

template <typename Iterator, typename Estimator>
Estimate<double> bootstrap(double confidence_level, Iterator first, Iterator last, sample const& resample, Estimator&& estimator) {
auto n_samples = last - first;

double point = estimator(first, last);
if (n_samples == 1) return { point, point, point, confidence_level };

sample jack = jackknife(estimator, first, last);
double jack_mean = mean(jack.begin(), jack.end());
double sum_squares, sum_cubes;
std::tie(sum_squares, sum_cubes) = std::accumulate(jack.begin(), jack.end(), std::make_pair(0., 0.), [jack_mean](std::pair<double, double> sqcb, double x) -> std::pair<double, double> {
auto d = jack_mean - x;
auto d2 = d * d;
auto d3 = d2 * d;
return { sqcb.first + d2, sqcb.second + d3 };
});

double accel = sum_cubes / (6 * std::pow(sum_squares, 1.5));
long n = static_cast<long>(resample.size());
double prob_n = std::count_if(resample.begin(), resample.end(), [point](double x) { return x < point; }) / static_cast<double>(n);
if ( directCompare( prob_n, 0. ) ) {
return { point, point, point, confidence_level };
}

double bias = normal_quantile(prob_n);
double z1 = normal_quantile((1. - confidence_level) / 2.);

auto cumn = [n]( double x ) -> long {
return std::lround( normal_cdf( x ) * static_cast<double>(n) );
};
auto a = [bias, accel](double b) { return bias + b / (1. - accel * b); };
double b1 = bias + z1;
double b2 = bias - z1;
double a1 = a(b1);
double a2 = a(b2);
auto lo = static_cast<size_t>((std::max)(cumn(a1), 0l));
auto hi = static_cast<size_t>((std::min)(cumn(a2), n - 1));

return { point, resample[lo], resample[hi], confidence_level };
}

double outlier_variance(Estimate<double> mean, Estimate<double> stddev, int n);

struct bootstrap_analysis {
Estimate<double> mean;
Estimate<double> standard_deviation;
double outlier_variance;
};

bootstrap_analysis analyse_samples(double confidence_level, unsigned int n_resamples, std::vector<double>::iterator first, std::vector<double>::iterator last);
} 
} 
} 

#endif 

#include <algorithm>
#include <iterator>
#include <vector>
#include <cmath>

namespace Catch {
namespace Benchmark {
namespace Detail {
template <typename Clock>
std::vector<double> resolution(int k) {
std::vector<TimePoint<Clock>> times;
times.reserve(static_cast<size_t>(k + 1));
std::generate_n(std::back_inserter(times), k + 1, now<Clock>{});

std::vector<double> deltas;
deltas.reserve(static_cast<size_t>(k));
std::transform(std::next(times.begin()), times.end(), times.begin(),
std::back_inserter(deltas),
[](TimePoint<Clock> a, TimePoint<Clock> b) { return static_cast<double>((a - b).count()); });

return deltas;
}

const auto warmup_iterations = 10000;
const auto warmup_time = std::chrono::milliseconds(100);
const auto minimum_ticks = 1000;
const auto warmup_seed = 10000;
const auto clock_resolution_estimation_time = std::chrono::milliseconds(500);
const auto clock_cost_estimation_time_limit = std::chrono::seconds(1);
const auto clock_cost_estimation_tick_limit = 100000;
const auto clock_cost_estimation_time = std::chrono::milliseconds(10);
const auto clock_cost_estimation_iterations = 10000;

template <typename Clock>
int warmup() {
return run_for_at_least<Clock>(std::chrono::duration_cast<ClockDuration<Clock>>(warmup_time), warmup_seed, &resolution<Clock>)
.iterations;
}
template <typename Clock>
EnvironmentEstimate<FloatDuration<Clock>> estimate_clock_resolution(int iterations) {
auto r = run_for_at_least<Clock>(std::chrono::duration_cast<ClockDuration<Clock>>(clock_resolution_estimation_time), iterations, &resolution<Clock>)
.result;
return {
FloatDuration<Clock>(mean(r.begin(), r.end())),
classify_outliers(r.begin(), r.end()),
};
}
template <typename Clock>
EnvironmentEstimate<FloatDuration<Clock>> estimate_clock_cost(FloatDuration<Clock> resolution) {
auto time_limit = (std::min)(
resolution * clock_cost_estimation_tick_limit,
FloatDuration<Clock>(clock_cost_estimation_time_limit));
auto time_clock = [](int k) {
return Detail::measure<Clock>([k] {
for (int i = 0; i < k; ++i) {
volatile auto ignored = Clock::now();
(void)ignored;
}
}).elapsed;
};
time_clock(1);
int iters = clock_cost_estimation_iterations;
auto&& r = run_for_at_least<Clock>(std::chrono::duration_cast<ClockDuration<Clock>>(clock_cost_estimation_time), iters, time_clock);
std::vector<double> times;
int nsamples = static_cast<int>(std::ceil(time_limit / r.elapsed));
times.reserve(static_cast<size_t>(nsamples));
std::generate_n(std::back_inserter(times), nsamples, [time_clock, &r] {
return static_cast<double>((time_clock(r.iterations) / r.iterations).count());
});
return {
FloatDuration<Clock>(mean(times.begin(), times.end())),
classify_outliers(times.begin(), times.end()),
};
}

template <typename Clock>
Environment<FloatDuration<Clock>> measure_environment() {
#if defined(__clang__)
#    pragma clang diagnostic push
#    pragma clang diagnostic ignored "-Wexit-time-destructors"
#endif
static Catch::Detail::unique_ptr<Environment<FloatDuration<Clock>>> env;
#if defined(__clang__)
#    pragma clang diagnostic pop
#endif
if (env) {
return *env;
}

auto iters = Detail::warmup<Clock>();
auto resolution = Detail::estimate_clock_resolution<Clock>(iters);
auto cost = Detail::estimate_clock_cost<Clock>(resolution.mean);

env = Catch::Detail::make_unique<Environment<FloatDuration<Clock>>>( Environment<FloatDuration<Clock>>{resolution, cost} );
return *env;
}
} 
} 
} 

#endif 



#ifndef CATCH_ANALYSE_HPP_INCLUDED
#define CATCH_ANALYSE_HPP_INCLUDED




#ifndef CATCH_SAMPLE_ANALYSIS_HPP_INCLUDED
#define CATCH_SAMPLE_ANALYSIS_HPP_INCLUDED


#include <algorithm>
#include <vector>
#include <iterator>

namespace Catch {
namespace Benchmark {
template <typename Duration>
struct SampleAnalysis {
std::vector<Duration> samples;
Estimate<Duration> mean;
Estimate<Duration> standard_deviation;
OutlierClassification outliers;
double outlier_variance;

template <typename Duration2>
operator SampleAnalysis<Duration2>() const {
std::vector<Duration2> samples2;
samples2.reserve(samples.size());
std::transform(samples.begin(), samples.end(), std::back_inserter(samples2), [](Duration d) { return Duration2(d); });
return {
CATCH_MOVE(samples2),
mean,
standard_deviation,
outliers,
outlier_variance,
};
}
};
} 
} 

#endif 

#include <algorithm>
#include <iterator>
#include <vector>

namespace Catch {
namespace Benchmark {
namespace Detail {
template <typename Duration, typename Iterator>
SampleAnalysis<Duration> analyse(const IConfig &cfg, Environment<Duration>, Iterator first, Iterator last) {
if (!cfg.benchmarkNoAnalysis()) {
std::vector<double> samples;
samples.reserve(static_cast<size_t>(last - first));
std::transform(first, last, std::back_inserter(samples), [](Duration d) { return d.count(); });

auto analysis = Catch::Benchmark::Detail::analyse_samples(cfg.benchmarkConfidenceInterval(), cfg.benchmarkResamples(), samples.begin(), samples.end());
auto outliers = Catch::Benchmark::Detail::classify_outliers(samples.begin(), samples.end());

auto wrap_estimate = [](Estimate<double> e) {
return Estimate<Duration> {
Duration(e.point),
Duration(e.lower_bound),
Duration(e.upper_bound),
e.confidence_interval,
};
};
std::vector<Duration> samples2;
samples2.reserve(samples.size());
std::transform(samples.begin(), samples.end(), std::back_inserter(samples2), [](double d) { return Duration(d); });
return {
CATCH_MOVE(samples2),
wrap_estimate(analysis.mean),
wrap_estimate(analysis.standard_deviation),
outliers,
analysis.outlier_variance,
};
} else {
std::vector<Duration> samples;
samples.reserve(static_cast<size_t>(last - first));

Duration mean = Duration(0);
int i = 0;
for (auto it = first; it < last; ++it, ++i) {
samples.push_back(Duration(*it));
mean += Duration(*it);
}
mean /= i;

return {
CATCH_MOVE(samples),
Estimate<Duration>{mean, mean, mean, 0.0},
Estimate<Duration>{Duration(0), Duration(0), Duration(0), 0.0},
OutlierClassification{},
0.0
};
}
}
} 
} 
} 

#endif 

#include <algorithm>
#include <functional>
#include <string>
#include <vector>
#include <cmath>

namespace Catch {
namespace Benchmark {
struct Benchmark {
Benchmark(std::string&& benchmarkName)
: name(CATCH_MOVE(benchmarkName)) {}

template <class FUN>
Benchmark(std::string&& benchmarkName , FUN &&func)
: fun(CATCH_MOVE(func)), name(CATCH_MOVE(benchmarkName)) {}

template <typename Clock>
ExecutionPlan<FloatDuration<Clock>> prepare(const IConfig &cfg, Environment<FloatDuration<Clock>> env) const {
auto min_time = env.clock_resolution.mean * Detail::minimum_ticks;
auto run_time = std::max(min_time, std::chrono::duration_cast<decltype(min_time)>(cfg.benchmarkWarmupTime()));
auto&& test = Detail::run_for_at_least<Clock>(std::chrono::duration_cast<ClockDuration<Clock>>(run_time), 1, fun);
int new_iters = static_cast<int>(std::ceil(min_time * test.iterations / test.elapsed));
return { new_iters, test.elapsed / test.iterations * new_iters * cfg.benchmarkSamples(), fun, std::chrono::duration_cast<FloatDuration<Clock>>(cfg.benchmarkWarmupTime()), Detail::warmup_iterations };
}

template <typename Clock = default_clock>
void run() {
auto const* cfg = getCurrentContext().getConfig();

auto env = Detail::measure_environment<Clock>();

getResultCapture().benchmarkPreparing(name);
CATCH_TRY{
auto plan = user_code([&] {
return prepare<Clock>(*cfg, env);
});

BenchmarkInfo info {
name,
plan.estimated_duration.count(),
plan.iterations_per_sample,
cfg->benchmarkSamples(),
cfg->benchmarkResamples(),
env.clock_resolution.mean.count(),
env.clock_cost.mean.count()
};

getResultCapture().benchmarkStarting(info);

auto samples = user_code([&] {
return plan.template run<Clock>(*cfg, env);
});

auto analysis = Detail::analyse(*cfg, env, samples.begin(), samples.end());
BenchmarkStats<FloatDuration<Clock>> stats{ info, analysis.samples, analysis.mean, analysis.standard_deviation, analysis.outliers, analysis.outlier_variance };
getResultCapture().benchmarkEnded(stats);
} CATCH_CATCH_ANON (TestFailureException) {
getResultCapture().benchmarkFailed("Benchmark failed due to failed assertion"_sr);
} CATCH_CATCH_ALL{
getResultCapture().benchmarkFailed(translateActiveException());
std::rethrow_exception(std::current_exception());
}
}

template <typename Fun, std::enable_if_t<!Detail::is_related<Fun, Benchmark>::value, int> = 0>
Benchmark & operator=(Fun func) {
auto const* cfg = getCurrentContext().getConfig();
if (!cfg->skipBenchmarks()) {
fun = Detail::BenchmarkFunction(func);
run();
}
return *this;
}

explicit operator bool() {
return true;
}

private:
Detail::BenchmarkFunction fun;
std::string name;
};
}
} 

#define INTERNAL_CATCH_GET_1_ARG(arg1, arg2, ...) arg1
#define INTERNAL_CATCH_GET_2_ARG(arg1, arg2, ...) arg2

#define INTERNAL_CATCH_BENCHMARK(BenchmarkName, name, benchmarkIndex)\
if( Catch::Benchmark::Benchmark BenchmarkName{name} ) \
BenchmarkName = [&](int benchmarkIndex)

#define INTERNAL_CATCH_BENCHMARK_ADVANCED(BenchmarkName, name)\
if( Catch::Benchmark::Benchmark BenchmarkName{name} ) \
BenchmarkName = [&]

#if defined(CATCH_CONFIG_PREFIX_ALL)

#define CATCH_BENCHMARK(...) \
INTERNAL_CATCH_BENCHMARK(INTERNAL_CATCH_UNIQUE_NAME(CATCH2_INTERNAL_BENCHMARK_), INTERNAL_CATCH_GET_1_ARG(__VA_ARGS__,,), INTERNAL_CATCH_GET_2_ARG(__VA_ARGS__,,))
#define CATCH_BENCHMARK_ADVANCED(name) \
INTERNAL_CATCH_BENCHMARK_ADVANCED(INTERNAL_CATCH_UNIQUE_NAME(CATCH2_INTERNAL_BENCHMARK_), name)

#else

#define BENCHMARK(...) \
INTERNAL_CATCH_BENCHMARK(INTERNAL_CATCH_UNIQUE_NAME(CATCH2_INTERNAL_BENCHMARK_), INTERNAL_CATCH_GET_1_ARG(__VA_ARGS__,,), INTERNAL_CATCH_GET_2_ARG(__VA_ARGS__,,))
#define BENCHMARK_ADVANCED(name) \
INTERNAL_CATCH_BENCHMARK_ADVANCED(INTERNAL_CATCH_UNIQUE_NAME(CATCH2_INTERNAL_BENCHMARK_), name)

#endif

#endif 



#ifndef CATCH_CONSTRUCTOR_HPP_INCLUDED
#define CATCH_CONSTRUCTOR_HPP_INCLUDED


#include <type_traits>

namespace Catch {
namespace Benchmark {
namespace Detail {
template <typename T, bool Destruct>
struct ObjectStorage
{
ObjectStorage() = default;

ObjectStorage(const ObjectStorage& other)
{
new(&data) T(other.stored_object());
}

ObjectStorage(ObjectStorage&& other)
{
new(data) T(CATCH_MOVE(other.stored_object()));
}

~ObjectStorage() { destruct_on_exit<T>(); }

template <typename... Args>
void construct(Args&&... args)
{
new (data) T(CATCH_FORWARD(args)...);
}

template <bool AllowManualDestruction = !Destruct>
std::enable_if_t<AllowManualDestruction> destruct()
{
stored_object().~T();
}

private:
template <typename U>
void destruct_on_exit(std::enable_if_t<Destruct, U>* = nullptr) { destruct<true>(); }
template <typename U>
void destruct_on_exit(std::enable_if_t<!Destruct, U>* = nullptr) { }

#if defined( __GNUC__ ) && __GNUC__ <= 6
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif
T& stored_object() { return *reinterpret_cast<T*>( data ); }

T const& stored_object() const {
return *reinterpret_cast<T const*>( data );
}
#if defined( __GNUC__ ) && __GNUC__ <= 6
#    pragma GCC diagnostic pop
#endif

alignas( T ) unsigned char data[sizeof( T )]{};
};
} 

template <typename T>
using storage_for = Detail::ObjectStorage<T, true>;

template <typename T>
using destructable_object = Detail::ObjectStorage<T, false>;
} 
} 

#endif 

#endif 


#ifndef CATCH_APPROX_HPP_INCLUDED
#define CATCH_APPROX_HPP_INCLUDED



#ifndef CATCH_TOSTRING_HPP_INCLUDED
#define CATCH_TOSTRING_HPP_INCLUDED


#include <vector>
#include <cstddef>
#include <type_traits>
#include <string>






#ifndef CATCH_CONFIG_WCHAR_HPP_INCLUDED
#define CATCH_CONFIG_WCHAR_HPP_INCLUDED


#if defined(__DJGPP__)
#  define CATCH_INTERNAL_CONFIG_NO_WCHAR
#endif 

#if !defined( CATCH_INTERNAL_CONFIG_NO_WCHAR ) && \
!defined( CATCH_CONFIG_NO_WCHAR ) && \
!defined( CATCH_CONFIG_WCHAR )
#    define CATCH_CONFIG_WCHAR
#endif

#endif 


#ifndef CATCH_REUSABLE_STRING_STREAM_HPP_INCLUDED
#define CATCH_REUSABLE_STRING_STREAM_HPP_INCLUDED


#include <iosfwd>
#include <cstddef>
#include <ostream>
#include <string>

namespace Catch {

class ReusableStringStream : Detail::NonCopyable {
std::size_t m_index;
std::ostream* m_oss;
public:
ReusableStringStream();
~ReusableStringStream();

std::string str() const;
void str(std::string const& str);

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpragmas"
#pragma GCC diagnostic ignored "-Waddress"
#pragma GCC diagnostic ignored "-Wnonnull-compare"
#endif

template<typename T>
auto operator << ( T const& value ) -> ReusableStringStream& {
*m_oss << value;
return *this;
}

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif
auto get() -> std::ostream& { return *m_oss; }
};
}

#endif 


#ifndef CATCH_VOID_TYPE_HPP_INCLUDED
#define CATCH_VOID_TYPE_HPP_INCLUDED


namespace Catch {
namespace Detail {

template <typename...>
struct make_void { using type = void; };

template <typename... Ts>
using void_t = typename make_void<Ts...>::type;

} 
} 


#endif 


#ifndef CATCH_INTERFACES_ENUM_VALUES_REGISTRY_HPP_INCLUDED
#define CATCH_INTERFACES_ENUM_VALUES_REGISTRY_HPP_INCLUDED


#include <vector>

namespace Catch {

namespace Detail {
struct EnumInfo {
StringRef m_name;
std::vector<std::pair<int, StringRef>> m_values;

~EnumInfo();

StringRef lookup( int value ) const;
};
} 

class IMutableEnumValuesRegistry {
public:
virtual ~IMutableEnumValuesRegistry(); 

virtual Detail::EnumInfo const& registerEnum( StringRef enumName, StringRef allEnums, std::vector<int> const& values ) = 0;

template<typename E>
Detail::EnumInfo const& registerEnum( StringRef enumName, StringRef allEnums, std::initializer_list<E> values ) {
static_assert(sizeof(int) >= sizeof(E), "Cannot serialize enum to int");
std::vector<int> intValues;
intValues.reserve( values.size() );
for( auto enumValue : values )
intValues.push_back( static_cast<int>( enumValue ) );
return registerEnum( enumName, allEnums, intValues );
}
};

} 

#endif 

#ifdef CATCH_CONFIG_CPP17_STRING_VIEW
#include <string_view>
#endif

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable:4180) 
#endif

struct Catch_global_namespace_dummy{};
std::ostream& operator<<(std::ostream&, Catch_global_namespace_dummy);

namespace Catch {
using ::operator<<;

namespace Detail {

inline std::size_t catch_strnlen(const char *str, std::size_t n) {
auto ret = std::char_traits<char>::find(str, n, '\0');
if (ret != nullptr) {
return static_cast<std::size_t>(ret - str);
}
return n;
}

constexpr StringRef unprintableString = "{?}"_sr;

std::string convertIntoString( StringRef string, bool escapeInvisibles );

std::string convertIntoString( StringRef string );

std::string rawMemoryToString( const void *object, std::size_t size );

template<typename T>
std::string rawMemoryToString( const T& object ) {
return rawMemoryToString( &object, sizeof(object) );
}

template<typename T>
class IsStreamInsertable {
template<typename Stream, typename U>
static auto test(int)
-> decltype(std::declval<Stream&>() << std::declval<U>(), std::true_type());

template<typename, typename>
static auto test(...)->std::false_type;

public:
static const bool value = decltype(test<std::ostream, const T&>(0))::value;
};

template<typename E>
std::string convertUnknownEnumToString( E e );

template<typename T>
std::enable_if_t<
!std::is_enum<T>::value && !std::is_base_of<std::exception, T>::value,
std::string> convertUnstreamable( T const& ) {
return std::string(Detail::unprintableString);
}
template<typename T>
std::enable_if_t<
!std::is_enum<T>::value && std::is_base_of<std::exception, T>::value,
std::string> convertUnstreamable(T const& ex) {
return ex.what();
}


template<typename T>
std::enable_if_t<
std::is_enum<T>::value,
std::string> convertUnstreamable( T const& value ) {
return convertUnknownEnumToString( value );
}

#if defined(_MANAGED)
template<typename T>
std::string clrReferenceToString( T^ ref ) {
if (ref == nullptr)
return std::string("null");
auto bytes = System::Text::Encoding::UTF8->GetBytes(ref->ToString());
cli::pin_ptr<System::Byte> p = &bytes[0];
return std::string(reinterpret_cast<char const *>(p), bytes->Length);
}
#endif

} 


template <typename T, typename = void>
struct StringMaker {
template <typename Fake = T>
static
std::enable_if_t<::Catch::Detail::IsStreamInsertable<Fake>::value, std::string>
convert(const Fake& value) {
ReusableStringStream rss;
rss.operator<<(value);
return rss.str();
}

template <typename Fake = T>
static
std::enable_if_t<!::Catch::Detail::IsStreamInsertable<Fake>::value, std::string>
convert( const Fake& value ) {
#if !defined(CATCH_CONFIG_FALLBACK_STRINGIFIER)
return Detail::convertUnstreamable(value);
#else
return CATCH_CONFIG_FALLBACK_STRINGIFIER(value);
#endif
}
};

namespace Detail {

template <typename T>
std::string stringify(const T& e) {
return ::Catch::StringMaker<std::remove_cv_t<std::remove_reference_t<T>>>::convert(e);
}

template<typename E>
std::string convertUnknownEnumToString( E e ) {
return ::Catch::Detail::stringify(static_cast<std::underlying_type_t<E>>(e));
}

#if defined(_MANAGED)
template <typename T>
std::string stringify( T^ e ) {
return ::Catch::StringMaker<T^>::convert(e);
}
#endif

} 


template<>
struct StringMaker<std::string> {
static std::string convert(const std::string& str);
};

#ifdef CATCH_CONFIG_CPP17_STRING_VIEW
template<>
struct StringMaker<std::string_view> {
static std::string convert(std::string_view str);
};
#endif

template<>
struct StringMaker<char const *> {
static std::string convert(char const * str);
};
template<>
struct StringMaker<char *> {
static std::string convert(char * str);
};

#if defined(CATCH_CONFIG_WCHAR)
template<>
struct StringMaker<std::wstring> {
static std::string convert(const std::wstring& wstr);
};

# ifdef CATCH_CONFIG_CPP17_STRING_VIEW
template<>
struct StringMaker<std::wstring_view> {
static std::string convert(std::wstring_view str);
};
# endif

template<>
struct StringMaker<wchar_t const *> {
static std::string convert(wchar_t const * str);
};
template<>
struct StringMaker<wchar_t *> {
static std::string convert(wchar_t * str);
};
#endif 

template<size_t SZ>
struct StringMaker<char[SZ]> {
static std::string convert(char const* str) {
return Detail::convertIntoString(
StringRef( str, Detail::catch_strnlen( str, SZ ) ) );
}
};
template<size_t SZ>
struct StringMaker<signed char[SZ]> {
static std::string convert(signed char const* str) {
auto reinterpreted = reinterpret_cast<char const*>(str);
return Detail::convertIntoString(
StringRef(reinterpreted, Detail::catch_strnlen(reinterpreted, SZ)));
}
};
template<size_t SZ>
struct StringMaker<unsigned char[SZ]> {
static std::string convert(unsigned char const* str) {
auto reinterpreted = reinterpret_cast<char const*>(str);
return Detail::convertIntoString(
StringRef(reinterpreted, Detail::catch_strnlen(reinterpreted, SZ)));
}
};

#if defined(CATCH_CONFIG_CPP17_BYTE)
template<>
struct StringMaker<std::byte> {
static std::string convert(std::byte value);
};
#endif 
template<>
struct StringMaker<int> {
static std::string convert(int value);
};
template<>
struct StringMaker<long> {
static std::string convert(long value);
};
template<>
struct StringMaker<long long> {
static std::string convert(long long value);
};
template<>
struct StringMaker<unsigned int> {
static std::string convert(unsigned int value);
};
template<>
struct StringMaker<unsigned long> {
static std::string convert(unsigned long value);
};
template<>
struct StringMaker<unsigned long long> {
static std::string convert(unsigned long long value);
};

template<>
struct StringMaker<bool> {
static std::string convert(bool b) {
using namespace std::string_literals;
return b ? "true"s : "false"s;
}
};

template<>
struct StringMaker<char> {
static std::string convert(char c);
};
template<>
struct StringMaker<signed char> {
static std::string convert(signed char c);
};
template<>
struct StringMaker<unsigned char> {
static std::string convert(unsigned char c);
};

template<>
struct StringMaker<std::nullptr_t> {
static std::string convert(std::nullptr_t) {
using namespace std::string_literals;
return "nullptr"s;
}
};

template<>
struct StringMaker<float> {
static std::string convert(float value);
CATCH_EXPORT static int precision;
};

template<>
struct StringMaker<double> {
static std::string convert(double value);
CATCH_EXPORT static int precision;
};

template <typename T>
struct StringMaker<T*> {
template <typename U>
static std::string convert(U* p) {
if (p) {
return ::Catch::Detail::rawMemoryToString(p);
} else {
return "nullptr";
}
}
};

template <typename R, typename C>
struct StringMaker<R C::*> {
static std::string convert(R C::* p) {
if (p) {
return ::Catch::Detail::rawMemoryToString(p);
} else {
return "nullptr";
}
}
};

#if defined(_MANAGED)
template <typename T>
struct StringMaker<T^> {
static std::string convert( T^ ref ) {
return ::Catch::Detail::clrReferenceToString(ref);
}
};
#endif

namespace Detail {
template<typename InputIterator, typename Sentinel = InputIterator>
std::string rangeToString(InputIterator first, Sentinel last) {
ReusableStringStream rss;
rss << "{ ";
if (first != last) {
rss << ::Catch::Detail::stringify(*first);
for (++first; first != last; ++first)
rss << ", " << ::Catch::Detail::stringify(*first);
}
rss << " }";
return rss.str();
}
}

} 


#if defined(CATCH_CONFIG_ENABLE_ALL_STRINGMAKERS)
#  define CATCH_CONFIG_ENABLE_PAIR_STRINGMAKER
#  define CATCH_CONFIG_ENABLE_TUPLE_STRINGMAKER
#  define CATCH_CONFIG_ENABLE_VARIANT_STRINGMAKER
#  define CATCH_CONFIG_ENABLE_OPTIONAL_STRINGMAKER
#endif

#if defined(CATCH_CONFIG_ENABLE_PAIR_STRINGMAKER)
#include <utility>
namespace Catch {
template<typename T1, typename T2>
struct StringMaker<std::pair<T1, T2> > {
static std::string convert(const std::pair<T1, T2>& pair) {
ReusableStringStream rss;
rss << "{ "
<< ::Catch::Detail::stringify(pair.first)
<< ", "
<< ::Catch::Detail::stringify(pair.second)
<< " }";
return rss.str();
}
};
}
#endif 

#if defined(CATCH_CONFIG_ENABLE_OPTIONAL_STRINGMAKER) && defined(CATCH_CONFIG_CPP17_OPTIONAL)
#include <optional>
namespace Catch {
template<typename T>
struct StringMaker<std::optional<T> > {
static std::string convert(const std::optional<T>& optional) {
if (optional.has_value()) {
return ::Catch::Detail::stringify(*optional);
} else {
return "{ }";
}
}
};
}
#endif 

#if defined(CATCH_CONFIG_ENABLE_TUPLE_STRINGMAKER)
#include <tuple>
namespace Catch {
namespace Detail {
template<
typename Tuple,
std::size_t N = 0,
bool = (N < std::tuple_size<Tuple>::value)
>
struct TupleElementPrinter {
static void print(const Tuple& tuple, std::ostream& os) {
os << (N ? ", " : " ")
<< ::Catch::Detail::stringify(std::get<N>(tuple));
TupleElementPrinter<Tuple, N + 1>::print(tuple, os);
}
};

template<
typename Tuple,
std::size_t N
>
struct TupleElementPrinter<Tuple, N, false> {
static void print(const Tuple&, std::ostream&) {}
};

}


template<typename ...Types>
struct StringMaker<std::tuple<Types...>> {
static std::string convert(const std::tuple<Types...>& tuple) {
ReusableStringStream rss;
rss << '{';
Detail::TupleElementPrinter<std::tuple<Types...>>::print(tuple, rss.get());
rss << " }";
return rss.str();
}
};
}
#endif 

#if defined(CATCH_CONFIG_ENABLE_VARIANT_STRINGMAKER) && defined(CATCH_CONFIG_CPP17_VARIANT)
#include <variant>
namespace Catch {
template<>
struct StringMaker<std::monostate> {
static std::string convert(const std::monostate&) {
return "{ }";
}
};

template<typename... Elements>
struct StringMaker<std::variant<Elements...>> {
static std::string convert(const std::variant<Elements...>& variant) {
if (variant.valueless_by_exception()) {
return "{valueless variant}";
} else {
return std::visit(
[](const auto& value) {
return ::Catch::Detail::stringify(value);
},
variant
);
}
}
};
}
#endif 

namespace Catch {
using std::begin;
using std::end;

namespace Detail {
template <typename T, typename = void>
struct is_range_impl : std::false_type {};

template <typename T>
struct is_range_impl<T, void_t<decltype(begin(std::declval<T>()))>> : std::true_type {};
} 

template <typename T>
struct is_range : Detail::is_range_impl<T> {};

#if defined(_MANAGED) 
template <typename T>
struct is_range<T^> {
static const bool value = false;
};
#endif

template<typename Range>
std::string rangeToString( Range const& range ) {
return ::Catch::Detail::rangeToString( begin( range ), end( range ) );
}

template<typename Allocator>
std::string rangeToString( std::vector<bool, Allocator> const& v ) {
ReusableStringStream rss;
rss << "{ ";
bool first = true;
for( bool b : v ) {
if( first )
first = false;
else
rss << ", ";
rss << ::Catch::Detail::stringify( b );
}
rss << " }";
return rss.str();
}

template<typename R>
struct StringMaker<R, std::enable_if_t<is_range<R>::value && !::Catch::Detail::IsStreamInsertable<R>::value>> {
static std::string convert( R const& range ) {
return rangeToString( range );
}
};

template <typename T, size_t SZ>
struct StringMaker<T[SZ]> {
static std::string convert(T const(&arr)[SZ]) {
return rangeToString(arr);
}
};


} 

#include <ctime>
#include <ratio>
#include <chrono>


namespace Catch {

template <class Ratio>
struct ratio_string {
static std::string symbol() {
Catch::ReusableStringStream rss;
rss << '[' << Ratio::num << '/'
<< Ratio::den << ']';
return rss.str();
}
};

template <>
struct ratio_string<std::atto> {
static char symbol() { return 'a'; }
};
template <>
struct ratio_string<std::femto> {
static char symbol() { return 'f'; }
};
template <>
struct ratio_string<std::pico> {
static char symbol() { return 'p'; }
};
template <>
struct ratio_string<std::nano> {
static char symbol() { return 'n'; }
};
template <>
struct ratio_string<std::micro> {
static char symbol() { return 'u'; }
};
template <>
struct ratio_string<std::milli> {
static char symbol() { return 'm'; }
};

template<typename Value, typename Ratio>
struct StringMaker<std::chrono::duration<Value, Ratio>> {
static std::string convert(std::chrono::duration<Value, Ratio> const& duration) {
ReusableStringStream rss;
rss << duration.count() << ' ' << ratio_string<Ratio>::symbol() << 's';
return rss.str();
}
};
template<typename Value>
struct StringMaker<std::chrono::duration<Value, std::ratio<1>>> {
static std::string convert(std::chrono::duration<Value, std::ratio<1>> const& duration) {
ReusableStringStream rss;
rss << duration.count() << " s";
return rss.str();
}
};
template<typename Value>
struct StringMaker<std::chrono::duration<Value, std::ratio<60>>> {
static std::string convert(std::chrono::duration<Value, std::ratio<60>> const& duration) {
ReusableStringStream rss;
rss << duration.count() << " m";
return rss.str();
}
};
template<typename Value>
struct StringMaker<std::chrono::duration<Value, std::ratio<3600>>> {
static std::string convert(std::chrono::duration<Value, std::ratio<3600>> const& duration) {
ReusableStringStream rss;
rss << duration.count() << " h";
return rss.str();
}
};

template<typename Clock, typename Duration>
struct StringMaker<std::chrono::time_point<Clock, Duration>> {
static std::string convert(std::chrono::time_point<Clock, Duration> const& time_point) {
return ::Catch::Detail::stringify(time_point.time_since_epoch()) + " since epoch";
}
};
template<typename Duration>
struct StringMaker<std::chrono::time_point<std::chrono::system_clock, Duration>> {
static std::string convert(std::chrono::time_point<std::chrono::system_clock, Duration> const& time_point) {
auto converted = std::chrono::system_clock::to_time_t(time_point);

#ifdef _MSC_VER
std::tm timeInfo = {};
gmtime_s(&timeInfo, &converted);
#else
std::tm* timeInfo = std::gmtime(&converted);
#endif

auto const timeStampSize = sizeof("2017-01-16T17:06:45Z");
char timeStamp[timeStampSize];
const char * const fmt = "%Y-%m-%dT%H:%M:%SZ";

#ifdef _MSC_VER
std::strftime(timeStamp, timeStampSize, fmt, &timeInfo);
#else
std::strftime(timeStamp, timeStampSize, fmt, timeInfo);
#endif
return std::string(timeStamp, timeStampSize - 1);
}
};
}


#define INTERNAL_CATCH_REGISTER_ENUM( enumName, ... ) \
namespace Catch { \
template<> struct StringMaker<enumName> { \
static std::string convert( enumName value ) { \
static const auto& enumInfo = ::Catch::getMutableRegistryHub().getMutableEnumValuesRegistry().registerEnum( #enumName, #__VA_ARGS__, { __VA_ARGS__ } ); \
return static_cast<std::string>(enumInfo.lookup( static_cast<int>( value ) )); \
} \
}; \
}

#define CATCH_REGISTER_ENUM( enumName, ... ) INTERNAL_CATCH_REGISTER_ENUM( enumName, __VA_ARGS__ )

#ifdef _MSC_VER
#pragma warning(pop)
#endif

#endif 

#include <type_traits>

namespace Catch {

class Approx {
private:
bool equalityComparisonImpl(double other) const;
void setMargin(double margin);
void setEpsilon(double epsilon);

public:
explicit Approx ( double value );

static Approx custom();

Approx operator-() const;

template <typename T, typename = std::enable_if_t<std::is_constructible<double, T>::value>>
Approx operator()( T const& value ) const {
Approx approx( static_cast<double>(value) );
approx.m_epsilon = m_epsilon;
approx.m_margin = m_margin;
approx.m_scale = m_scale;
return approx;
}

template <typename T, typename = std::enable_if_t<std::is_constructible<double, T>::value>>
explicit Approx( T const& value ): Approx(static_cast<double>(value))
{}


template <typename T, typename = std::enable_if_t<std::is_constructible<double, T>::value>>
friend bool operator == ( const T& lhs, Approx const& rhs ) {
auto lhs_v = static_cast<double>(lhs);
return rhs.equalityComparisonImpl(lhs_v);
}

template <typename T, typename = std::enable_if_t<std::is_constructible<double, T>::value>>
friend bool operator == ( Approx const& lhs, const T& rhs ) {
return operator==( rhs, lhs );
}

template <typename T, typename = std::enable_if_t<std::is_constructible<double, T>::value>>
friend bool operator != ( T const& lhs, Approx const& rhs ) {
return !operator==( lhs, rhs );
}

template <typename T, typename = std::enable_if_t<std::is_constructible<double, T>::value>>
friend bool operator != ( Approx const& lhs, T const& rhs ) {
return !operator==( rhs, lhs );
}

template <typename T, typename = std::enable_if_t<std::is_constructible<double, T>::value>>
friend bool operator <= ( T const& lhs, Approx const& rhs ) {
return static_cast<double>(lhs) < rhs.m_value || lhs == rhs;
}

template <typename T, typename = std::enable_if_t<std::is_constructible<double, T>::value>>
friend bool operator <= ( Approx const& lhs, T const& rhs ) {
return lhs.m_value < static_cast<double>(rhs) || lhs == rhs;
}

template <typename T, typename = std::enable_if_t<std::is_constructible<double, T>::value>>
friend bool operator >= ( T const& lhs, Approx const& rhs ) {
return static_cast<double>(lhs) > rhs.m_value || lhs == rhs;
}

template <typename T, typename = std::enable_if_t<std::is_constructible<double, T>::value>>
friend bool operator >= ( Approx const& lhs, T const& rhs ) {
return lhs.m_value > static_cast<double>(rhs) || lhs == rhs;
}

template <typename T, typename = std::enable_if_t<std::is_constructible<double, T>::value>>
Approx& epsilon( T const& newEpsilon ) {
const auto epsilonAsDouble = static_cast<double>(newEpsilon);
setEpsilon(epsilonAsDouble);
return *this;
}

template <typename T, typename = std::enable_if_t<std::is_constructible<double, T>::value>>
Approx& margin( T const& newMargin ) {
const auto marginAsDouble = static_cast<double>(newMargin);
setMargin(marginAsDouble);
return *this;
}

template <typename T, typename = std::enable_if_t<std::is_constructible<double, T>::value>>
Approx& scale( T const& newScale ) {
m_scale = static_cast<double>(newScale);
return *this;
}

std::string toString() const;

private:
double m_epsilon;
double m_margin;
double m_scale;
double m_value;
};

namespace literals {
Approx operator ""_a(long double val);
Approx operator ""_a(unsigned long long val);
} 

template<>
struct StringMaker<Catch::Approx> {
static std::string convert(Catch::Approx const& value);
};

} 

#endif 


#ifndef CATCH_CONFIG_HPP_INCLUDED
#define CATCH_CONFIG_HPP_INCLUDED



#ifndef CATCH_TEST_SPEC_HPP_INCLUDED
#define CATCH_TEST_SPEC_HPP_INCLUDED

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wpadded"
#endif



#ifndef CATCH_WILDCARD_PATTERN_HPP_INCLUDED
#define CATCH_WILDCARD_PATTERN_HPP_INCLUDED



#ifndef CATCH_CASE_SENSITIVE_HPP_INCLUDED
#define CATCH_CASE_SENSITIVE_HPP_INCLUDED

namespace Catch {

enum class CaseSensitive { Yes, No };

} 

#endif 

#include <string>

namespace Catch
{
class WildcardPattern {
enum WildcardPosition {
NoWildcard = 0,
WildcardAtStart = 1,
WildcardAtEnd = 2,
WildcardAtBothEnds = WildcardAtStart | WildcardAtEnd
};

public:

WildcardPattern( std::string const& pattern, CaseSensitive caseSensitivity );
bool matches( std::string const& str ) const;

private:
std::string normaliseString( std::string const& str ) const;
CaseSensitive m_caseSensitivity;
WildcardPosition m_wildcard = NoWildcard;
std::string m_pattern;
};
}

#endif 

#include <iosfwd>
#include <string>
#include <vector>

namespace Catch {

class IConfig;
struct TestCaseInfo;
class TestCaseHandle;

class TestSpec {

class Pattern {
public:
explicit Pattern( std::string const& name );
virtual ~Pattern();
virtual bool matches( TestCaseInfo const& testCase ) const = 0;
std::string const& name() const;
private:
virtual void serializeTo( std::ostream& out ) const = 0;
friend std::ostream& operator<<(std::ostream& out,
Pattern const& pattern) {
pattern.serializeTo( out );
return out;
}

std::string const m_name;
};

class NamePattern : public Pattern {
public:
explicit NamePattern( std::string const& name, std::string const& filterString );
bool matches( TestCaseInfo const& testCase ) const override;
private:
void serializeTo( std::ostream& out ) const override;

WildcardPattern m_wildcardPattern;
};

class TagPattern : public Pattern {
public:
explicit TagPattern( std::string const& tag, std::string const& filterString );
bool matches( TestCaseInfo const& testCase ) const override;
private:
void serializeTo( std::ostream& out ) const override;

std::string m_tag;
};

struct Filter {
std::vector<Detail::unique_ptr<Pattern>> m_required;
std::vector<Detail::unique_ptr<Pattern>> m_forbidden;

void serializeTo( std::ostream& out ) const;
friend std::ostream& operator<<(std::ostream& out, Filter const& f) {
f.serializeTo( out );
return out;
}

bool matches( TestCaseInfo const& testCase ) const;
};

static std::string extractFilterName( Filter const& filter );

public:
struct FilterMatch {
std::string name;
std::vector<TestCaseHandle const*> tests;
};
using Matches = std::vector<FilterMatch>;
using vectorStrings = std::vector<std::string>;

bool hasFilters() const;
bool matches( TestCaseInfo const& testCase ) const;
Matches matchesByFilter( std::vector<TestCaseHandle> const& testCases, IConfig const& config ) const;
const vectorStrings & getInvalidSpecs() const;

private:
std::vector<Filter> m_filters;
std::vector<std::string> m_invalidSpecs;

friend class TestSpecParser;
void serializeTo( std::ostream& out ) const;
friend std::ostream& operator<<(std::ostream& out,
TestSpec const& spec) {
spec.serializeTo( out );
return out;
}
};
}

#ifdef __clang__
#pragma clang diagnostic pop
#endif

#endif 


#ifndef CATCH_OPTIONAL_HPP_INCLUDED
#define CATCH_OPTIONAL_HPP_INCLUDED

#include <cassert>

namespace Catch {

template<typename T>
class Optional {
public:
Optional() : nullableValue( nullptr ) {}
Optional( T const& _value )
: nullableValue( new( storage ) T( _value ) )
{}
Optional( Optional const& _other )
: nullableValue( _other ? new( storage ) T( *_other ) : nullptr )
{}

~Optional() {
reset();
}

Optional& operator= ( Optional const& _other ) {
if( &_other != this ) {
reset();
if( _other )
nullableValue = new( storage ) T( *_other );
}
return *this;
}
Optional& operator = ( T const& _value ) {
reset();
nullableValue = new( storage ) T( _value );
return *this;
}

void reset() {
if( nullableValue )
nullableValue->~T();
nullableValue = nullptr;
}

T& operator*() {
assert(nullableValue);
return *nullableValue;
}
T const& operator*() const {
assert(nullableValue);
return *nullableValue;
}
T* operator->() {
assert(nullableValue);
return nullableValue;
}
const T* operator->() const {
assert(nullableValue);
return nullableValue;
}

T valueOr( T const& defaultValue ) const {
return nullableValue ? *nullableValue : defaultValue;
}

bool some() const { return nullableValue != nullptr; }
bool none() const { return nullableValue == nullptr; }

bool operator !() const { return nullableValue == nullptr; }
explicit operator bool() const {
return some();
}

friend bool operator==(Optional const& a, Optional const& b) {
if (a.none() && b.none()) {
return true;
} else if (a.some() && b.some()) {
return *a == *b;
} else {
return false;
}
}
friend bool operator!=(Optional const& a, Optional const& b) {
return !( a == b );
}

private:
T *nullableValue;
alignas(alignof(T)) char storage[sizeof(T)];
};

} 

#endif 


#ifndef CATCH_RANDOM_SEED_GENERATION_HPP_INCLUDED
#define CATCH_RANDOM_SEED_GENERATION_HPP_INCLUDED

#include <cstdint>

namespace Catch {

enum class GenerateFrom {
Time,
RandomDevice,
Default
};

std::uint32_t generateRandomSeed(GenerateFrom from);

} 

#endif 


#ifndef CATCH_REPORTER_SPEC_PARSER_HPP_INCLUDED
#define CATCH_REPORTER_SPEC_PARSER_HPP_INCLUDED



#ifndef CATCH_CONSOLE_COLOUR_HPP_INCLUDED
#define CATCH_CONSOLE_COLOUR_HPP_INCLUDED


#include <iosfwd>
#include <cstdint>

namespace Catch {

enum class ColourMode : std::uint8_t;
class IStream;

struct Colour {
enum Code {
None = 0,

White,
Red,
Green,
Blue,
Cyan,
Yellow,
Grey,

Bright = 0x10,

BrightRed = Bright | Red,
BrightGreen = Bright | Green,
LightGrey = Bright | Grey,
BrightWhite = Bright | White,
BrightYellow = Bright | Yellow,

FileName = LightGrey,
Warning = BrightYellow,
ResultError = BrightRed,
ResultSuccess = BrightGreen,
ResultExpectedFailure = Warning,

Error = BrightRed,
Success = Green,

OriginalExpression = Cyan,
ReconstructedExpression = BrightYellow,

SecondaryText = LightGrey,
Headers = White
};
};

class ColourImpl {
protected:
IStream* m_stream;
public:
ColourImpl( IStream* stream ): m_stream( stream ) {}

class ColourGuard {
ColourImpl const* m_colourImpl;
Colour::Code m_code;
bool m_engaged = false;

public:
ColourGuard( Colour::Code code,
ColourImpl const* colour );

ColourGuard( ColourGuard const& rhs ) = delete;
ColourGuard& operator=( ColourGuard const& rhs ) = delete;

ColourGuard( ColourGuard&& rhs ) noexcept;
ColourGuard& operator=( ColourGuard&& rhs ) noexcept;

~ColourGuard();


ColourGuard& engage( std::ostream& stream ) &;

ColourGuard&& engage( std::ostream& stream ) &&;

private:
friend std::ostream& operator<<( std::ostream& lhs,
ColourGuard& guard ) {
guard.engageImpl( lhs );
return lhs;
}
friend std::ostream& operator<<( std::ostream& lhs,
ColourGuard&& guard) {
guard.engageImpl( lhs );
return lhs;
}

void engageImpl( std::ostream& stream );

};

virtual ~ColourImpl(); 

ColourGuard guardColour( Colour::Code colourCode );

private:
virtual void use( Colour::Code colourCode ) const = 0;
};

Detail::unique_ptr<ColourImpl> makeColourImpl( ColourMode colourSelection,
IStream* stream );

bool isColourImplAvailable( ColourMode colourSelection );

} 

#endif 

#include <map>
#include <string>
#include <vector>

namespace Catch {

enum class ColourMode : std::uint8_t;

namespace Detail {
std::vector<std::string> splitReporterSpec( StringRef reporterSpec );

Optional<ColourMode> stringToColourMode( StringRef colourMode );
}


class ReporterSpec {
std::string m_name;
Optional<std::string> m_outputFileName;
Optional<ColourMode> m_colourMode;
std::map<std::string, std::string> m_customOptions;

friend bool operator==( ReporterSpec const& lhs,
ReporterSpec const& rhs );
friend bool operator!=( ReporterSpec const& lhs,
ReporterSpec const& rhs ) {
return !( lhs == rhs );
}

public:
ReporterSpec(
std::string name,
Optional<std::string> outputFileName,
Optional<ColourMode> colourMode,
std::map<std::string, std::string> customOptions );

std::string const& name() const { return m_name; }

Optional<std::string> const& outputFile() const {
return m_outputFileName;
}

Optional<ColourMode> const& colourMode() const { return m_colourMode; }

std::map<std::string, std::string> const& customOptions() const {
return m_customOptions;
}
};


Optional<ReporterSpec> parseReporterSpec( StringRef reporterSpec );

}

#endif 

#include <chrono>
#include <map>
#include <string>
#include <vector>

namespace Catch {

class IStream;


struct ProcessedReporterSpec {
std::string name;
std::string outputFilename;
ColourMode colourMode;
std::map<std::string, std::string> customOptions;
friend bool operator==( ProcessedReporterSpec const& lhs,
ProcessedReporterSpec const& rhs );
friend bool operator!=( ProcessedReporterSpec const& lhs,
ProcessedReporterSpec const& rhs ) {
return !( lhs == rhs );
}
};

struct ConfigData {

bool listTests = false;
bool listTags = false;
bool listReporters = false;
bool listListeners = false;

bool showSuccessfulTests = false;
bool shouldDebugBreak = false;
bool noThrow = false;
bool showHelp = false;
bool showInvisibles = false;
bool filenamesAsTags = false;
bool libIdentify = false;
bool allowZeroTests = false;

int abortAfter = -1;
uint32_t rngSeed = generateRandomSeed(GenerateFrom::Default);

unsigned int shardCount = 1;
unsigned int shardIndex = 0;

bool skipBenchmarks = false;
bool benchmarkNoAnalysis = false;
unsigned int benchmarkSamples = 100;
double benchmarkConfidenceInterval = 0.95;
unsigned int benchmarkResamples = 100000;
std::chrono::milliseconds::rep benchmarkWarmupTime = 100;

Verbosity verbosity = Verbosity::Normal;
WarnAbout::What warnings = WarnAbout::Nothing;
ShowDurations showDurations = ShowDurations::DefaultForReporter;
double minDuration = -1;
TestRunOrder runOrder = TestRunOrder::Declared;
ColourMode defaultColourMode = ColourMode::PlatformDefault;
WaitForKeypress::When waitForKeypress = WaitForKeypress::Never;

std::string defaultOutputFilename;
std::string name;
std::string processName;
std::vector<ReporterSpec> reporterSpecifications;

std::vector<std::string> testsOrTags;
std::vector<std::string> sectionsToRun;
};


class Config : public IConfig {
public:

Config() = default;
Config( ConfigData const& data );
~Config() override; 

bool listTests() const;
bool listTags() const;
bool listReporters() const;
bool listListeners() const;

std::vector<ReporterSpec> const& getReporterSpecs() const;
std::vector<ProcessedReporterSpec> const&
getProcessedReporterSpecs() const;

std::vector<std::string> const& getTestsOrTags() const override;
std::vector<std::string> const& getSectionsToRun() const override;

TestSpec const& testSpec() const override;
bool hasTestFilters() const override;

bool showHelp() const;

bool allowThrows() const override;
StringRef name() const override;
bool includeSuccessfulResults() const override;
bool warnAboutMissingAssertions() const override;
bool warnAboutUnmatchedTestSpecs() const override;
bool zeroTestsCountAsSuccess() const override;
ShowDurations showDurations() const override;
double minDuration() const override;
TestRunOrder runOrder() const override;
uint32_t rngSeed() const override;
unsigned int shardCount() const override;
unsigned int shardIndex() const override;
ColourMode defaultColourMode() const override;
bool shouldDebugBreak() const override;
int abortAfter() const override;
bool showInvisibles() const override;
Verbosity verbosity() const override;
bool skipBenchmarks() const override;
bool benchmarkNoAnalysis() const override;
unsigned int benchmarkSamples() const override;
double benchmarkConfidenceInterval() const override;
unsigned int benchmarkResamples() const override;
std::chrono::milliseconds benchmarkWarmupTime() const override;

private:
void readBazelEnvVars();

ConfigData m_data;
std::vector<ProcessedReporterSpec> m_processedReporterSpecs;
TestSpec m_testSpec;
bool m_hasTestFilters = false;
};
} 

#endif 


#ifndef CATCH_GET_RANDOM_SEED_HPP_INCLUDED
#define CATCH_GET_RANDOM_SEED_HPP_INCLUDED

#include <cstdint>

namespace Catch {
std::uint32_t getSeed();
}

#endif 


#ifndef CATCH_MESSAGE_HPP_INCLUDED
#define CATCH_MESSAGE_HPP_INCLUDED



#ifndef CATCH_STREAM_END_STOP_HPP_INCLUDED
#define CATCH_STREAM_END_STOP_HPP_INCLUDED


namespace Catch {

struct StreamEndStop {
StringRef operator+() const { return StringRef(); }

template <typename T>
friend T const& operator+( T const& value, StreamEndStop ) {
return value;
}
};

} 

#endif 

#include <string>
#include <vector>

namespace Catch {

struct SourceLineInfo;

struct MessageStream {

template<typename T>
MessageStream& operator << ( T const& value ) {
m_stream << value;
return *this;
}

ReusableStringStream m_stream;
};

struct MessageBuilder : MessageStream {
MessageBuilder( StringRef macroName,
SourceLineInfo const& lineInfo,
ResultWas::OfType type ):
m_info(macroName, lineInfo, type) {}


template<typename T>
MessageBuilder& operator << ( T const& value ) {
m_stream << value;
return *this;
}

MessageInfo m_info;
};

class ScopedMessage {
public:
explicit ScopedMessage( MessageBuilder const& builder );
ScopedMessage( ScopedMessage& duplicate ) = delete;
ScopedMessage( ScopedMessage&& old ) noexcept;
~ScopedMessage();

MessageInfo m_info;
bool m_moved = false;
};

class Capturer {
std::vector<MessageInfo> m_messages;
IResultCapture& m_resultCapture = getResultCapture();
size_t m_captured = 0;
public:
Capturer( StringRef macroName, SourceLineInfo const& lineInfo, ResultWas::OfType resultType, StringRef names );

Capturer(Capturer const&) = delete;
Capturer& operator=(Capturer const&) = delete;

~Capturer();

void captureValue( size_t index, std::string const& value );

template<typename T>
void captureValues( size_t index, T const& value ) {
captureValue( index, Catch::Detail::stringify( value ) );
}

template<typename T, typename... Ts>
void captureValues( size_t index, T const& value, Ts const&... values ) {
captureValue( index, Catch::Detail::stringify(value) );
captureValues( index+1, values... );
}
};

} 

#define INTERNAL_CATCH_MSG( macroName, messageType, resultDisposition, ... ) \
do { \
Catch::AssertionHandler catchAssertionHandler( macroName##_catch_sr, CATCH_INTERNAL_LINEINFO, Catch::StringRef(), resultDisposition ); \
catchAssertionHandler.handleMessage( messageType, ( Catch::MessageStream() << __VA_ARGS__ + ::Catch::StreamEndStop() ).m_stream.str() ); \
INTERNAL_CATCH_REACT( catchAssertionHandler ) \
} while( false )

#define INTERNAL_CATCH_CAPTURE( varName, macroName, ... ) \
Catch::Capturer varName( macroName, CATCH_INTERNAL_LINEINFO, Catch::ResultWas::Info, #__VA_ARGS__ ); \
varName.captureValues( 0, __VA_ARGS__ )

#define INTERNAL_CATCH_INFO( macroName, log ) \
const Catch::ScopedMessage INTERNAL_CATCH_UNIQUE_NAME( scopedMessage )( Catch::MessageBuilder( macroName##_catch_sr, CATCH_INTERNAL_LINEINFO, Catch::ResultWas::Info ) << log )

#define INTERNAL_CATCH_UNSCOPED_INFO( macroName, log ) \
Catch::getResultCapture().emplaceUnscopedMessage( Catch::MessageBuilder( macroName##_catch_sr, CATCH_INTERNAL_LINEINFO, Catch::ResultWas::Info ) << log )


#if defined(CATCH_CONFIG_PREFIX_ALL) && !defined(CATCH_CONFIG_DISABLE)

#define CATCH_INFO( msg ) INTERNAL_CATCH_INFO( "CATCH_INFO", msg )
#define CATCH_UNSCOPED_INFO( msg ) INTERNAL_CATCH_UNSCOPED_INFO( "CATCH_UNSCOPED_INFO", msg )
#define CATCH_WARN( msg ) INTERNAL_CATCH_MSG( "CATCH_WARN", Catch::ResultWas::Warning, Catch::ResultDisposition::ContinueOnFailure, msg )
#define CATCH_CAPTURE( ... ) INTERNAL_CATCH_CAPTURE( INTERNAL_CATCH_UNIQUE_NAME(capturer), "CATCH_CAPTURE", __VA_ARGS__ )

#elif defined(CATCH_CONFIG_PREFIX_ALL) && defined(CATCH_CONFIG_DISABLE)

#define CATCH_INFO( msg )          (void)(0)
#define CATCH_UNSCOPED_INFO( msg ) (void)(0)
#define CATCH_WARN( msg )          (void)(0)
#define CATCH_CAPTURE( ... )       (void)(0)

#elif !defined(CATCH_CONFIG_PREFIX_ALL) && !defined(CATCH_CONFIG_DISABLE)

#define INFO( msg ) INTERNAL_CATCH_INFO( "INFO", msg )
#define UNSCOPED_INFO( msg ) INTERNAL_CATCH_UNSCOPED_INFO( "UNSCOPED_INFO", msg )
#define WARN( msg ) INTERNAL_CATCH_MSG( "WARN", Catch::ResultWas::Warning, Catch::ResultDisposition::ContinueOnFailure, msg )
#define CAPTURE( ... ) INTERNAL_CATCH_CAPTURE( INTERNAL_CATCH_UNIQUE_NAME(capturer), "CAPTURE", __VA_ARGS__ )

#elif !defined(CATCH_CONFIG_PREFIX_ALL) && defined(CATCH_CONFIG_DISABLE)

#define INFO( msg )          (void)(0)
#define UNSCOPED_INFO( msg ) (void)(0)
#define WARN( msg )          (void)(0)
#define CAPTURE( ... )       (void)(0)

#endif 




#endif 


#ifndef CATCH_SESSION_HPP_INCLUDED
#define CATCH_SESSION_HPP_INCLUDED



#ifndef CATCH_COMMANDLINE_HPP_INCLUDED
#define CATCH_COMMANDLINE_HPP_INCLUDED



#ifndef CATCH_CLARA_HPP_INCLUDED
#define CATCH_CLARA_HPP_INCLUDED

#if defined( __clang__ )
#    pragma clang diagnostic push
#    pragma clang diagnostic ignored "-Wweak-vtables"
#    pragma clang diagnostic ignored "-Wshadow"
#    pragma clang diagnostic ignored "-Wdeprecated"
#endif

#if defined( __GNUC__ )
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Wsign-conversion"
#endif

#ifndef CLARA_CONFIG_OPTIONAL_TYPE
#    ifdef __has_include
#        if __has_include( <optional>) && __cplusplus >= 201703L
#            include <optional>
#            define CLARA_CONFIG_OPTIONAL_TYPE std::optional
#        endif
#    endif
#endif


#include <cassert>
#include <memory>
#include <ostream>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

namespace Catch {
namespace Clara {

class Args;
class Parser;

enum class ParseResultType {
Matched,
NoMatch,
ShortCircuitAll,
ShortCircuitSame
};

struct accept_many_t {};
constexpr accept_many_t accept_many {};

namespace Detail {
struct fake_arg {
template <typename T>
operator T();
};

template <typename F, typename = void>
struct is_unary_function : std::false_type {};

template <typename F>
struct is_unary_function<
F,
Catch::Detail::void_t<decltype(
std::declval<F>()( fake_arg() ) )
>
> : std::true_type {};

template <typename L>
struct UnaryLambdaTraits
: UnaryLambdaTraits<decltype( &L::operator() )> {};

template <typename ClassT, typename ReturnT, typename... Args>
struct UnaryLambdaTraits<ReturnT ( ClassT::* )( Args... ) const> {
static const bool isValid = false;
};

template <typename ClassT, typename ReturnT, typename ArgT>
struct UnaryLambdaTraits<ReturnT ( ClassT::* )( ArgT ) const> {
static const bool isValid = true;
using ArgType = std::remove_const_t<std::remove_reference_t<ArgT>>;
using ReturnType = ReturnT;
};

class TokenStream;

enum class TokenType { Option, Argument };
struct Token {
TokenType type;
std::string token;
};

class TokenStream {
using Iterator = std::vector<std::string>::const_iterator;
Iterator it;
Iterator itEnd;
std::vector<Token> m_tokenBuffer;

void loadBuffer();

public:
explicit TokenStream( Args const& args );
TokenStream( Iterator it, Iterator itEnd );

explicit operator bool() const {
return !m_tokenBuffer.empty() || it != itEnd;
}

size_t count() const {
return m_tokenBuffer.size() + ( itEnd - it );
}

Token operator*() const {
assert( !m_tokenBuffer.empty() );
return m_tokenBuffer.front();
}

Token const* operator->() const {
assert( !m_tokenBuffer.empty() );
return &m_tokenBuffer.front();
}

TokenStream& operator++();
};

enum class ResultType {
Ok,          
LogicError,  
RuntimeError 
};

class ResultBase {
protected:
ResultBase( ResultType type ): m_type( type ) {}
virtual ~ResultBase(); 


ResultBase(ResultBase const&) = default;
ResultBase& operator=(ResultBase const&) = default;
ResultBase(ResultBase&&) = default;
ResultBase& operator=(ResultBase&&) = default;

virtual void enforceOk() const = 0;

ResultType m_type;
};

template <typename T> class ResultValueBase : public ResultBase {
public:
auto value() const -> T const& {
enforceOk();
return m_value;
}

protected:
ResultValueBase( ResultType type ): ResultBase( type ) {}

ResultValueBase( ResultValueBase const& other ):
ResultBase( other ) {
if ( m_type == ResultType::Ok )
new ( &m_value ) T( other.m_value );
}

ResultValueBase( ResultType, T const& value ): ResultBase( ResultType::Ok ) {
new ( &m_value ) T( value );
}

auto operator=( ResultValueBase const& other )
-> ResultValueBase& {
if ( m_type == ResultType::Ok )
m_value.~T();
ResultBase::operator=( other );
if ( m_type == ResultType::Ok )
new ( &m_value ) T( other.m_value );
return *this;
}

~ResultValueBase() override {
if ( m_type == ResultType::Ok )
m_value.~T();
}

union {
T m_value;
};
};

template <> class ResultValueBase<void> : public ResultBase {
protected:
using ResultBase::ResultBase;
};

template <typename T = void>
class BasicResult : public ResultValueBase<T> {
public:
template <typename U>
explicit BasicResult( BasicResult<U> const& other ):
ResultValueBase<T>( other.type() ),
m_errorMessage( other.errorMessage() ) {
assert( type() != ResultType::Ok );
}

template <typename U>
static auto ok( U const& value ) -> BasicResult {
return { ResultType::Ok, value };
}
static auto ok() -> BasicResult { return { ResultType::Ok }; }
static auto logicError( std::string&& message )
-> BasicResult {
return { ResultType::LogicError, CATCH_MOVE(message) };
}
static auto runtimeError( std::string&& message )
-> BasicResult {
return { ResultType::RuntimeError, CATCH_MOVE(message) };
}

explicit operator bool() const {
return m_type == ResultType::Ok;
}
auto type() const -> ResultType { return m_type; }
auto errorMessage() const -> std::string const& {
return m_errorMessage;
}

protected:
void enforceOk() const override {

assert( m_type != ResultType::LogicError );
assert( m_type != ResultType::RuntimeError );
if ( m_type != ResultType::Ok )
std::abort();
}

std::string
m_errorMessage; 

BasicResult( ResultType type,
std::string&& message ):
ResultValueBase<T>( type ), m_errorMessage( CATCH_MOVE(message) ) {
assert( m_type != ResultType::Ok );
}

using ResultValueBase<T>::ResultValueBase;
using ResultBase::m_type;
};

class ParseState {
public:
ParseState( ParseResultType type,
TokenStream const& remainingTokens );

ParseResultType type() const { return m_type; }
TokenStream const& remainingTokens() const {
return m_remainingTokens;
}

private:
ParseResultType m_type;
TokenStream m_remainingTokens;
};

using Result = BasicResult<void>;
using ParserResult = BasicResult<ParseResultType>;
using InternalParseResult = BasicResult<ParseState>;

struct HelpColumns {
std::string left;
std::string right;
};

template <typename T>
ParserResult convertInto( std::string const& source, T& target ) {
std::stringstream ss( source );
ss >> target;
if ( ss.fail() ) {
return ParserResult::runtimeError(
"Unable to convert '" + source +
"' to destination type" );
} else {
return ParserResult::ok( ParseResultType::Matched );
}
}
ParserResult convertInto( std::string const& source,
std::string& target );
ParserResult convertInto( std::string const& source, bool& target );

#ifdef CLARA_CONFIG_OPTIONAL_TYPE
template <typename T>
auto convertInto( std::string const& source,
CLARA_CONFIG_OPTIONAL_TYPE<T>& target )
-> ParserResult {
T temp;
auto result = convertInto( source, temp );
if ( result )
target = CATCH_MOVE( temp );
return result;
}
#endif 

struct BoundRef : Catch::Detail::NonCopyable {
virtual ~BoundRef() = default;
virtual bool isContainer() const;
virtual bool isFlag() const;
};
struct BoundValueRefBase : BoundRef {
virtual auto setValue( std::string const& arg )
-> ParserResult = 0;
};
struct BoundFlagRefBase : BoundRef {
virtual auto setFlag( bool flag ) -> ParserResult = 0;
bool isFlag() const override;
};

template <typename T> struct BoundValueRef : BoundValueRefBase {
T& m_ref;

explicit BoundValueRef( T& ref ): m_ref( ref ) {}

ParserResult setValue( std::string const& arg ) override {
return convertInto( arg, m_ref );
}
};

template <typename T>
struct BoundValueRef<std::vector<T>> : BoundValueRefBase {
std::vector<T>& m_ref;

explicit BoundValueRef( std::vector<T>& ref ): m_ref( ref ) {}

auto isContainer() const -> bool override { return true; }

auto setValue( std::string const& arg )
-> ParserResult override {
T temp;
auto result = convertInto( arg, temp );
if ( result )
m_ref.push_back( temp );
return result;
}
};

struct BoundFlagRef : BoundFlagRefBase {
bool& m_ref;

explicit BoundFlagRef( bool& ref ): m_ref( ref ) {}

ParserResult setFlag( bool flag ) override;
};

template <typename ReturnType> struct LambdaInvoker {
static_assert(
std::is_same<ReturnType, ParserResult>::value,
"Lambda must return void or clara::ParserResult" );

template <typename L, typename ArgType>
static auto invoke( L const& lambda, ArgType const& arg )
-> ParserResult {
return lambda( arg );
}
};

template <> struct LambdaInvoker<void> {
template <typename L, typename ArgType>
static auto invoke( L const& lambda, ArgType const& arg )
-> ParserResult {
lambda( arg );
return ParserResult::ok( ParseResultType::Matched );
}
};

template <typename ArgType, typename L>
auto invokeLambda( L const& lambda, std::string const& arg )
-> ParserResult {
ArgType temp{};
auto result = convertInto( arg, temp );
return !result ? result
: LambdaInvoker<typename UnaryLambdaTraits<
L>::ReturnType>::invoke( lambda, temp );
}

template <typename L> struct BoundLambda : BoundValueRefBase {
L m_lambda;

static_assert(
UnaryLambdaTraits<L>::isValid,
"Supplied lambda must take exactly one argument" );
explicit BoundLambda( L const& lambda ): m_lambda( lambda ) {}

auto setValue( std::string const& arg )
-> ParserResult override {
return invokeLambda<typename UnaryLambdaTraits<L>::ArgType>(
m_lambda, arg );
}
};

template <typename L> struct BoundManyLambda : BoundLambda<L> {
explicit BoundManyLambda( L const& lambda ): BoundLambda<L>( lambda ) {}
bool isContainer() const override { return true; }
};

template <typename L> struct BoundFlagLambda : BoundFlagRefBase {
L m_lambda;

static_assert(
UnaryLambdaTraits<L>::isValid,
"Supplied lambda must take exactly one argument" );
static_assert(
std::is_same<typename UnaryLambdaTraits<L>::ArgType,
bool>::value,
"flags must be boolean" );

explicit BoundFlagLambda( L const& lambda ):
m_lambda( lambda ) {}

auto setFlag( bool flag ) -> ParserResult override {
return LambdaInvoker<typename UnaryLambdaTraits<
L>::ReturnType>::invoke( m_lambda, flag );
}
};

enum class Optionality { Optional, Required };

class ParserBase {
public:
virtual ~ParserBase() = default;
virtual auto validate() const -> Result { return Result::ok(); }
virtual auto parse( std::string const& exeName,
TokenStream const& tokens ) const
-> InternalParseResult = 0;
virtual size_t cardinality() const;

InternalParseResult parse( Args const& args ) const;
};

template <typename DerivedT>
class ComposableParserImpl : public ParserBase {
public:
template <typename T>
auto operator|( T const& other ) const -> Parser;
};

template <typename DerivedT>
class ParserRefImpl : public ComposableParserImpl<DerivedT> {
protected:
Optionality m_optionality = Optionality::Optional;
std::shared_ptr<BoundRef> m_ref;
std::string m_hint;
std::string m_description;

explicit ParserRefImpl( std::shared_ptr<BoundRef> const& ref ):
m_ref( ref ) {}

public:
template <typename LambdaT>
ParserRefImpl( accept_many_t,
LambdaT const& ref,
std::string const& hint ):
m_ref( std::make_shared<BoundManyLambda<LambdaT>>( ref ) ),
m_hint( hint ) {}

template <typename T,
typename = typename std::enable_if_t<
!Detail::is_unary_function<T>::value>>
ParserRefImpl( T& ref, std::string const& hint ):
m_ref( std::make_shared<BoundValueRef<T>>( ref ) ),
m_hint( hint ) {}

template <typename LambdaT,
typename = typename std::enable_if_t<
Detail::is_unary_function<LambdaT>::value>>
ParserRefImpl( LambdaT const& ref, std::string const& hint ):
m_ref( std::make_shared<BoundLambda<LambdaT>>( ref ) ),
m_hint( hint ) {}

auto operator()( std::string const& description ) -> DerivedT& {
m_description = description;
return static_cast<DerivedT&>( *this );
}

auto optional() -> DerivedT& {
m_optionality = Optionality::Optional;
return static_cast<DerivedT&>( *this );
}

auto required() -> DerivedT& {
m_optionality = Optionality::Required;
return static_cast<DerivedT&>( *this );
}

auto isOptional() const -> bool {
return m_optionality == Optionality::Optional;
}

auto cardinality() const -> size_t override {
if ( m_ref->isContainer() )
return 0;
else
return 1;
}

std::string const& hint() const { return m_hint; }
};

} 


class Arg : public Detail::ParserRefImpl<Arg> {
public:
using ParserRefImpl::ParserRefImpl;
using ParserBase::parse;

Detail::InternalParseResult
parse(std::string const&,
Detail::TokenStream const& tokens) const override;
};

class Opt : public Detail::ParserRefImpl<Opt> {
protected:
std::vector<std::string> m_optNames;

public:
template <typename LambdaT>
explicit Opt(LambdaT const& ref) :
ParserRefImpl(
std::make_shared<Detail::BoundFlagLambda<LambdaT>>(ref)) {}

explicit Opt(bool& ref);

template <typename LambdaT,
typename = typename std::enable_if_t<
Detail::is_unary_function<LambdaT>::value>>
Opt( LambdaT const& ref, std::string const& hint ):
ParserRefImpl( ref, hint ) {}

template <typename LambdaT>
Opt( accept_many_t, LambdaT const& ref, std::string const& hint ):
ParserRefImpl( accept_many, ref, hint ) {}

template <typename T,
typename = typename std::enable_if_t<
!Detail::is_unary_function<T>::value>>
Opt( T& ref, std::string const& hint ):
ParserRefImpl( ref, hint ) {}

auto operator[](std::string const& optName) -> Opt& {
m_optNames.push_back(optName);
return *this;
}

std::vector<Detail::HelpColumns> getHelpColumns() const;

bool isMatch(std::string const& optToken) const;

using ParserBase::parse;

Detail::InternalParseResult
parse(std::string const&,
Detail::TokenStream const& tokens) const override;

Detail::Result validate() const override;
};

class ExeName : public Detail::ComposableParserImpl<ExeName> {
std::shared_ptr<std::string> m_name;
std::shared_ptr<Detail::BoundValueRefBase> m_ref;

public:
ExeName();
explicit ExeName(std::string& ref);

template <typename LambdaT>
explicit ExeName(LambdaT const& lambda) : ExeName() {
m_ref = std::make_shared<Detail::BoundLambda<LambdaT>>(lambda);
}

Detail::InternalParseResult
parse(std::string const&,
Detail::TokenStream const& tokens) const override;

std::string const& name() const { return *m_name; }
Detail::ParserResult set(std::string const& newName);
};


class Parser : Detail::ParserBase {
mutable ExeName m_exeName;
std::vector<Opt> m_options;
std::vector<Arg> m_args;

public:

auto operator|=(ExeName const& exeName) -> Parser& {
m_exeName = exeName;
return *this;
}

auto operator|=(Arg const& arg) -> Parser& {
m_args.push_back(arg);
return *this;
}

auto operator|=(Opt const& opt) -> Parser& {
m_options.push_back(opt);
return *this;
}

Parser& operator|=(Parser const& other);

template <typename T>
auto operator|(T const& other) const -> Parser {
return Parser(*this) |= other;
}

std::vector<Detail::HelpColumns> getHelpColumns() const;

void writeToStream(std::ostream& os) const;

friend auto operator<<(std::ostream& os, Parser const& parser)
-> std::ostream& {
parser.writeToStream(os);
return os;
}

Detail::Result validate() const override;

using ParserBase::parse;
Detail::InternalParseResult
parse(std::string const& exeName,
Detail::TokenStream const& tokens) const override;
};

class Args {
friend Detail::TokenStream;
std::string m_exeName;
std::vector<std::string> m_args;

public:
Args(int argc, char const* const* argv);
Args(std::initializer_list<std::string> args);

std::string const& exeName() const { return m_exeName; }
};


struct Help : Opt {
Help(bool& showHelpFlag);
};

using Detail::ParserResult;

namespace Detail {
template <typename DerivedT>
template <typename T>
Parser
ComposableParserImpl<DerivedT>::operator|(T const& other) const {
return Parser() | static_cast<DerivedT const&>(*this) | other;
}
}

} 
} 

#if defined( __clang__ )
#    pragma clang diagnostic pop
#endif

#if defined( __GNUC__ )
#    pragma GCC diagnostic pop
#endif

#endif 

namespace Catch {

struct ConfigData;

Clara::Parser makeCommandLineParser( ConfigData& config );

} 

#endif 

namespace Catch {

class Session : Detail::NonCopyable {
public:

Session();
~Session();

void showHelp() const;
void libIdentify();

int applyCommandLine( int argc, char const * const * argv );
#if defined(CATCH_CONFIG_WCHAR) && defined(_WIN32) && defined(UNICODE)
int applyCommandLine( int argc, wchar_t const * const * argv );
#endif

void useConfigData( ConfigData const& configData );

template<typename CharT>
int run(int argc, CharT const * const argv[]) {
if (m_startupExceptions)
return 1;
int returnCode = applyCommandLine(argc, argv);
if (returnCode == 0)
returnCode = run();
return returnCode;
}

int run();

Clara::Parser const& cli() const;
void cli( Clara::Parser const& newParser );
ConfigData& configData();
Config& config();
private:
int runInternal();

Clara::Parser m_cli;
ConfigData m_configData;
Detail::unique_ptr<Config> m_config;
bool m_startupExceptions = false;
};

} 

#endif 


#ifndef CATCH_TAG_ALIAS_HPP_INCLUDED
#define CATCH_TAG_ALIAS_HPP_INCLUDED


#include <string>

namespace Catch {

struct TagAlias {
TagAlias(std::string const& _tag, SourceLineInfo _lineInfo):
tag(_tag),
lineInfo(_lineInfo)
{}

std::string tag;
SourceLineInfo lineInfo;
};

} 

#endif 


#ifndef CATCH_TAG_ALIAS_AUTOREGISTRAR_HPP_INCLUDED
#define CATCH_TAG_ALIAS_AUTOREGISTRAR_HPP_INCLUDED


namespace Catch {

struct RegistrarForTagAliases {
RegistrarForTagAliases( char const* alias, char const* tag, SourceLineInfo const& lineInfo );
};

} 

#define CATCH_REGISTER_TAG_ALIAS( alias, spec ) \
CATCH_INTERNAL_START_WARNINGS_SUPPRESSION \
CATCH_INTERNAL_SUPPRESS_GLOBALS_WARNINGS \
namespace{ Catch::RegistrarForTagAliases INTERNAL_CATCH_UNIQUE_NAME( AutoRegisterTagAlias )( alias, spec, CATCH_INTERNAL_LINEINFO ); } \
CATCH_INTERNAL_STOP_WARNINGS_SUPPRESSION

#endif 


#ifndef CATCH_TEMPLATE_TEST_MACROS_HPP_INCLUDED
#define CATCH_TEMPLATE_TEST_MACROS_HPP_INCLUDED

#if defined(__GNUC__) && !defined(__clang__) && !defined(__ICC) && __GNUC__ < 10
#pragma GCC diagnostic ignored "-Wparentheses"
#endif




#ifndef CATCH_TEST_MACROS_HPP_INCLUDED
#define CATCH_TEST_MACROS_HPP_INCLUDED



#ifndef CATCH_TEST_MACRO_IMPL_HPP_INCLUDED
#define CATCH_TEST_MACRO_IMPL_HPP_INCLUDED



#ifndef CATCH_ASSERTION_HANDLER_HPP_INCLUDED
#define CATCH_ASSERTION_HANDLER_HPP_INCLUDED



#ifndef CATCH_DECOMPOSER_HPP_INCLUDED
#define CATCH_DECOMPOSER_HPP_INCLUDED



#ifndef CATCH_COMPARE_TRAITS_HPP_INCLUDED
#define CATCH_COMPARE_TRAITS_HPP_INCLUDED


#include <type_traits>

namespace Catch {
namespace Detail {

#if defined( __GNUC__ ) && !defined( __clang__ )
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Wbool-compare"
#    pragma GCC diagnostic ignored "-Wextra"
#    pragma GCC diagnostic ignored "-Wfloat-equal"
#endif

#if defined( __clang__ )
#    pragma clang diagnostic push
#    pragma clang diagnostic ignored "-Wfloat-equal"
#endif

#define CATCH_DEFINE_COMPARABLE_TRAIT( id, op )                               \
template <typename, typename, typename = void>                            \
struct is_##id##_comparable : std::false_type {};                         \
template <typename T, typename U>                                         \
struct is_##id##_comparable<                                              \
T,                                                                    \
U,                                                                    \
void_t<decltype( std::declval<T>() op std::declval<U>() )>>           \
: std::true_type {};                                                  \
template <typename, typename = void>                                      \
struct is_##id##_0_comparable : std::false_type {};                       \
template <typename T>                                                     \
struct is_##id##_0_comparable<T,                                          \
void_t<decltype( std::declval<T>() op 0 )>> \
: std::true_type {};

CATCH_DEFINE_COMPARABLE_TRAIT( lt, < )
CATCH_DEFINE_COMPARABLE_TRAIT( le, <= )
CATCH_DEFINE_COMPARABLE_TRAIT( gt, > )
CATCH_DEFINE_COMPARABLE_TRAIT( ge, >= )
CATCH_DEFINE_COMPARABLE_TRAIT( eq, == )
CATCH_DEFINE_COMPARABLE_TRAIT( ne, != )

#undef CATCH_DEFINE_COMPARABLE_TRAIT

#if defined( __GNUC__ ) && !defined( __clang__ )
#    pragma GCC diagnostic pop
#endif
#if defined( __clang__ )
#    pragma clang diagnostic pop
#endif


} 
} 

#endif 


#ifndef CATCH_LOGICAL_TRAITS_HPP_INCLUDED
#define CATCH_LOGICAL_TRAITS_HPP_INCLUDED

#include <type_traits>

namespace Catch {
namespace Detail {

#if defined( __cpp_lib_logical_traits ) && __cpp_lib_logical_traits >= 201510

using std::conjunction;
using std::disjunction;
using std::negation;

#else

template <class...> struct conjunction : std::true_type {};
template <class B1> struct conjunction<B1> : B1 {};
template <class B1, class... Bn>
struct conjunction<B1, Bn...>
: std::conditional_t<bool( B1::value ), conjunction<Bn...>, B1> {};

template <class...> struct disjunction : std::false_type {};
template <class B1> struct disjunction<B1> : B1 {};
template <class B1, class... Bn>
struct disjunction<B1, Bn...>
: std::conditional_t<bool( B1::value ), B1, disjunction<Bn...>> {};

template <class B>
struct negation : std::integral_constant<bool, !bool(B::value)> {};

#endif

} 
} 

#endif 

#include <type_traits>
#include <iosfwd>

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable:4389) 
#pragma warning(disable:4018) 
#pragma warning(disable:4312) 
#pragma warning(disable:4180) 
#pragma warning(disable:4800) 
#endif

#ifdef __clang__
#  pragma clang diagnostic push
#  pragma clang diagnostic ignored "-Wsign-compare"
#elif defined __GNUC__
#  pragma GCC diagnostic push
#  pragma GCC diagnostic ignored "-Wsign-compare"
#endif

namespace Catch {

template <typename T>
struct always_false : std::false_type {};

class ITransientExpression {
bool m_isBinaryExpression;
bool m_result;

public:
auto isBinaryExpression() const -> bool { return m_isBinaryExpression; }
auto getResult() const -> bool { return m_result; }
virtual void streamReconstructedExpression( std::ostream &os ) const = 0;

ITransientExpression( bool isBinaryExpression, bool result )
:   m_isBinaryExpression( isBinaryExpression ),
m_result( result )
{}

ITransientExpression() = default;
ITransientExpression(ITransientExpression const&) = default;
ITransientExpression& operator=(ITransientExpression const&) = default;

virtual ~ITransientExpression(); 

friend std::ostream& operator<<(std::ostream& out, ITransientExpression const& expr) {
expr.streamReconstructedExpression(out);
return out;
}
};

void formatReconstructedExpression( std::ostream &os, std::string const& lhs, StringRef op, std::string const& rhs );

template<typename LhsT, typename RhsT>
class BinaryExpr  : public ITransientExpression {
LhsT m_lhs;
StringRef m_op;
RhsT m_rhs;

void streamReconstructedExpression( std::ostream &os ) const override {
formatReconstructedExpression
( os, Catch::Detail::stringify( m_lhs ), m_op, Catch::Detail::stringify( m_rhs ) );
}

public:
BinaryExpr( bool comparisonResult, LhsT lhs, StringRef op, RhsT rhs )
:   ITransientExpression{ true, comparisonResult },
m_lhs( lhs ),
m_op( op ),
m_rhs( rhs )
{}

template<typename T>
auto operator && ( T ) const -> BinaryExpr<LhsT, RhsT const&> const {
static_assert(always_false<T>::value,
"chained comparisons are not supported inside assertions, "
"wrap the expression inside parentheses, or decompose it");
}

template<typename T>
auto operator || ( T ) const -> BinaryExpr<LhsT, RhsT const&> const {
static_assert(always_false<T>::value,
"chained comparisons are not supported inside assertions, "
"wrap the expression inside parentheses, or decompose it");
}

template<typename T>
auto operator == ( T ) const -> BinaryExpr<LhsT, RhsT const&> const {
static_assert(always_false<T>::value,
"chained comparisons are not supported inside assertions, "
"wrap the expression inside parentheses, or decompose it");
}

template<typename T>
auto operator != ( T ) const -> BinaryExpr<LhsT, RhsT const&> const {
static_assert(always_false<T>::value,
"chained comparisons are not supported inside assertions, "
"wrap the expression inside parentheses, or decompose it");
}

template<typename T>
auto operator > ( T ) const -> BinaryExpr<LhsT, RhsT const&> const {
static_assert(always_false<T>::value,
"chained comparisons are not supported inside assertions, "
"wrap the expression inside parentheses, or decompose it");
}

template<typename T>
auto operator < ( T ) const -> BinaryExpr<LhsT, RhsT const&> const {
static_assert(always_false<T>::value,
"chained comparisons are not supported inside assertions, "
"wrap the expression inside parentheses, or decompose it");
}

template<typename T>
auto operator >= ( T ) const -> BinaryExpr<LhsT, RhsT const&> const {
static_assert(always_false<T>::value,
"chained comparisons are not supported inside assertions, "
"wrap the expression inside parentheses, or decompose it");
}

template<typename T>
auto operator <= ( T ) const -> BinaryExpr<LhsT, RhsT const&> const {
static_assert(always_false<T>::value,
"chained comparisons are not supported inside assertions, "
"wrap the expression inside parentheses, or decompose it");
}
};

template<typename LhsT>
class UnaryExpr : public ITransientExpression {
LhsT m_lhs;

void streamReconstructedExpression( std::ostream &os ) const override {
os << Catch::Detail::stringify( m_lhs );
}

public:
explicit UnaryExpr( LhsT lhs )
:   ITransientExpression{ false, static_cast<bool>(lhs) },
m_lhs( lhs )
{}
};


template<typename LhsT>
class ExprLhs {
LhsT m_lhs;
public:
explicit ExprLhs( LhsT lhs ) : m_lhs( lhs ) {}

#define CATCH_INTERNAL_DEFINE_EXPRESSION_EQUALITY_OPERATOR( id, op )           \
template <typename RhsT>                                                   \
friend auto operator op( ExprLhs&& lhs, RhsT&& rhs )                       \
->std::enable_if_t<                                                    \
Detail::conjunction<Detail::is_##id##_comparable<LhsT, RhsT>,      \
Detail::negation<std::is_arithmetic<           \
std::remove_reference_t<RhsT>>>>::value,   \
BinaryExpr<LhsT, RhsT const&>> {                                   \
return {                                                               \
static_cast<bool>( lhs.m_lhs op rhs ), lhs.m_lhs, #op##_sr, rhs }; \
}                                                                          \
template <typename RhsT>                                                   \
friend auto operator op( ExprLhs&& lhs, RhsT rhs )                         \
->std::enable_if_t<                                                    \
Detail::conjunction<Detail::is_##id##_comparable<LhsT, RhsT>,      \
std::is_arithmetic<RhsT>>::value,              \
BinaryExpr<LhsT, RhsT>> {                                          \
return {                                                               \
static_cast<bool>( lhs.m_lhs op rhs ), lhs.m_lhs, #op##_sr, rhs }; \
}                                                                          \
template <typename RhsT>                                                   \
friend auto operator op( ExprLhs&& lhs, RhsT rhs )                         \
->std::enable_if_t<                                                    \
Detail::conjunction<                                               \
Detail::negation<Detail::is_##id##_comparable<LhsT, RhsT>>,    \
Detail::is_eq_0_comparable<LhsT>,                              \
\
Detail::disjunction<std::is_same<RhsT, int>,                   \
std::is_same<RhsT, long>>>::value,         \
BinaryExpr<LhsT, RhsT>> {                                          \
if ( rhs != 0 ) { throw_test_failure_exception(); }                    \
return {                                                               \
static_cast<bool>( lhs.m_lhs op 0 ), lhs.m_lhs, #op##_sr, rhs };   \
}                                                                          \
template <typename RhsT>                                                   \
friend auto operator op( ExprLhs&& lhs, RhsT rhs )                         \
->std::enable_if_t<                                                    \
Detail::conjunction<                                               \
Detail::negation<Detail::is_##id##_comparable<LhsT, RhsT>>,    \
Detail::is_eq_0_comparable<RhsT>,                              \
\
Detail::disjunction<std::is_same<LhsT, int>,                   \
std::is_same<LhsT, long>>>::value,         \
BinaryExpr<LhsT, RhsT>> {                                          \
if ( lhs.m_lhs != 0 ) { throw_test_failure_exception(); }              \
return { static_cast<bool>( 0 op rhs ), lhs.m_lhs, #op##_sr, rhs };    \
}

CATCH_INTERNAL_DEFINE_EXPRESSION_EQUALITY_OPERATOR( eq, == )
CATCH_INTERNAL_DEFINE_EXPRESSION_EQUALITY_OPERATOR( ne, != )

#undef CATCH_INTERNAL_DEFINE_EXPRESSION_EQUALITY_OPERATOR

#define CATCH_INTERNAL_DEFINE_EXPRESSION_COMPARISON_OPERATOR( id, op )         \
template <typename RhsT>                                                   \
friend auto operator op( ExprLhs&& lhs, RhsT&& rhs )                       \
->std::enable_if_t<                                                    \
Detail::conjunction<Detail::is_##id##_comparable<LhsT, RhsT>,      \
Detail::negation<std::is_arithmetic<           \
std::remove_reference_t<RhsT>>>>::value,   \
BinaryExpr<LhsT, RhsT const&>> {                                   \
return {                                                               \
static_cast<bool>( lhs.m_lhs op rhs ), lhs.m_lhs, #op##_sr, rhs }; \
}                                                                          \
template <typename RhsT>                                                   \
friend auto operator op( ExprLhs&& lhs, RhsT rhs )                         \
->std::enable_if_t<                                                    \
Detail::conjunction<Detail::is_##id##_comparable<LhsT, RhsT>,      \
std::is_arithmetic<RhsT>>::value,              \
BinaryExpr<LhsT, RhsT>> {                                          \
return {                                                               \
static_cast<bool>( lhs.m_lhs op rhs ), lhs.m_lhs, #op##_sr, rhs }; \
}                                                                          \
template <typename RhsT>                                                   \
friend auto operator op( ExprLhs&& lhs, RhsT rhs )                         \
->std::enable_if_t<                                                    \
Detail::conjunction<                                               \
Detail::negation<Detail::is_##id##_comparable<LhsT, RhsT>>,    \
Detail::is_##id##_0_comparable<LhsT>,                          \
std::is_same<RhsT, int>>::value,                               \
BinaryExpr<LhsT, RhsT>> {                                          \
if ( rhs != 0 ) { throw_test_failure_exception(); }                    \
return {                                                               \
static_cast<bool>( lhs.m_lhs op 0 ), lhs.m_lhs, #op##_sr, rhs };   \
}                                                                          \
template <typename RhsT>                                                   \
friend auto operator op( ExprLhs&& lhs, RhsT rhs )                         \
->std::enable_if_t<                                                    \
Detail::conjunction<                                               \
Detail::negation<Detail::is_##id##_comparable<LhsT, RhsT>>,    \
Detail::is_##id##_0_comparable<RhsT>,                          \
std::is_same<LhsT, int>>::value,                               \
BinaryExpr<LhsT, RhsT>> {                                          \
if ( lhs.m_lhs != 0 ) { throw_test_failure_exception(); }              \
return { static_cast<bool>( 0 op rhs ), lhs.m_lhs, #op##_sr, rhs };    \
}

CATCH_INTERNAL_DEFINE_EXPRESSION_COMPARISON_OPERATOR( lt, < )
CATCH_INTERNAL_DEFINE_EXPRESSION_COMPARISON_OPERATOR( le, <= )
CATCH_INTERNAL_DEFINE_EXPRESSION_COMPARISON_OPERATOR( gt, > )
CATCH_INTERNAL_DEFINE_EXPRESSION_COMPARISON_OPERATOR( ge, >= )

#undef CATCH_INTERNAL_DEFINE_EXPRESSION_COMPARISON_OPERATOR


#define CATCH_INTERNAL_DEFINE_EXPRESSION_OPERATOR( op )                        \
template <typename RhsT>                                                   \
friend auto operator op( ExprLhs&& lhs, RhsT&& rhs )                       \
->std::enable_if_t<                                                    \
!std::is_arithmetic<std::remove_reference_t<RhsT>>::value,         \
BinaryExpr<LhsT, RhsT const&>> {                                   \
return {                                                               \
static_cast<bool>( lhs.m_lhs op rhs ), lhs.m_lhs, #op##_sr, rhs }; \
}                                                                          \
template <typename RhsT>                                                   \
friend auto operator op( ExprLhs&& lhs, RhsT rhs )                         \
->std::enable_if_t<std::is_arithmetic<RhsT>::value,                    \
BinaryExpr<LhsT, RhsT>> {                           \
return {                                                               \
static_cast<bool>( lhs.m_lhs op rhs ), lhs.m_lhs, #op##_sr, rhs }; \
}

CATCH_INTERNAL_DEFINE_EXPRESSION_OPERATOR(|)
CATCH_INTERNAL_DEFINE_EXPRESSION_OPERATOR(&)
CATCH_INTERNAL_DEFINE_EXPRESSION_OPERATOR(^)

#undef CATCH_INTERNAL_DEFINE_EXPRESSION_OPERATOR

template<typename RhsT>
friend auto operator && ( ExprLhs &&, RhsT && ) -> BinaryExpr<LhsT, RhsT const&> {
static_assert(always_false<RhsT>::value,
"operator&& is not supported inside assertions, "
"wrap the expression inside parentheses, or decompose it");
}

template<typename RhsT>
friend auto operator || ( ExprLhs &&, RhsT && ) -> BinaryExpr<LhsT, RhsT const&> {
static_assert(always_false<RhsT>::value,
"operator|| is not supported inside assertions, "
"wrap the expression inside parentheses, or decompose it");
}

auto makeUnaryExpr() const -> UnaryExpr<LhsT> {
return UnaryExpr<LhsT>{ m_lhs };
}
};

struct Decomposer {
template<typename T, std::enable_if_t<!std::is_arithmetic<std::remove_reference_t<T>>::value, int> = 0>
friend auto operator <= ( Decomposer &&, T && lhs ) -> ExprLhs<T const&> {
return ExprLhs<const T&>{ lhs };
}

template<typename T, std::enable_if_t<std::is_arithmetic<T>::value, int> = 0>
friend auto operator <= ( Decomposer &&, T value ) -> ExprLhs<T> {
return ExprLhs<T>{ value };
}
};

} 

#ifdef _MSC_VER
#pragma warning(pop)
#endif
#ifdef __clang__
#  pragma clang diagnostic pop
#elif defined __GNUC__
#  pragma GCC diagnostic pop
#endif

#endif 

#include <string>

namespace Catch {

class IResultCapture;

struct AssertionReaction {
bool shouldDebugBreak = false;
bool shouldThrow = false;
};

class AssertionHandler {
AssertionInfo m_assertionInfo;
AssertionReaction m_reaction;
bool m_completed = false;
IResultCapture& m_resultCapture;

public:
AssertionHandler
(   StringRef macroName,
SourceLineInfo const& lineInfo,
StringRef capturedExpression,
ResultDisposition::Flags resultDisposition );
~AssertionHandler() {
if ( !m_completed ) {
m_resultCapture.handleIncomplete( m_assertionInfo );
}
}


template<typename T>
void handleExpr( ExprLhs<T> const& expr ) {
handleExpr( expr.makeUnaryExpr() );
}
void handleExpr( ITransientExpression const& expr );

void handleMessage(ResultWas::OfType resultType, StringRef message);

void handleExceptionThrownAsExpected();
void handleUnexpectedExceptionNotThrown();
void handleExceptionNotThrownAsExpected();
void handleThrowingCallSkipped();
void handleUnexpectedInflightException();

void complete();
void setCompleted();

auto allowThrows() const -> bool;
};

void handleExceptionMatchExpr( AssertionHandler& handler, std::string const& str );

} 

#endif 

#if defined(__GNUC__) && !defined(__clang__) && !defined(__ICC) && __GNUC__ <= 9
#pragma GCC diagnostic ignored "-Wparentheses"
#endif

#if !defined(CATCH_CONFIG_DISABLE)

#if !defined(CATCH_CONFIG_DISABLE_STRINGIFICATION)
#define CATCH_INTERNAL_STRINGIFY(...) #__VA_ARGS__
#else
#define CATCH_INTERNAL_STRINGIFY(...) "Disabled by CATCH_CONFIG_DISABLE_STRINGIFICATION"
#endif

#if defined(CATCH_CONFIG_FAST_COMPILE) || defined(CATCH_CONFIG_DISABLE_EXCEPTIONS)

#define INTERNAL_CATCH_TRY
#define INTERNAL_CATCH_CATCH( capturer )

#else 

#define INTERNAL_CATCH_TRY try
#define INTERNAL_CATCH_CATCH( handler ) catch(...) { handler.handleUnexpectedInflightException(); }

#endif

#define INTERNAL_CATCH_REACT( handler ) handler.complete();

#define INTERNAL_CATCH_TEST( macroName, resultDisposition, ... ) \
do {  \
\
CATCH_INTERNAL_IGNORE_BUT_WARN(__VA_ARGS__); \
Catch::AssertionHandler catchAssertionHandler( macroName##_catch_sr, CATCH_INTERNAL_LINEINFO, CATCH_INTERNAL_STRINGIFY(__VA_ARGS__), resultDisposition ); \
INTERNAL_CATCH_TRY { \
CATCH_INTERNAL_START_WARNINGS_SUPPRESSION \
CATCH_INTERNAL_SUPPRESS_PARENTHESES_WARNINGS \
catchAssertionHandler.handleExpr( Catch::Decomposer() <= __VA_ARGS__ ); \
CATCH_INTERNAL_STOP_WARNINGS_SUPPRESSION \
} INTERNAL_CATCH_CATCH( catchAssertionHandler ) \
INTERNAL_CATCH_REACT( catchAssertionHandler ) \
} while( (void)0, (false) && static_cast<const bool&>( !!(__VA_ARGS__) ) ) 

#define INTERNAL_CATCH_IF( macroName, resultDisposition, ... ) \
INTERNAL_CATCH_TEST( macroName, resultDisposition, __VA_ARGS__ ); \
if( Catch::getResultCapture().lastAssertionPassed() )

#define INTERNAL_CATCH_ELSE( macroName, resultDisposition, ... ) \
INTERNAL_CATCH_TEST( macroName, resultDisposition, __VA_ARGS__ ); \
if( !Catch::getResultCapture().lastAssertionPassed() )

#define INTERNAL_CATCH_NO_THROW( macroName, resultDisposition, ... ) \
do { \
Catch::AssertionHandler catchAssertionHandler( macroName##_catch_sr, CATCH_INTERNAL_LINEINFO, CATCH_INTERNAL_STRINGIFY(__VA_ARGS__), resultDisposition ); \
try { \
CATCH_INTERNAL_START_WARNINGS_SUPPRESSION \
CATCH_INTERNAL_SUPPRESS_USELESS_CAST_WARNINGS \
static_cast<void>(__VA_ARGS__); \
CATCH_INTERNAL_STOP_WARNINGS_SUPPRESSION \
catchAssertionHandler.handleExceptionNotThrownAsExpected(); \
} \
catch( ... ) { \
catchAssertionHandler.handleUnexpectedInflightException(); \
} \
INTERNAL_CATCH_REACT( catchAssertionHandler ) \
} while( false )

#define INTERNAL_CATCH_THROWS( macroName, resultDisposition, ... ) \
do { \
Catch::AssertionHandler catchAssertionHandler( macroName##_catch_sr, CATCH_INTERNAL_LINEINFO, CATCH_INTERNAL_STRINGIFY(__VA_ARGS__), resultDisposition); \
if( catchAssertionHandler.allowThrows() ) \
try { \
CATCH_INTERNAL_START_WARNINGS_SUPPRESSION \
CATCH_INTERNAL_SUPPRESS_USELESS_CAST_WARNINGS \
static_cast<void>(__VA_ARGS__); \
CATCH_INTERNAL_STOP_WARNINGS_SUPPRESSION \
catchAssertionHandler.handleUnexpectedExceptionNotThrown(); \
} \
catch( ... ) { \
catchAssertionHandler.handleExceptionThrownAsExpected(); \
} \
else \
catchAssertionHandler.handleThrowingCallSkipped(); \
INTERNAL_CATCH_REACT( catchAssertionHandler ) \
} while( false )

#define INTERNAL_CATCH_THROWS_AS( macroName, exceptionType, resultDisposition, expr ) \
do { \
Catch::AssertionHandler catchAssertionHandler( macroName##_catch_sr, CATCH_INTERNAL_LINEINFO, CATCH_INTERNAL_STRINGIFY(expr) ", " CATCH_INTERNAL_STRINGIFY(exceptionType), resultDisposition ); \
if( catchAssertionHandler.allowThrows() ) \
try { \
CATCH_INTERNAL_START_WARNINGS_SUPPRESSION \
CATCH_INTERNAL_SUPPRESS_USELESS_CAST_WARNINGS \
static_cast<void>(expr); \
CATCH_INTERNAL_STOP_WARNINGS_SUPPRESSION \
catchAssertionHandler.handleUnexpectedExceptionNotThrown(); \
} \
catch( exceptionType const& ) { \
catchAssertionHandler.handleExceptionThrownAsExpected(); \
} \
catch( ... ) { \
catchAssertionHandler.handleUnexpectedInflightException(); \
} \
else \
catchAssertionHandler.handleThrowingCallSkipped(); \
INTERNAL_CATCH_REACT( catchAssertionHandler ) \
} while( false )



#define INTERNAL_CATCH_THROWS_STR_MATCHES( macroName, resultDisposition, matcher, ... ) \
do { \
Catch::AssertionHandler catchAssertionHandler( macroName##_catch_sr, CATCH_INTERNAL_LINEINFO, CATCH_INTERNAL_STRINGIFY(__VA_ARGS__) ", " CATCH_INTERNAL_STRINGIFY(matcher), resultDisposition ); \
if( catchAssertionHandler.allowThrows() ) \
try { \
CATCH_INTERNAL_START_WARNINGS_SUPPRESSION \
CATCH_INTERNAL_SUPPRESS_USELESS_CAST_WARNINGS \
static_cast<void>(__VA_ARGS__); \
CATCH_INTERNAL_STOP_WARNINGS_SUPPRESSION \
catchAssertionHandler.handleUnexpectedExceptionNotThrown(); \
} \
catch( ... ) { \
Catch::handleExceptionMatchExpr( catchAssertionHandler, matcher ); \
} \
else \
catchAssertionHandler.handleThrowingCallSkipped(); \
INTERNAL_CATCH_REACT( catchAssertionHandler ) \
} while( false )

#endif 

#endif 


#ifndef CATCH_SECTION_HPP_INCLUDED
#define CATCH_SECTION_HPP_INCLUDED



#ifndef CATCH_TIMER_HPP_INCLUDED
#define CATCH_TIMER_HPP_INCLUDED

#include <cstdint>

namespace Catch {

class Timer {
uint64_t m_nanoseconds = 0;
public:
void start();
auto getElapsedNanoseconds() const -> uint64_t;
auto getElapsedMicroseconds() const -> uint64_t;
auto getElapsedMilliseconds() const -> unsigned int;
auto getElapsedSeconds() const -> double;
};

} 

#endif 

namespace Catch {

class Section : Detail::NonCopyable {
public:
Section( SectionInfo&& info );
~Section();

explicit operator bool() const;

private:
SectionInfo m_info;

Counts m_assertions;
bool m_sectionIncluded;
Timer m_timer;
};

} 

#define INTERNAL_CATCH_SECTION( ... ) \
CATCH_INTERNAL_START_WARNINGS_SUPPRESSION \
CATCH_INTERNAL_SUPPRESS_UNUSED_VARIABLE_WARNINGS \
if( Catch::Section const& INTERNAL_CATCH_UNIQUE_NAME( catch_internal_Section ) = Catch::SectionInfo( CATCH_INTERNAL_LINEINFO, __VA_ARGS__ ) ) \
CATCH_INTERNAL_STOP_WARNINGS_SUPPRESSION

#define INTERNAL_CATCH_DYNAMIC_SECTION( ... ) \
CATCH_INTERNAL_START_WARNINGS_SUPPRESSION \
CATCH_INTERNAL_SUPPRESS_UNUSED_VARIABLE_WARNINGS \
if( Catch::Section const& INTERNAL_CATCH_UNIQUE_NAME( catch_internal_Section ) = Catch::SectionInfo( CATCH_INTERNAL_LINEINFO, (Catch::ReusableStringStream() << __VA_ARGS__).str() ) ) \
CATCH_INTERNAL_STOP_WARNINGS_SUPPRESSION

#endif 


#ifndef CATCH_TEST_REGISTRY_HPP_INCLUDED
#define CATCH_TEST_REGISTRY_HPP_INCLUDED



#ifndef CATCH_INTERFACES_TESTCASE_HPP_INCLUDED
#define CATCH_INTERFACES_TESTCASE_HPP_INCLUDED

#include <vector>

namespace Catch {

class TestSpec;
struct TestCaseInfo;

class ITestInvoker {
public:
virtual void invoke () const = 0;
virtual ~ITestInvoker(); 
};

class TestCaseHandle;
class IConfig;

class ITestCaseRegistry {
public:
virtual ~ITestCaseRegistry(); 
virtual std::vector<TestCaseInfo* > const& getAllInfos() const = 0;
virtual std::vector<TestCaseHandle> const& getAllTests() const = 0;
virtual std::vector<TestCaseHandle> const& getAllTestsSorted( IConfig const& config ) const = 0;
};

bool isThrowSafe( TestCaseHandle const& testCase, IConfig const& config );
bool matchTest( TestCaseHandle const& testCase, TestSpec const& testSpec, IConfig const& config );
std::vector<TestCaseHandle> filterTests( std::vector<TestCaseHandle> const& testCases, TestSpec const& testSpec, IConfig const& config );
std::vector<TestCaseHandle> const& getAllTestCasesSorted( IConfig const& config );

}

#endif 


#ifndef CATCH_PREPROCESSOR_REMOVE_PARENS_HPP_INCLUDED
#define CATCH_PREPROCESSOR_REMOVE_PARENS_HPP_INCLUDED

#define INTERNAL_CATCH_EXPAND1( param ) INTERNAL_CATCH_EXPAND2( param )
#define INTERNAL_CATCH_EXPAND2( ... ) INTERNAL_CATCH_NO##__VA_ARGS__
#define INTERNAL_CATCH_DEF( ... ) INTERNAL_CATCH_DEF __VA_ARGS__
#define INTERNAL_CATCH_NOINTERNAL_CATCH_DEF

#define INTERNAL_CATCH_REMOVE_PARENS( ... ) \
INTERNAL_CATCH_EXPAND1( INTERNAL_CATCH_DEF __VA_ARGS__ )

#endif 

#if defined(__GNUC__) && !defined(__clang__) && __GNUC__ <= 5
#pragma GCC diagnostic ignored "-Wunused-variable"
#endif



namespace Catch {

template<typename C>
class TestInvokerAsMethod : public ITestInvoker {
void (C::*m_testAsMethod)();
public:
TestInvokerAsMethod( void (C::*testAsMethod)() ) noexcept : m_testAsMethod( testAsMethod ) {}

void invoke() const override {
C obj;
(obj.*m_testAsMethod)();
}
};

Detail::unique_ptr<ITestInvoker> makeTestInvoker( void(*testAsFunction)() );

template<typename C>
Detail::unique_ptr<ITestInvoker> makeTestInvoker( void (C::*testAsMethod)() ) {
return Detail::make_unique<TestInvokerAsMethod<C>>( testAsMethod );
}

struct NameAndTags {
constexpr NameAndTags( StringRef name_ = StringRef(),
StringRef tags_ = StringRef() ) noexcept:
name( name_ ), tags( tags_ ) {}
StringRef name;
StringRef tags;
};

struct AutoReg : Detail::NonCopyable {
AutoReg( Detail::unique_ptr<ITestInvoker> invoker, SourceLineInfo const& lineInfo, StringRef classOrMethod, NameAndTags const& nameAndTags ) noexcept;
};

} 

#if defined(CATCH_CONFIG_DISABLE)
#define INTERNAL_CATCH_TESTCASE_NO_REGISTRATION( TestName, ... ) \
static inline void TestName()
#define INTERNAL_CATCH_TESTCASE_METHOD_NO_REGISTRATION( TestName, ClassName, ... ) \
namespace{                        \
struct TestName : INTERNAL_CATCH_REMOVE_PARENS(ClassName) { \
void test();              \
};                            \
}                                 \
void TestName::test()
#endif

#define INTERNAL_CATCH_TESTCASE2( TestName, ... ) \
static void TestName(); \
CATCH_INTERNAL_START_WARNINGS_SUPPRESSION \
CATCH_INTERNAL_SUPPRESS_GLOBALS_WARNINGS \
CATCH_INTERNAL_SUPPRESS_UNUSED_VARIABLE_WARNINGS \
namespace{ Catch::AutoReg INTERNAL_CATCH_UNIQUE_NAME( autoRegistrar )( Catch::makeTestInvoker( &TestName ), CATCH_INTERNAL_LINEINFO, Catch::StringRef(), Catch::NameAndTags{ __VA_ARGS__ } ); }  \
CATCH_INTERNAL_STOP_WARNINGS_SUPPRESSION \
static void TestName()
#define INTERNAL_CATCH_TESTCASE( ... ) \
INTERNAL_CATCH_TESTCASE2( INTERNAL_CATCH_UNIQUE_NAME( CATCH2_INTERNAL_TEST_ ), __VA_ARGS__ )

#define INTERNAL_CATCH_METHOD_AS_TEST_CASE( QualifiedMethod, ... ) \
CATCH_INTERNAL_START_WARNINGS_SUPPRESSION \
CATCH_INTERNAL_SUPPRESS_GLOBALS_WARNINGS \
CATCH_INTERNAL_SUPPRESS_UNUSED_VARIABLE_WARNINGS \
namespace{ Catch::AutoReg INTERNAL_CATCH_UNIQUE_NAME( autoRegistrar )( Catch::makeTestInvoker( &QualifiedMethod ), CATCH_INTERNAL_LINEINFO, "&" #QualifiedMethod, Catch::NameAndTags{ __VA_ARGS__ } ); }  \
CATCH_INTERNAL_STOP_WARNINGS_SUPPRESSION

#define INTERNAL_CATCH_TEST_CASE_METHOD2( TestName, ClassName, ... )\
CATCH_INTERNAL_START_WARNINGS_SUPPRESSION \
CATCH_INTERNAL_SUPPRESS_GLOBALS_WARNINGS \
CATCH_INTERNAL_SUPPRESS_UNUSED_VARIABLE_WARNINGS \
namespace{ \
struct TestName : INTERNAL_CATCH_REMOVE_PARENS(ClassName) { \
void test(); \
}; \
Catch::AutoReg INTERNAL_CATCH_UNIQUE_NAME( autoRegistrar ) ( Catch::makeTestInvoker( &TestName::test ), CATCH_INTERNAL_LINEINFO, #ClassName, Catch::NameAndTags{ __VA_ARGS__ } );  \
} \
CATCH_INTERNAL_STOP_WARNINGS_SUPPRESSION \
void TestName::test()
#define INTERNAL_CATCH_TEST_CASE_METHOD( ClassName, ... ) \
INTERNAL_CATCH_TEST_CASE_METHOD2( INTERNAL_CATCH_UNIQUE_NAME( CATCH2_INTERNAL_TEST_ ), ClassName, __VA_ARGS__ )

#define INTERNAL_CATCH_REGISTER_TESTCASE( Function, ... ) \
do { \
CATCH_INTERNAL_START_WARNINGS_SUPPRESSION \
CATCH_INTERNAL_SUPPRESS_GLOBALS_WARNINGS \
CATCH_INTERNAL_SUPPRESS_UNUSED_VARIABLE_WARNINGS \
Catch::AutoReg INTERNAL_CATCH_UNIQUE_NAME( autoRegistrar )( Catch::makeTestInvoker( Function ), CATCH_INTERNAL_LINEINFO, Catch::StringRef(), Catch::NameAndTags{ __VA_ARGS__ } );  \
CATCH_INTERNAL_STOP_WARNINGS_SUPPRESSION \
} while(false)


#endif 



#if defined(CATCH_CONFIG_PREFIX_ALL) && !defined(CATCH_CONFIG_DISABLE)

#define CATCH_REQUIRE( ... ) INTERNAL_CATCH_TEST( "CATCH_REQUIRE", Catch::ResultDisposition::Normal, __VA_ARGS__ )
#define CATCH_REQUIRE_FALSE( ... ) INTERNAL_CATCH_TEST( "CATCH_REQUIRE_FALSE", Catch::ResultDisposition::Normal | Catch::ResultDisposition::FalseTest, __VA_ARGS__ )

#define CATCH_REQUIRE_THROWS( ... ) INTERNAL_CATCH_THROWS( "CATCH_REQUIRE_THROWS", Catch::ResultDisposition::Normal, __VA_ARGS__ )
#define CATCH_REQUIRE_THROWS_AS( expr, exceptionType ) INTERNAL_CATCH_THROWS_AS( "CATCH_REQUIRE_THROWS_AS", exceptionType, Catch::ResultDisposition::Normal, expr )
#define CATCH_REQUIRE_NOTHROW( ... ) INTERNAL_CATCH_NO_THROW( "CATCH_REQUIRE_NOTHROW", Catch::ResultDisposition::Normal, __VA_ARGS__ )

#define CATCH_CHECK( ... ) INTERNAL_CATCH_TEST( "CATCH_CHECK", Catch::ResultDisposition::ContinueOnFailure, __VA_ARGS__ )
#define CATCH_CHECK_FALSE( ... ) INTERNAL_CATCH_TEST( "CATCH_CHECK_FALSE", Catch::ResultDisposition::ContinueOnFailure | Catch::ResultDisposition::FalseTest, __VA_ARGS__ )
#define CATCH_CHECKED_IF( ... ) INTERNAL_CATCH_IF( "CATCH_CHECKED_IF", Catch::ResultDisposition::ContinueOnFailure | Catch::ResultDisposition::SuppressFail, __VA_ARGS__ )
#define CATCH_CHECKED_ELSE( ... ) INTERNAL_CATCH_ELSE( "CATCH_CHECKED_ELSE", Catch::ResultDisposition::ContinueOnFailure | Catch::ResultDisposition::SuppressFail, __VA_ARGS__ )
#define CATCH_CHECK_NOFAIL( ... ) INTERNAL_CATCH_TEST( "CATCH_CHECK_NOFAIL", Catch::ResultDisposition::ContinueOnFailure | Catch::ResultDisposition::SuppressFail, __VA_ARGS__ )

#define CATCH_CHECK_THROWS( ... )  INTERNAL_CATCH_THROWS( "CATCH_CHECK_THROWS", Catch::ResultDisposition::ContinueOnFailure, __VA_ARGS__ )
#define CATCH_CHECK_THROWS_AS( expr, exceptionType ) INTERNAL_CATCH_THROWS_AS( "CATCH_CHECK_THROWS_AS", exceptionType, Catch::ResultDisposition::ContinueOnFailure, expr )
#define CATCH_CHECK_NOTHROW( ... ) INTERNAL_CATCH_NO_THROW( "CATCH_CHECK_NOTHROW", Catch::ResultDisposition::ContinueOnFailure, __VA_ARGS__ )

#define CATCH_TEST_CASE( ... ) INTERNAL_CATCH_TESTCASE( __VA_ARGS__ )
#define CATCH_TEST_CASE_METHOD( className, ... ) INTERNAL_CATCH_TEST_CASE_METHOD( className, __VA_ARGS__ )
#define CATCH_METHOD_AS_TEST_CASE( method, ... ) INTERNAL_CATCH_METHOD_AS_TEST_CASE( method, __VA_ARGS__ )
#define CATCH_REGISTER_TEST_CASE( Function, ... ) INTERNAL_CATCH_REGISTER_TESTCASE( Function, __VA_ARGS__ )
#define CATCH_SECTION( ... ) INTERNAL_CATCH_SECTION( __VA_ARGS__ )
#define CATCH_DYNAMIC_SECTION( ... ) INTERNAL_CATCH_DYNAMIC_SECTION( __VA_ARGS__ )
#define CATCH_FAIL( ... ) INTERNAL_CATCH_MSG( "CATCH_FAIL", Catch::ResultWas::ExplicitFailure, Catch::ResultDisposition::Normal, __VA_ARGS__ )
#define CATCH_FAIL_CHECK( ... ) INTERNAL_CATCH_MSG( "CATCH_FAIL_CHECK", Catch::ResultWas::ExplicitFailure, Catch::ResultDisposition::ContinueOnFailure, __VA_ARGS__ )
#define CATCH_SUCCEED( ... ) INTERNAL_CATCH_MSG( "CATCH_SUCCEED", Catch::ResultWas::Ok, Catch::ResultDisposition::ContinueOnFailure, __VA_ARGS__ )


#if !defined(CATCH_CONFIG_RUNTIME_STATIC_REQUIRE)
#define CATCH_STATIC_REQUIRE( ... )       static_assert(   __VA_ARGS__ ,      #__VA_ARGS__ );     CATCH_SUCCEED( #__VA_ARGS__ )
#define CATCH_STATIC_REQUIRE_FALSE( ... ) static_assert( !(__VA_ARGS__), "!(" #__VA_ARGS__ ")" ); CATCH_SUCCEED( #__VA_ARGS__ )
#define CATCH_STATIC_CHECK( ... )       static_assert(   __VA_ARGS__ ,      #__VA_ARGS__ );     CATCH_SUCCEED( #__VA_ARGS__ )
#define CATCH_STATIC_CHECK_FALSE( ... ) static_assert( !(__VA_ARGS__), "!(" #__VA_ARGS__ ")" ); CATCH_SUCCEED( #__VA_ARGS__ )
#else
#define CATCH_STATIC_REQUIRE( ... )       CATCH_REQUIRE( __VA_ARGS__ )
#define CATCH_STATIC_REQUIRE_FALSE( ... ) CATCH_REQUIRE_FALSE( __VA_ARGS__ )
#define CATCH_STATIC_CHECK( ... )       CATCH_CHECK( __VA_ARGS__ )
#define CATCH_STATIC_CHECK_FALSE( ... ) CATCH_CHECK_FALSE( __VA_ARGS__ )
#endif


#define CATCH_SCENARIO( ... ) CATCH_TEST_CASE( "Scenario: " __VA_ARGS__ )
#define CATCH_SCENARIO_METHOD( className, ... ) INTERNAL_CATCH_TEST_CASE_METHOD( className, "Scenario: " __VA_ARGS__ )
#define CATCH_GIVEN( desc )     INTERNAL_CATCH_DYNAMIC_SECTION( "    Given: " << desc )
#define CATCH_AND_GIVEN( desc ) INTERNAL_CATCH_DYNAMIC_SECTION( "And given: " << desc )
#define CATCH_WHEN( desc )      INTERNAL_CATCH_DYNAMIC_SECTION( "     When: " << desc )
#define CATCH_AND_WHEN( desc )  INTERNAL_CATCH_DYNAMIC_SECTION( " And when: " << desc )
#define CATCH_THEN( desc )      INTERNAL_CATCH_DYNAMIC_SECTION( "     Then: " << desc )
#define CATCH_AND_THEN( desc )  INTERNAL_CATCH_DYNAMIC_SECTION( "      And: " << desc )

#elif defined(CATCH_CONFIG_PREFIX_ALL) && defined(CATCH_CONFIG_DISABLE) 

#define CATCH_REQUIRE( ... )        (void)(0)
#define CATCH_REQUIRE_FALSE( ... )  (void)(0)

#define CATCH_REQUIRE_THROWS( ... ) (void)(0)
#define CATCH_REQUIRE_THROWS_AS( expr, exceptionType ) (void)(0)
#define CATCH_REQUIRE_NOTHROW( ... ) (void)(0)

#define CATCH_CHECK( ... )         (void)(0)
#define CATCH_CHECK_FALSE( ... )   (void)(0)
#define CATCH_CHECKED_IF( ... )    if (__VA_ARGS__)
#define CATCH_CHECKED_ELSE( ... )  if (!(__VA_ARGS__))
#define CATCH_CHECK_NOFAIL( ... )  (void)(0)

#define CATCH_CHECK_THROWS( ... )  (void)(0)
#define CATCH_CHECK_THROWS_AS( expr, exceptionType ) (void)(0)
#define CATCH_CHECK_NOTHROW( ... ) (void)(0)

#define CATCH_TEST_CASE( ... ) INTERNAL_CATCH_TESTCASE_NO_REGISTRATION(INTERNAL_CATCH_UNIQUE_NAME( CATCH2_INTERNAL_TEST_ ))
#define CATCH_TEST_CASE_METHOD( className, ... ) INTERNAL_CATCH_TESTCASE_NO_REGISTRATION(INTERNAL_CATCH_UNIQUE_NAME( CATCH2_INTERNAL_TEST_ ))
#define CATCH_METHOD_AS_TEST_CASE( method, ... )
#define CATCH_REGISTER_TEST_CASE( Function, ... ) (void)(0)
#define CATCH_SECTION( ... )
#define CATCH_DYNAMIC_SECTION( ... )
#define CATCH_FAIL( ... ) (void)(0)
#define CATCH_FAIL_CHECK( ... ) (void)(0)
#define CATCH_SUCCEED( ... ) (void)(0)

#define CATCH_STATIC_REQUIRE( ... )       (void)(0)
#define CATCH_STATIC_REQUIRE_FALSE( ... ) (void)(0)
#define CATCH_STATIC_CHECK( ... )       (void)(0)
#define CATCH_STATIC_CHECK_FALSE( ... ) (void)(0)

#define CATCH_SCENARIO( ... ) INTERNAL_CATCH_TESTCASE_NO_REGISTRATION(INTERNAL_CATCH_UNIQUE_NAME( CATCH2_INTERNAL_TEST_ ))
#define CATCH_SCENARIO_METHOD( className, ... ) INTERNAL_CATCH_TESTCASE_METHOD_NO_REGISTRATION(INTERNAL_CATCH_UNIQUE_NAME( CATCH2_INTERNAL_TEST_ ), className )
#define CATCH_GIVEN( desc )
#define CATCH_AND_GIVEN( desc )
#define CATCH_WHEN( desc )
#define CATCH_AND_WHEN( desc )
#define CATCH_THEN( desc )
#define CATCH_AND_THEN( desc )

#elif !defined(CATCH_CONFIG_PREFIX_ALL) && !defined(CATCH_CONFIG_DISABLE) 

#define REQUIRE( ... ) INTERNAL_CATCH_TEST( "REQUIRE", Catch::ResultDisposition::Normal, __VA_ARGS__  )
#define REQUIRE_FALSE( ... ) INTERNAL_CATCH_TEST( "REQUIRE_FALSE", Catch::ResultDisposition::Normal | Catch::ResultDisposition::FalseTest, __VA_ARGS__ )

#define REQUIRE_THROWS( ... ) INTERNAL_CATCH_THROWS( "REQUIRE_THROWS", Catch::ResultDisposition::Normal, __VA_ARGS__ )
#define REQUIRE_THROWS_AS( expr, exceptionType ) INTERNAL_CATCH_THROWS_AS( "REQUIRE_THROWS_AS", exceptionType, Catch::ResultDisposition::Normal, expr )
#define REQUIRE_NOTHROW( ... ) INTERNAL_CATCH_NO_THROW( "REQUIRE_NOTHROW", Catch::ResultDisposition::Normal, __VA_ARGS__ )

#define CHECK( ... ) INTERNAL_CATCH_TEST( "CHECK", Catch::ResultDisposition::ContinueOnFailure, __VA_ARGS__ )
#define CHECK_FALSE( ... ) INTERNAL_CATCH_TEST( "CHECK_FALSE", Catch::ResultDisposition::ContinueOnFailure | Catch::ResultDisposition::FalseTest, __VA_ARGS__ )
#define CHECKED_IF( ... ) INTERNAL_CATCH_IF( "CHECKED_IF", Catch::ResultDisposition::ContinueOnFailure | Catch::ResultDisposition::SuppressFail, __VA_ARGS__ )
#define CHECKED_ELSE( ... ) INTERNAL_CATCH_ELSE( "CHECKED_ELSE", Catch::ResultDisposition::ContinueOnFailure | Catch::ResultDisposition::SuppressFail, __VA_ARGS__ )
#define CHECK_NOFAIL( ... ) INTERNAL_CATCH_TEST( "CHECK_NOFAIL", Catch::ResultDisposition::ContinueOnFailure | Catch::ResultDisposition::SuppressFail, __VA_ARGS__ )

#define CHECK_THROWS( ... )  INTERNAL_CATCH_THROWS( "CHECK_THROWS", Catch::ResultDisposition::ContinueOnFailure, __VA_ARGS__ )
#define CHECK_THROWS_AS( expr, exceptionType ) INTERNAL_CATCH_THROWS_AS( "CHECK_THROWS_AS", exceptionType, Catch::ResultDisposition::ContinueOnFailure, expr )
#define CHECK_NOTHROW( ... ) INTERNAL_CATCH_NO_THROW( "CHECK_NOTHROW", Catch::ResultDisposition::ContinueOnFailure, __VA_ARGS__ )

#define TEST_CASE( ... ) INTERNAL_CATCH_TESTCASE( __VA_ARGS__ )
#define TEST_CASE_METHOD( className, ... ) INTERNAL_CATCH_TEST_CASE_METHOD( className, __VA_ARGS__ )
#define METHOD_AS_TEST_CASE( method, ... ) INTERNAL_CATCH_METHOD_AS_TEST_CASE( method, __VA_ARGS__ )
#define REGISTER_TEST_CASE( Function, ... ) INTERNAL_CATCH_REGISTER_TESTCASE( Function, __VA_ARGS__ )
#define SECTION( ... ) INTERNAL_CATCH_SECTION( __VA_ARGS__ )
#define DYNAMIC_SECTION( ... ) INTERNAL_CATCH_DYNAMIC_SECTION( __VA_ARGS__ )
#define FAIL( ... ) INTERNAL_CATCH_MSG( "FAIL", Catch::ResultWas::ExplicitFailure, Catch::ResultDisposition::Normal, __VA_ARGS__ )
#define FAIL_CHECK( ... ) INTERNAL_CATCH_MSG( "FAIL_CHECK", Catch::ResultWas::ExplicitFailure, Catch::ResultDisposition::ContinueOnFailure, __VA_ARGS__ )
#define SUCCEED( ... ) INTERNAL_CATCH_MSG( "SUCCEED", Catch::ResultWas::Ok, Catch::ResultDisposition::ContinueOnFailure, __VA_ARGS__ )


#if !defined(CATCH_CONFIG_RUNTIME_STATIC_REQUIRE)
#define STATIC_REQUIRE( ... )       static_assert(   __VA_ARGS__,  #__VA_ARGS__ ); SUCCEED( #__VA_ARGS__ )
#define STATIC_REQUIRE_FALSE( ... ) static_assert( !(__VA_ARGS__), "!(" #__VA_ARGS__ ")" ); SUCCEED( "!(" #__VA_ARGS__ ")" )
#define STATIC_CHECK( ... )       static_assert(   __VA_ARGS__,  #__VA_ARGS__ ); SUCCEED( #__VA_ARGS__ )
#define STATIC_CHECK_FALSE( ... ) static_assert( !(__VA_ARGS__), "!(" #__VA_ARGS__ ")" ); SUCCEED( "!(" #__VA_ARGS__ ")" )
#else
#define STATIC_REQUIRE( ... )       REQUIRE( __VA_ARGS__ )
#define STATIC_REQUIRE_FALSE( ... ) REQUIRE_FALSE( __VA_ARGS__ )
#define STATIC_CHECK( ... )       CHECK( __VA_ARGS__ )
#define STATIC_CHECK_FALSE( ... ) CHECK_FALSE( __VA_ARGS__ )
#endif

#define SCENARIO( ... ) TEST_CASE( "Scenario: " __VA_ARGS__ )
#define SCENARIO_METHOD( className, ... ) INTERNAL_CATCH_TEST_CASE_METHOD( className, "Scenario: " __VA_ARGS__ )
#define GIVEN( desc )     INTERNAL_CATCH_DYNAMIC_SECTION( "    Given: " << desc )
#define AND_GIVEN( desc ) INTERNAL_CATCH_DYNAMIC_SECTION( "And given: " << desc )
#define WHEN( desc )      INTERNAL_CATCH_DYNAMIC_SECTION( "     When: " << desc )
#define AND_WHEN( desc )  INTERNAL_CATCH_DYNAMIC_SECTION( " And when: " << desc )
#define THEN( desc )      INTERNAL_CATCH_DYNAMIC_SECTION( "     Then: " << desc )
#define AND_THEN( desc )  INTERNAL_CATCH_DYNAMIC_SECTION( "      And: " << desc )

#elif !defined(CATCH_CONFIG_PREFIX_ALL) && defined(CATCH_CONFIG_DISABLE) 

#define REQUIRE( ... )       (void)(0)
#define REQUIRE_FALSE( ... ) (void)(0)

#define REQUIRE_THROWS( ... ) (void)(0)
#define REQUIRE_THROWS_AS( expr, exceptionType ) (void)(0)
#define REQUIRE_NOTHROW( ... ) (void)(0)

#define CHECK( ... ) (void)(0)
#define CHECK_FALSE( ... ) (void)(0)
#define CHECKED_IF( ... ) if (__VA_ARGS__)
#define CHECKED_ELSE( ... ) if (!(__VA_ARGS__))
#define CHECK_NOFAIL( ... ) (void)(0)

#define CHECK_THROWS( ... )  (void)(0)
#define CHECK_THROWS_AS( expr, exceptionType ) (void)(0)
#define CHECK_NOTHROW( ... ) (void)(0)

#define TEST_CASE( ... )  INTERNAL_CATCH_TESTCASE_NO_REGISTRATION(INTERNAL_CATCH_UNIQUE_NAME( CATCH2_INTERNAL_TEST_ ), __VA_ARGS__)
#define TEST_CASE_METHOD( className, ... ) INTERNAL_CATCH_TESTCASE_NO_REGISTRATION(INTERNAL_CATCH_UNIQUE_NAME( CATCH2_INTERNAL_TEST_ ))
#define METHOD_AS_TEST_CASE( method, ... )
#define REGISTER_TEST_CASE( Function, ... ) (void)(0)
#define SECTION( ... )
#define DYNAMIC_SECTION( ... )
#define FAIL( ... ) (void)(0)
#define FAIL_CHECK( ... ) (void)(0)
#define SUCCEED( ... ) (void)(0)

#define STATIC_REQUIRE( ... )       (void)(0)
#define STATIC_REQUIRE_FALSE( ... ) (void)(0)
#define STATIC_CHECK( ... )       (void)(0)
#define STATIC_CHECK_FALSE( ... ) (void)(0)

#define SCENARIO( ... ) INTERNAL_CATCH_TESTCASE_NO_REGISTRATION(INTERNAL_CATCH_UNIQUE_NAME( CATCH2_INTERNAL_TEST_ ) )
#define SCENARIO_METHOD( className, ... ) INTERNAL_CATCH_TESTCASE_METHOD_NO_REGISTRATION(INTERNAL_CATCH_UNIQUE_NAME( CATCH2_INTERNAL_TEST_ ), className )

#define GIVEN( desc )
#define AND_GIVEN( desc )
#define WHEN( desc )
#define AND_WHEN( desc )
#define THEN( desc )
#define AND_THEN( desc )

#endif 


#endif 


#ifndef CATCH_TEMPLATE_TEST_REGISTRY_HPP_INCLUDED
#define CATCH_TEMPLATE_TEST_REGISTRY_HPP_INCLUDED



#ifndef CATCH_PREPROCESSOR_HPP_INCLUDED
#define CATCH_PREPROCESSOR_HPP_INCLUDED


#if defined(__GNUC__)
#pragma GCC system_header
#endif


#define CATCH_RECURSION_LEVEL0(...) __VA_ARGS__
#define CATCH_RECURSION_LEVEL1(...) CATCH_RECURSION_LEVEL0(CATCH_RECURSION_LEVEL0(CATCH_RECURSION_LEVEL0(__VA_ARGS__)))
#define CATCH_RECURSION_LEVEL2(...) CATCH_RECURSION_LEVEL1(CATCH_RECURSION_LEVEL1(CATCH_RECURSION_LEVEL1(__VA_ARGS__)))
#define CATCH_RECURSION_LEVEL3(...) CATCH_RECURSION_LEVEL2(CATCH_RECURSION_LEVEL2(CATCH_RECURSION_LEVEL2(__VA_ARGS__)))
#define CATCH_RECURSION_LEVEL4(...) CATCH_RECURSION_LEVEL3(CATCH_RECURSION_LEVEL3(CATCH_RECURSION_LEVEL3(__VA_ARGS__)))
#define CATCH_RECURSION_LEVEL5(...) CATCH_RECURSION_LEVEL4(CATCH_RECURSION_LEVEL4(CATCH_RECURSION_LEVEL4(__VA_ARGS__)))

#ifdef CATCH_CONFIG_TRADITIONAL_MSVC_PREPROCESSOR
#define INTERNAL_CATCH_EXPAND_VARGS(...) __VA_ARGS__
#define CATCH_RECURSION_LEVEL6(...) CATCH_RECURSION_LEVEL5(CATCH_RECURSION_LEVEL5(CATCH_RECURSION_LEVEL5(__VA_ARGS__)))
#define CATCH_RECURSE(...)  CATCH_RECURSION_LEVEL6(CATCH_RECURSION_LEVEL6(__VA_ARGS__))
#else
#define CATCH_RECURSE(...)  CATCH_RECURSION_LEVEL5(__VA_ARGS__)
#endif

#define CATCH_REC_END(...)
#define CATCH_REC_OUT

#define CATCH_EMPTY()
#define CATCH_DEFER(id) id CATCH_EMPTY()

#define CATCH_REC_GET_END2() 0, CATCH_REC_END
#define CATCH_REC_GET_END1(...) CATCH_REC_GET_END2
#define CATCH_REC_GET_END(...) CATCH_REC_GET_END1
#define CATCH_REC_NEXT0(test, next, ...) next CATCH_REC_OUT
#define CATCH_REC_NEXT1(test, next) CATCH_DEFER ( CATCH_REC_NEXT0 ) ( test, next, 0)
#define CATCH_REC_NEXT(test, next)  CATCH_REC_NEXT1(CATCH_REC_GET_END test, next)

#define CATCH_REC_LIST0(f, x, peek, ...) , f(x) CATCH_DEFER ( CATCH_REC_NEXT(peek, CATCH_REC_LIST1) ) ( f, peek, __VA_ARGS__ )
#define CATCH_REC_LIST1(f, x, peek, ...) , f(x) CATCH_DEFER ( CATCH_REC_NEXT(peek, CATCH_REC_LIST0) ) ( f, peek, __VA_ARGS__ )
#define CATCH_REC_LIST2(f, x, peek, ...)   f(x) CATCH_DEFER ( CATCH_REC_NEXT(peek, CATCH_REC_LIST1) ) ( f, peek, __VA_ARGS__ )

#define CATCH_REC_LIST0_UD(f, userdata, x, peek, ...) , f(userdata, x) CATCH_DEFER ( CATCH_REC_NEXT(peek, CATCH_REC_LIST1_UD) ) ( f, userdata, peek, __VA_ARGS__ )
#define CATCH_REC_LIST1_UD(f, userdata, x, peek, ...) , f(userdata, x) CATCH_DEFER ( CATCH_REC_NEXT(peek, CATCH_REC_LIST0_UD) ) ( f, userdata, peek, __VA_ARGS__ )
#define CATCH_REC_LIST2_UD(f, userdata, x, peek, ...)   f(userdata, x) CATCH_DEFER ( CATCH_REC_NEXT(peek, CATCH_REC_LIST1_UD) ) ( f, userdata, peek, __VA_ARGS__ )

#define CATCH_REC_LIST_UD(f, userdata, ...) CATCH_RECURSE(CATCH_REC_LIST2_UD(f, userdata, __VA_ARGS__, ()()(), ()()(), ()()(), 0))

#define CATCH_REC_LIST(f, ...) CATCH_RECURSE(CATCH_REC_LIST2(f, __VA_ARGS__, ()()(), ()()(), ()()(), 0))

#define INTERNAL_CATCH_STRINGIZE(...) INTERNAL_CATCH_STRINGIZE2(__VA_ARGS__)
#ifndef CATCH_CONFIG_TRADITIONAL_MSVC_PREPROCESSOR
#define INTERNAL_CATCH_STRINGIZE2(...) #__VA_ARGS__
#define INTERNAL_CATCH_STRINGIZE_WITHOUT_PARENS(param) INTERNAL_CATCH_STRINGIZE(INTERNAL_CATCH_REMOVE_PARENS(param))
#else
#define INTERNAL_CATCH_STRINGIZE2(...) INTERNAL_CATCH_STRINGIZE3(__VA_ARGS__)
#define INTERNAL_CATCH_STRINGIZE3(...) #__VA_ARGS__
#define INTERNAL_CATCH_STRINGIZE_WITHOUT_PARENS(param) (INTERNAL_CATCH_STRINGIZE(INTERNAL_CATCH_REMOVE_PARENS(param)) + 1)
#endif

#define INTERNAL_CATCH_MAKE_NAMESPACE2(...) ns_##__VA_ARGS__
#define INTERNAL_CATCH_MAKE_NAMESPACE(name) INTERNAL_CATCH_MAKE_NAMESPACE2(name)

#ifndef CATCH_CONFIG_TRADITIONAL_MSVC_PREPROCESSOR
#define INTERNAL_CATCH_MAKE_TYPE_LIST2(...) decltype(get_wrapper<INTERNAL_CATCH_REMOVE_PARENS_GEN(__VA_ARGS__)>())
#define INTERNAL_CATCH_MAKE_TYPE_LIST(...) INTERNAL_CATCH_MAKE_TYPE_LIST2(INTERNAL_CATCH_REMOVE_PARENS(__VA_ARGS__))
#else
#define INTERNAL_CATCH_MAKE_TYPE_LIST2(...) INTERNAL_CATCH_EXPAND_VARGS(decltype(get_wrapper<INTERNAL_CATCH_REMOVE_PARENS_GEN(__VA_ARGS__)>()))
#define INTERNAL_CATCH_MAKE_TYPE_LIST(...) INTERNAL_CATCH_EXPAND_VARGS(INTERNAL_CATCH_MAKE_TYPE_LIST2(INTERNAL_CATCH_REMOVE_PARENS(__VA_ARGS__)))
#endif

#define INTERNAL_CATCH_MAKE_TYPE_LISTS_FROM_TYPES(...)\
CATCH_REC_LIST(INTERNAL_CATCH_MAKE_TYPE_LIST,__VA_ARGS__)

#define INTERNAL_CATCH_REMOVE_PARENS_1_ARG(_0) INTERNAL_CATCH_REMOVE_PARENS(_0)
#define INTERNAL_CATCH_REMOVE_PARENS_2_ARG(_0, _1) INTERNAL_CATCH_REMOVE_PARENS(_0), INTERNAL_CATCH_REMOVE_PARENS_1_ARG(_1)
#define INTERNAL_CATCH_REMOVE_PARENS_3_ARG(_0, _1, _2) INTERNAL_CATCH_REMOVE_PARENS(_0), INTERNAL_CATCH_REMOVE_PARENS_2_ARG(_1, _2)
#define INTERNAL_CATCH_REMOVE_PARENS_4_ARG(_0, _1, _2, _3) INTERNAL_CATCH_REMOVE_PARENS(_0), INTERNAL_CATCH_REMOVE_PARENS_3_ARG(_1, _2, _3)
#define INTERNAL_CATCH_REMOVE_PARENS_5_ARG(_0, _1, _2, _3, _4) INTERNAL_CATCH_REMOVE_PARENS(_0), INTERNAL_CATCH_REMOVE_PARENS_4_ARG(_1, _2, _3, _4)
#define INTERNAL_CATCH_REMOVE_PARENS_6_ARG(_0, _1, _2, _3, _4, _5) INTERNAL_CATCH_REMOVE_PARENS(_0), INTERNAL_CATCH_REMOVE_PARENS_5_ARG(_1, _2, _3, _4, _5)
#define INTERNAL_CATCH_REMOVE_PARENS_7_ARG(_0, _1, _2, _3, _4, _5, _6) INTERNAL_CATCH_REMOVE_PARENS(_0), INTERNAL_CATCH_REMOVE_PARENS_6_ARG(_1, _2, _3, _4, _5, _6)
#define INTERNAL_CATCH_REMOVE_PARENS_8_ARG(_0, _1, _2, _3, _4, _5, _6, _7) INTERNAL_CATCH_REMOVE_PARENS(_0), INTERNAL_CATCH_REMOVE_PARENS_7_ARG(_1, _2, _3, _4, _5, _6, _7)
#define INTERNAL_CATCH_REMOVE_PARENS_9_ARG(_0, _1, _2, _3, _4, _5, _6, _7, _8) INTERNAL_CATCH_REMOVE_PARENS(_0), INTERNAL_CATCH_REMOVE_PARENS_8_ARG(_1, _2, _3, _4, _5, _6, _7, _8)
#define INTERNAL_CATCH_REMOVE_PARENS_10_ARG(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9) INTERNAL_CATCH_REMOVE_PARENS(_0), INTERNAL_CATCH_REMOVE_PARENS_9_ARG(_1, _2, _3, _4, _5, _6, _7, _8, _9)
#define INTERNAL_CATCH_REMOVE_PARENS_11_ARG(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10) INTERNAL_CATCH_REMOVE_PARENS(_0), INTERNAL_CATCH_REMOVE_PARENS_10_ARG(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10)

#define INTERNAL_CATCH_VA_NARGS_IMPL(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, N, ...) N

#define INTERNAL_CATCH_TYPE_GEN\
template<typename...> struct TypeList {};\
template<typename...Ts>\
constexpr auto get_wrapper() noexcept -> TypeList<Ts...> { return {}; }\
template<template<typename...> class...> struct TemplateTypeList{};\
template<template<typename...> class...Cs>\
constexpr auto get_wrapper() noexcept -> TemplateTypeList<Cs...> { return {}; }\
template<typename...>\
struct append;\
template<typename...>\
struct rewrap;\
template<template<typename...> class, typename...>\
struct create;\
template<template<typename...> class, typename>\
struct convert;\
\
template<typename T> \
struct append<T> { using type = T; };\
template< template<typename...> class L1, typename...E1, template<typename...> class L2, typename...E2, typename...Rest>\
struct append<L1<E1...>, L2<E2...>, Rest...> { using type = typename append<L1<E1...,E2...>, Rest...>::type; };\
template< template<typename...> class L1, typename...E1, typename...Rest>\
struct append<L1<E1...>, TypeList<mpl_::na>, Rest...> { using type = L1<E1...>; };\
\
template< template<typename...> class Container, template<typename...> class List, typename...elems>\
struct rewrap<TemplateTypeList<Container>, List<elems...>> { using type = TypeList<Container<elems...>>; };\
template< template<typename...> class Container, template<typename...> class List, class...Elems, typename...Elements>\
struct rewrap<TemplateTypeList<Container>, List<Elems...>, Elements...> { using type = typename append<TypeList<Container<Elems...>>, typename rewrap<TemplateTypeList<Container>, Elements...>::type>::type; };\
\
template<template <typename...> class Final, template< typename...> class...Containers, typename...Types>\
struct create<Final, TemplateTypeList<Containers...>, TypeList<Types...>> { using type = typename append<Final<>, typename rewrap<TemplateTypeList<Containers>, Types...>::type...>::type; };\
template<template <typename...> class Final, template <typename...> class List, typename...Ts>\
struct convert<Final, List<Ts...>> { using type = typename append<Final<>,TypeList<Ts>...>::type; };

#define INTERNAL_CATCH_NTTP_1(signature, ...)\
template<INTERNAL_CATCH_REMOVE_PARENS(signature)> struct Nttp{};\
template<INTERNAL_CATCH_REMOVE_PARENS(signature)>\
constexpr auto get_wrapper() noexcept -> Nttp<__VA_ARGS__> { return {}; } \
template<template<INTERNAL_CATCH_REMOVE_PARENS(signature)> class...> struct NttpTemplateTypeList{};\
template<template<INTERNAL_CATCH_REMOVE_PARENS(signature)> class...Cs>\
constexpr auto get_wrapper() noexcept -> NttpTemplateTypeList<Cs...> { return {}; } \
\
template< template<INTERNAL_CATCH_REMOVE_PARENS(signature)> class Container, template<INTERNAL_CATCH_REMOVE_PARENS(signature)> class List, INTERNAL_CATCH_REMOVE_PARENS(signature)>\
struct rewrap<NttpTemplateTypeList<Container>, List<__VA_ARGS__>> { using type = TypeList<Container<__VA_ARGS__>>; };\
template< template<INTERNAL_CATCH_REMOVE_PARENS(signature)> class Container, template<INTERNAL_CATCH_REMOVE_PARENS(signature)> class List, INTERNAL_CATCH_REMOVE_PARENS(signature), typename...Elements>\
struct rewrap<NttpTemplateTypeList<Container>, List<__VA_ARGS__>, Elements...> { using type = typename append<TypeList<Container<__VA_ARGS__>>, typename rewrap<NttpTemplateTypeList<Container>, Elements...>::type>::type; };\
template<template <typename...> class Final, template<INTERNAL_CATCH_REMOVE_PARENS(signature)> class...Containers, typename...Types>\
struct create<Final, NttpTemplateTypeList<Containers...>, TypeList<Types...>> { using type = typename append<Final<>, typename rewrap<NttpTemplateTypeList<Containers>, Types...>::type...>::type; };

#define INTERNAL_CATCH_DECLARE_SIG_TEST0(TestName)
#define INTERNAL_CATCH_DECLARE_SIG_TEST1(TestName, signature)\
template<INTERNAL_CATCH_REMOVE_PARENS(signature)>\
static void TestName()
#define INTERNAL_CATCH_DECLARE_SIG_TEST_X(TestName, signature, ...)\
template<INTERNAL_CATCH_REMOVE_PARENS(signature)>\
static void TestName()

#define INTERNAL_CATCH_DEFINE_SIG_TEST0(TestName)
#define INTERNAL_CATCH_DEFINE_SIG_TEST1(TestName, signature)\
template<INTERNAL_CATCH_REMOVE_PARENS(signature)>\
static void TestName()
#define INTERNAL_CATCH_DEFINE_SIG_TEST_X(TestName, signature,...)\
template<INTERNAL_CATCH_REMOVE_PARENS(signature)>\
static void TestName()

#define INTERNAL_CATCH_NTTP_REGISTER0(TestFunc, signature)\
template<typename Type>\
void reg_test(TypeList<Type>, Catch::NameAndTags nameAndTags)\
{\
Catch::AutoReg( Catch::makeTestInvoker(&TestFunc<Type>), CATCH_INTERNAL_LINEINFO, Catch::StringRef(), nameAndTags);\
}

#define INTERNAL_CATCH_NTTP_REGISTER(TestFunc, signature, ...)\
template<INTERNAL_CATCH_REMOVE_PARENS(signature)>\
void reg_test(Nttp<__VA_ARGS__>, Catch::NameAndTags nameAndTags)\
{\
Catch::AutoReg( Catch::makeTestInvoker(&TestFunc<__VA_ARGS__>), CATCH_INTERNAL_LINEINFO, Catch::StringRef(), nameAndTags);\
}

#define INTERNAL_CATCH_NTTP_REGISTER_METHOD0(TestName, signature, ...)\
template<typename Type>\
void reg_test(TypeList<Type>, Catch::StringRef className, Catch::NameAndTags nameAndTags)\
{\
Catch::AutoReg( Catch::makeTestInvoker(&TestName<Type>::test), CATCH_INTERNAL_LINEINFO, className, nameAndTags);\
}

#define INTERNAL_CATCH_NTTP_REGISTER_METHOD(TestName, signature, ...)\
template<INTERNAL_CATCH_REMOVE_PARENS(signature)>\
void reg_test(Nttp<__VA_ARGS__>, Catch::StringRef className, Catch::NameAndTags nameAndTags)\
{\
Catch::AutoReg( Catch::makeTestInvoker(&TestName<__VA_ARGS__>::test), CATCH_INTERNAL_LINEINFO, className, nameAndTags);\
}

#define INTERNAL_CATCH_DECLARE_SIG_TEST_METHOD0(TestName, ClassName)
#define INTERNAL_CATCH_DECLARE_SIG_TEST_METHOD1(TestName, ClassName, signature)\
template<typename TestType> \
struct TestName : INTERNAL_CATCH_REMOVE_PARENS(ClassName)<TestType> { \
void test();\
}

#define INTERNAL_CATCH_DECLARE_SIG_TEST_METHOD_X(TestName, ClassName, signature, ...)\
template<INTERNAL_CATCH_REMOVE_PARENS(signature)> \
struct TestName : INTERNAL_CATCH_REMOVE_PARENS(ClassName)<__VA_ARGS__> { \
void test();\
}

#define INTERNAL_CATCH_DEFINE_SIG_TEST_METHOD0(TestName)
#define INTERNAL_CATCH_DEFINE_SIG_TEST_METHOD1(TestName, signature)\
template<typename TestType> \
void INTERNAL_CATCH_MAKE_NAMESPACE(TestName)::TestName<TestType>::test()
#define INTERNAL_CATCH_DEFINE_SIG_TEST_METHOD_X(TestName, signature, ...)\
template<INTERNAL_CATCH_REMOVE_PARENS(signature)> \
void INTERNAL_CATCH_MAKE_NAMESPACE(TestName)::TestName<__VA_ARGS__>::test()

#ifndef CATCH_CONFIG_TRADITIONAL_MSVC_PREPROCESSOR
#define INTERNAL_CATCH_NTTP_0
#define INTERNAL_CATCH_NTTP_GEN(...) INTERNAL_CATCH_VA_NARGS_IMPL(__VA_ARGS__, INTERNAL_CATCH_NTTP_1(__VA_ARGS__), INTERNAL_CATCH_NTTP_1(__VA_ARGS__), INTERNAL_CATCH_NTTP_1(__VA_ARGS__), INTERNAL_CATCH_NTTP_1(__VA_ARGS__), INTERNAL_CATCH_NTTP_1(__VA_ARGS__), INTERNAL_CATCH_NTTP_1( __VA_ARGS__), INTERNAL_CATCH_NTTP_1( __VA_ARGS__), INTERNAL_CATCH_NTTP_1( __VA_ARGS__), INTERNAL_CATCH_NTTP_1( __VA_ARGS__),INTERNAL_CATCH_NTTP_1( __VA_ARGS__), INTERNAL_CATCH_NTTP_0)
#define INTERNAL_CATCH_DEFINE_SIG_TEST_METHOD(TestName, ...) INTERNAL_CATCH_VA_NARGS_IMPL( "dummy", __VA_ARGS__, INTERNAL_CATCH_DEFINE_SIG_TEST_METHOD_X,INTERNAL_CATCH_DEFINE_SIG_TEST_METHOD_X, INTERNAL_CATCH_DEFINE_SIG_TEST_METHOD_X, INTERNAL_CATCH_DEFINE_SIG_TEST_METHOD_X, INTERNAL_CATCH_DEFINE_SIG_TEST_METHOD_X, INTERNAL_CATCH_DEFINE_SIG_TEST_METHOD_X, INTERNAL_CATCH_DEFINE_SIG_TEST_METHOD_X,INTERNAL_CATCH_DEFINE_SIG_TEST_METHOD_X,INTERNAL_CATCH_DEFINE_SIG_TEST_METHOD_X, INTERNAL_CATCH_DEFINE_SIG_TEST_METHOD1, INTERNAL_CATCH_DEFINE_SIG_TEST_METHOD0)(TestName, __VA_ARGS__)
#define INTERNAL_CATCH_DECLARE_SIG_TEST_METHOD(TestName, ClassName, ...) INTERNAL_CATCH_VA_NARGS_IMPL( "dummy", __VA_ARGS__, INTERNAL_CATCH_DECLARE_SIG_TEST_METHOD_X,INTERNAL_CATCH_DECLARE_SIG_TEST_METHOD_X, INTERNAL_CATCH_DECLARE_SIG_TEST_METHOD_X, INTERNAL_CATCH_DECLARE_SIG_TEST_METHOD_X, INTERNAL_CATCH_DECLARE_SIG_TEST_METHOD_X, INTERNAL_CATCH_DECLARE_SIG_TEST_METHOD_X, INTERNAL_CATCH_DECLARE_SIG_TEST_METHOD_X,INTERNAL_CATCH_DECLARE_SIG_TEST_METHOD_X,INTERNAL_CATCH_DECLARE_SIG_TEST_METHOD_X, INTERNAL_CATCH_DECLARE_SIG_TEST_METHOD1, INTERNAL_CATCH_DECLARE_SIG_TEST_METHOD0)(TestName, ClassName, __VA_ARGS__)
#define INTERNAL_CATCH_NTTP_REG_METHOD_GEN(TestName, ...) INTERNAL_CATCH_VA_NARGS_IMPL( "dummy", __VA_ARGS__, INTERNAL_CATCH_NTTP_REGISTER_METHOD, INTERNAL_CATCH_NTTP_REGISTER_METHOD, INTERNAL_CATCH_NTTP_REGISTER_METHOD, INTERNAL_CATCH_NTTP_REGISTER_METHOD, INTERNAL_CATCH_NTTP_REGISTER_METHOD, INTERNAL_CATCH_NTTP_REGISTER_METHOD, INTERNAL_CATCH_NTTP_REGISTER_METHOD, INTERNAL_CATCH_NTTP_REGISTER_METHOD, INTERNAL_CATCH_NTTP_REGISTER_METHOD, INTERNAL_CATCH_NTTP_REGISTER_METHOD0, INTERNAL_CATCH_NTTP_REGISTER_METHOD0)(TestName, __VA_ARGS__)
#define INTERNAL_CATCH_NTTP_REG_GEN(TestFunc, ...) INTERNAL_CATCH_VA_NARGS_IMPL( "dummy", __VA_ARGS__, INTERNAL_CATCH_NTTP_REGISTER, INTERNAL_CATCH_NTTP_REGISTER, INTERNAL_CATCH_NTTP_REGISTER, INTERNAL_CATCH_NTTP_REGISTER, INTERNAL_CATCH_NTTP_REGISTER, INTERNAL_CATCH_NTTP_REGISTER, INTERNAL_CATCH_NTTP_REGISTER, INTERNAL_CATCH_NTTP_REGISTER, INTERNAL_CATCH_NTTP_REGISTER, INTERNAL_CATCH_NTTP_REGISTER0, INTERNAL_CATCH_NTTP_REGISTER0)(TestFunc, __VA_ARGS__)
#define INTERNAL_CATCH_DEFINE_SIG_TEST(TestName, ...) INTERNAL_CATCH_VA_NARGS_IMPL( "dummy", __VA_ARGS__, INTERNAL_CATCH_DEFINE_SIG_TEST_X, INTERNAL_CATCH_DEFINE_SIG_TEST_X, INTERNAL_CATCH_DEFINE_SIG_TEST_X, INTERNAL_CATCH_DEFINE_SIG_TEST_X, INTERNAL_CATCH_DEFINE_SIG_TEST_X, INTERNAL_CATCH_DEFINE_SIG_TEST_X, INTERNAL_CATCH_DEFINE_SIG_TEST_X, INTERNAL_CATCH_DEFINE_SIG_TEST_X,INTERNAL_CATCH_DEFINE_SIG_TEST_X,INTERNAL_CATCH_DEFINE_SIG_TEST1, INTERNAL_CATCH_DEFINE_SIG_TEST0)(TestName, __VA_ARGS__)
#define INTERNAL_CATCH_DECLARE_SIG_TEST(TestName, ...) INTERNAL_CATCH_VA_NARGS_IMPL( "dummy", __VA_ARGS__, INTERNAL_CATCH_DECLARE_SIG_TEST_X,INTERNAL_CATCH_DECLARE_SIG_TEST_X, INTERNAL_CATCH_DECLARE_SIG_TEST_X, INTERNAL_CATCH_DECLARE_SIG_TEST_X, INTERNAL_CATCH_DECLARE_SIG_TEST_X, INTERNAL_CATCH_DECLARE_SIG_TEST_X, INTERNAL_CATCH_DEFINE_SIG_TEST_X,INTERNAL_CATCH_DECLARE_SIG_TEST_X,INTERNAL_CATCH_DECLARE_SIG_TEST_X, INTERNAL_CATCH_DECLARE_SIG_TEST1, INTERNAL_CATCH_DECLARE_SIG_TEST0)(TestName, __VA_ARGS__)
#define INTERNAL_CATCH_REMOVE_PARENS_GEN(...) INTERNAL_CATCH_VA_NARGS_IMPL(__VA_ARGS__, INTERNAL_CATCH_REMOVE_PARENS_11_ARG,INTERNAL_CATCH_REMOVE_PARENS_10_ARG,INTERNAL_CATCH_REMOVE_PARENS_9_ARG,INTERNAL_CATCH_REMOVE_PARENS_8_ARG,INTERNAL_CATCH_REMOVE_PARENS_7_ARG,INTERNAL_CATCH_REMOVE_PARENS_6_ARG,INTERNAL_CATCH_REMOVE_PARENS_5_ARG,INTERNAL_CATCH_REMOVE_PARENS_4_ARG,INTERNAL_CATCH_REMOVE_PARENS_3_ARG,INTERNAL_CATCH_REMOVE_PARENS_2_ARG,INTERNAL_CATCH_REMOVE_PARENS_1_ARG)(__VA_ARGS__)
#else
#define INTERNAL_CATCH_NTTP_0(signature)
#define INTERNAL_CATCH_NTTP_GEN(...) INTERNAL_CATCH_EXPAND_VARGS(INTERNAL_CATCH_VA_NARGS_IMPL(__VA_ARGS__, INTERNAL_CATCH_NTTP_1, INTERNAL_CATCH_NTTP_1, INTERNAL_CATCH_NTTP_1, INTERNAL_CATCH_NTTP_1, INTERNAL_CATCH_NTTP_1, INTERNAL_CATCH_NTTP_1, INTERNAL_CATCH_NTTP_1, INTERNAL_CATCH_NTTP_1, INTERNAL_CATCH_NTTP_1,INTERNAL_CATCH_NTTP_1, INTERNAL_CATCH_NTTP_0)( __VA_ARGS__))
#define INTERNAL_CATCH_DEFINE_SIG_TEST_METHOD(TestName, ...) INTERNAL_CATCH_EXPAND_VARGS(INTERNAL_CATCH_VA_NARGS_IMPL( "dummy", __VA_ARGS__, INTERNAL_CATCH_DEFINE_SIG_TEST_METHOD_X,INTERNAL_CATCH_DEFINE_SIG_TEST_METHOD_X, INTERNAL_CATCH_DEFINE_SIG_TEST_METHOD_X, INTERNAL_CATCH_DEFINE_SIG_TEST_METHOD_X, INTERNAL_CATCH_DEFINE_SIG_TEST_METHOD_X, INTERNAL_CATCH_DEFINE_SIG_TEST_METHOD_X, INTERNAL_CATCH_DEFINE_SIG_TEST_METHOD_X,INTERNAL_CATCH_DEFINE_SIG_TEST_METHOD_X,INTERNAL_CATCH_DEFINE_SIG_TEST_METHOD_X, INTERNAL_CATCH_DEFINE_SIG_TEST_METHOD1, INTERNAL_CATCH_DEFINE_SIG_TEST_METHOD0)(TestName, __VA_ARGS__))
#define INTERNAL_CATCH_DECLARE_SIG_TEST_METHOD(TestName, ClassName, ...) INTERNAL_CATCH_EXPAND_VARGS(INTERNAL_CATCH_VA_NARGS_IMPL( "dummy", __VA_ARGS__, INTERNAL_CATCH_DECLARE_SIG_TEST_METHOD_X,INTERNAL_CATCH_DECLARE_SIG_TEST_METHOD_X, INTERNAL_CATCH_DECLARE_SIG_TEST_METHOD_X, INTERNAL_CATCH_DECLARE_SIG_TEST_METHOD_X, INTERNAL_CATCH_DECLARE_SIG_TEST_METHOD_X, INTERNAL_CATCH_DECLARE_SIG_TEST_METHOD_X, INTERNAL_CATCH_DECLARE_SIG_TEST_METHOD_X,INTERNAL_CATCH_DECLARE_SIG_TEST_METHOD_X,INTERNAL_CATCH_DECLARE_SIG_TEST_METHOD_X, INTERNAL_CATCH_DECLARE_SIG_TEST_METHOD1, INTERNAL_CATCH_DECLARE_SIG_TEST_METHOD0)(TestName, ClassName, __VA_ARGS__))
#define INTERNAL_CATCH_NTTP_REG_METHOD_GEN(TestName, ...) INTERNAL_CATCH_EXPAND_VARGS(INTERNAL_CATCH_VA_NARGS_IMPL( "dummy", __VA_ARGS__, INTERNAL_CATCH_NTTP_REGISTER_METHOD, INTERNAL_CATCH_NTTP_REGISTER_METHOD, INTERNAL_CATCH_NTTP_REGISTER_METHOD, INTERNAL_CATCH_NTTP_REGISTER_METHOD, INTERNAL_CATCH_NTTP_REGISTER_METHOD, INTERNAL_CATCH_NTTP_REGISTER_METHOD, INTERNAL_CATCH_NTTP_REGISTER_METHOD, INTERNAL_CATCH_NTTP_REGISTER_METHOD, INTERNAL_CATCH_NTTP_REGISTER_METHOD, INTERNAL_CATCH_NTTP_REGISTER_METHOD0, INTERNAL_CATCH_NTTP_REGISTER_METHOD0)(TestName, __VA_ARGS__))
#define INTERNAL_CATCH_NTTP_REG_GEN(TestFunc, ...) INTERNAL_CATCH_EXPAND_VARGS(INTERNAL_CATCH_VA_NARGS_IMPL( "dummy", __VA_ARGS__, INTERNAL_CATCH_NTTP_REGISTER, INTERNAL_CATCH_NTTP_REGISTER, INTERNAL_CATCH_NTTP_REGISTER, INTERNAL_CATCH_NTTP_REGISTER, INTERNAL_CATCH_NTTP_REGISTER, INTERNAL_CATCH_NTTP_REGISTER, INTERNAL_CATCH_NTTP_REGISTER, INTERNAL_CATCH_NTTP_REGISTER, INTERNAL_CATCH_NTTP_REGISTER, INTERNAL_CATCH_NTTP_REGISTER0, INTERNAL_CATCH_NTTP_REGISTER0)(TestFunc, __VA_ARGS__))
#define INTERNAL_CATCH_DEFINE_SIG_TEST(TestName, ...) INTERNAL_CATCH_EXPAND_VARGS(INTERNAL_CATCH_VA_NARGS_IMPL( "dummy", __VA_ARGS__, INTERNAL_CATCH_DEFINE_SIG_TEST_X, INTERNAL_CATCH_DEFINE_SIG_TEST_X, INTERNAL_CATCH_DEFINE_SIG_TEST_X, INTERNAL_CATCH_DEFINE_SIG_TEST_X, INTERNAL_CATCH_DEFINE_SIG_TEST_X, INTERNAL_CATCH_DEFINE_SIG_TEST_X, INTERNAL_CATCH_DEFINE_SIG_TEST_X, INTERNAL_CATCH_DEFINE_SIG_TEST_X,INTERNAL_CATCH_DEFINE_SIG_TEST_X,INTERNAL_CATCH_DEFINE_SIG_TEST1, INTERNAL_CATCH_DEFINE_SIG_TEST0)(TestName, __VA_ARGS__))
#define INTERNAL_CATCH_DECLARE_SIG_TEST(TestName, ...) INTERNAL_CATCH_EXPAND_VARGS(INTERNAL_CATCH_VA_NARGS_IMPL( "dummy", __VA_ARGS__, INTERNAL_CATCH_DECLARE_SIG_TEST_X,INTERNAL_CATCH_DECLARE_SIG_TEST_X, INTERNAL_CATCH_DECLARE_SIG_TEST_X, INTERNAL_CATCH_DECLARE_SIG_TEST_X, INTERNAL_CATCH_DECLARE_SIG_TEST_X, INTERNAL_CATCH_DECLARE_SIG_TEST_X, INTERNAL_CATCH_DEFINE_SIG_TEST_X,INTERNAL_CATCH_DECLARE_SIG_TEST_X,INTERNAL_CATCH_DECLARE_SIG_TEST_X, INTERNAL_CATCH_DECLARE_SIG_TEST1, INTERNAL_CATCH_DECLARE_SIG_TEST0)(TestName, __VA_ARGS__))
#define INTERNAL_CATCH_REMOVE_PARENS_GEN(...) INTERNAL_CATCH_EXPAND_VARGS(INTERNAL_CATCH_VA_NARGS_IMPL(__VA_ARGS__, INTERNAL_CATCH_REMOVE_PARENS_11_ARG,INTERNAL_CATCH_REMOVE_PARENS_10_ARG,INTERNAL_CATCH_REMOVE_PARENS_9_ARG,INTERNAL_CATCH_REMOVE_PARENS_8_ARG,INTERNAL_CATCH_REMOVE_PARENS_7_ARG,INTERNAL_CATCH_REMOVE_PARENS_6_ARG,INTERNAL_CATCH_REMOVE_PARENS_5_ARG,INTERNAL_CATCH_REMOVE_PARENS_4_ARG,INTERNAL_CATCH_REMOVE_PARENS_3_ARG,INTERNAL_CATCH_REMOVE_PARENS_2_ARG,INTERNAL_CATCH_REMOVE_PARENS_1_ARG)(__VA_ARGS__))
#endif

#endif 


#if defined(__GNUC__) && !defined(__clang__) && __GNUC__ <= 5
#pragma GCC diagnostic ignored "-Wunused-variable"
#endif

#if defined(CATCH_CONFIG_DISABLE)
#define INTERNAL_CATCH_TEMPLATE_TEST_CASE_NO_REGISTRATION_2( TestName, TestFunc, Name, Tags, Signature, ... )  \
INTERNAL_CATCH_DEFINE_SIG_TEST(TestFunc, INTERNAL_CATCH_REMOVE_PARENS(Signature))
#define INTERNAL_CATCH_TEMPLATE_TEST_CASE_METHOD_NO_REGISTRATION_2( TestNameClass, TestName, ClassName, Name, Tags, Signature, ... )    \
namespace{                                                                                  \
namespace INTERNAL_CATCH_MAKE_NAMESPACE(TestName) {                                      \
INTERNAL_CATCH_DECLARE_SIG_TEST_METHOD(TestName, ClassName, INTERNAL_CATCH_REMOVE_PARENS(Signature));\
}                                                                                           \
}                                                                                           \
INTERNAL_CATCH_DEFINE_SIG_TEST_METHOD(TestName, INTERNAL_CATCH_REMOVE_PARENS(Signature))

#ifndef CATCH_CONFIG_TRADITIONAL_MSVC_PREPROCESSOR
#define INTERNAL_CATCH_TEMPLATE_TEST_CASE_NO_REGISTRATION(Name, Tags, ...) \
INTERNAL_CATCH_TEMPLATE_TEST_CASE_NO_REGISTRATION_2( INTERNAL_CATCH_UNIQUE_NAME( CATCH2_INTERNAL_TEMPLATE_TEST_ ), INTERNAL_CATCH_UNIQUE_NAME( CATCH2_INTERNAL_TEMPLATE_TEST_ ), Name, Tags, typename TestType, __VA_ARGS__ )
#else
#define INTERNAL_CATCH_TEMPLATE_TEST_CASE_NO_REGISTRATION(Name, Tags, ...) \
INTERNAL_CATCH_EXPAND_VARGS( INTERNAL_CATCH_TEMPLATE_TEST_CASE_NO_REGISTRATION_2( INTERNAL_CATCH_UNIQUE_NAME( CATCH2_INTERNAL_TEMPLATE_TEST_ ), INTERNAL_CATCH_UNIQUE_NAME( CATCH2_INTERNAL_TEMPLATE_TEST_ ), Name, Tags, typename TestType, __VA_ARGS__ ) )
#endif

#ifndef CATCH_CONFIG_TRADITIONAL_MSVC_PREPROCESSOR
#define INTERNAL_CATCH_TEMPLATE_TEST_CASE_SIG_NO_REGISTRATION(Name, Tags, Signature, ...) \
INTERNAL_CATCH_TEMPLATE_TEST_CASE_NO_REGISTRATION_2( INTERNAL_CATCH_UNIQUE_NAME( CATCH2_INTERNAL_TEMPLATE_TEST_ ), INTERNAL_CATCH_UNIQUE_NAME( CATCH2_INTERNAL_TEMPLATE_TEST_ ), Name, Tags, Signature, __VA_ARGS__ )
#else
#define INTERNAL_CATCH_TEMPLATE_TEST_CASE_SIG_NO_REGISTRATION(Name, Tags, Signature, ...) \
INTERNAL_CATCH_EXPAND_VARGS( INTERNAL_CATCH_TEMPLATE_TEST_CASE_NO_REGISTRATION_2( INTERNAL_CATCH_UNIQUE_NAME( CATCH2_INTERNAL_TEMPLATE_TEST_ ), INTERNAL_CATCH_UNIQUE_NAME( CATCH2_INTERNAL_TEMPLATE_TEST_ ), Name, Tags, Signature, __VA_ARGS__ ) )
#endif

#ifndef CATCH_CONFIG_TRADITIONAL_MSVC_PREPROCESSOR
#define INTERNAL_CATCH_TEMPLATE_TEST_CASE_METHOD_NO_REGISTRATION( ClassName, Name, Tags,... ) \
INTERNAL_CATCH_TEMPLATE_TEST_CASE_METHOD_NO_REGISTRATION_2( INTERNAL_CATCH_UNIQUE_NAME( CATCH2_INTERNAL_TEMPLATE_TEST_CLASS_ ), INTERNAL_CATCH_UNIQUE_NAME( CATCH2_INTERNAL_TEMPLATE_TEST_ ) , ClassName, Name, Tags, typename T, __VA_ARGS__ )
#else
#define INTERNAL_CATCH_TEMPLATE_TEST_CASE_METHOD_NO_REGISTRATION( ClassName, Name, Tags,... ) \
INTERNAL_CATCH_EXPAND_VARGS( INTERNAL_CATCH_TEMPLATE_TEST_CASE_METHOD_NO_REGISTRATION_2( INTERNAL_CATCH_UNIQUE_NAME( CATCH2_INTERNAL_TEMPLATE_TEST_CLASS_ ), INTERNAL_CATCH_UNIQUE_NAME( CATCH2_INTERNAL_TEMPLATE_TEST_ ) , ClassName, Name, Tags, typename T, __VA_ARGS__ ) )
#endif

#ifndef CATCH_CONFIG_TRADITIONAL_MSVC_PREPROCESSOR
#define INTERNAL_CATCH_TEMPLATE_TEST_CASE_METHOD_SIG_NO_REGISTRATION( ClassName, Name, Tags, Signature, ... ) \
INTERNAL_CATCH_TEMPLATE_TEST_CASE_METHOD_NO_REGISTRATION_2( INTERNAL_CATCH_UNIQUE_NAME( CATCH2_INTERNAL_TEMPLATE_TEST_CLASS_ ), INTERNAL_CATCH_UNIQUE_NAME( CATCH2_INTERNAL_TEMPLATE_TEST_ ) , ClassName, Name, Tags, Signature, __VA_ARGS__ )
#else
#define INTERNAL_CATCH_TEMPLATE_TEST_CASE_METHOD_SIG_NO_REGISTRATION( ClassName, Name, Tags, Signature, ... ) \
INTERNAL_CATCH_EXPAND_VARGS( INTERNAL_CATCH_TEMPLATE_TEST_CASE_METHOD_NO_REGISTRATION_2( INTERNAL_CATCH_UNIQUE_NAME( CATCH2_INTERNAL_TEMPLATE_TEST_CLASS_ ), INTERNAL_CATCH_UNIQUE_NAME( CATCH2_INTERNAL_TEMPLATE_TEST_ ) , ClassName, Name, Tags, Signature, __VA_ARGS__ ) )
#endif
#endif


#define INTERNAL_CATCH_TEMPLATE_TEST_CASE_2(TestName, TestFunc, Name, Tags, Signature, ... )\
CATCH_INTERNAL_START_WARNINGS_SUPPRESSION \
CATCH_INTERNAL_SUPPRESS_GLOBALS_WARNINGS \
CATCH_INTERNAL_SUPPRESS_ZERO_VARIADIC_WARNINGS \
CATCH_INTERNAL_SUPPRESS_UNUSED_TEMPLATE_WARNINGS \
CATCH_INTERNAL_SUPPRESS_UNUSED_VARIABLE_WARNINGS \
CATCH_INTERNAL_SUPPRESS_COMMA_WARNINGS \
INTERNAL_CATCH_DECLARE_SIG_TEST(TestFunc, INTERNAL_CATCH_REMOVE_PARENS(Signature));\
namespace {\
namespace INTERNAL_CATCH_MAKE_NAMESPACE(TestName){\
INTERNAL_CATCH_TYPE_GEN\
INTERNAL_CATCH_NTTP_GEN(INTERNAL_CATCH_REMOVE_PARENS(Signature))\
INTERNAL_CATCH_NTTP_REG_GEN(TestFunc,INTERNAL_CATCH_REMOVE_PARENS(Signature))\
template<typename...Types> \
struct TestName{\
TestName(){\
size_t index = 0;                                    \
constexpr char const* tmpl_types[] = {CATCH_REC_LIST(INTERNAL_CATCH_STRINGIZE_WITHOUT_PARENS, __VA_ARGS__)}; \
using expander = size_t[]; \
(void)expander{(reg_test(Types{}, Catch::NameAndTags{ Name " - " + std::string(tmpl_types[index]), Tags } ), index++)... }; \
}\
};\
static const int INTERNAL_CATCH_UNIQUE_NAME( globalRegistrar ) = [](){\
TestName<INTERNAL_CATCH_MAKE_TYPE_LISTS_FROM_TYPES(__VA_ARGS__)>();\
return 0;\
}();\
}\
}\
CATCH_INTERNAL_STOP_WARNINGS_SUPPRESSION \
INTERNAL_CATCH_DEFINE_SIG_TEST(TestFunc,INTERNAL_CATCH_REMOVE_PARENS(Signature))

#ifndef CATCH_CONFIG_TRADITIONAL_MSVC_PREPROCESSOR
#define INTERNAL_CATCH_TEMPLATE_TEST_CASE(Name, Tags, ...) \
INTERNAL_CATCH_TEMPLATE_TEST_CASE_2( INTERNAL_CATCH_UNIQUE_NAME( CATCH2_INTERNAL_TEMPLATE_TEST_ ), INTERNAL_CATCH_UNIQUE_NAME( CATCH2_INTERNAL_TEMPLATE_TEST_ ), Name, Tags, typename TestType, __VA_ARGS__ )
#else
#define INTERNAL_CATCH_TEMPLATE_TEST_CASE(Name, Tags, ...) \
INTERNAL_CATCH_EXPAND_VARGS( INTERNAL_CATCH_TEMPLATE_TEST_CASE_2( INTERNAL_CATCH_UNIQUE_NAME( CATCH2_INTERNAL_TEMPLATE_TEST_ ), INTERNAL_CATCH_UNIQUE_NAME( CATCH2_INTERNAL_TEMPLATE_TEST_ ), Name, Tags, typename TestType, __VA_ARGS__ ) )
#endif

#ifndef CATCH_CONFIG_TRADITIONAL_MSVC_PREPROCESSOR
#define INTERNAL_CATCH_TEMPLATE_TEST_CASE_SIG(Name, Tags, Signature, ...) \
INTERNAL_CATCH_TEMPLATE_TEST_CASE_2( INTERNAL_CATCH_UNIQUE_NAME( CATCH2_INTERNAL_TEMPLATE_TEST_ ), INTERNAL_CATCH_UNIQUE_NAME( CATCH2_INTERNAL_TEMPLATE_TEST_ ), Name, Tags, Signature, __VA_ARGS__ )
#else
#define INTERNAL_CATCH_TEMPLATE_TEST_CASE_SIG(Name, Tags, Signature, ...) \
INTERNAL_CATCH_EXPAND_VARGS( INTERNAL_CATCH_TEMPLATE_TEST_CASE_2( INTERNAL_CATCH_UNIQUE_NAME( CATCH2_INTERNAL_TEMPLATE_TEST_ ), INTERNAL_CATCH_UNIQUE_NAME( CATCH2_INTERNAL_TEMPLATE_TEST_ ), Name, Tags, Signature, __VA_ARGS__ ) )
#endif

#define INTERNAL_CATCH_TEMPLATE_PRODUCT_TEST_CASE2(TestName, TestFuncName, Name, Tags, Signature, TmplTypes, TypesList) \
CATCH_INTERNAL_START_WARNINGS_SUPPRESSION                      \
CATCH_INTERNAL_SUPPRESS_GLOBALS_WARNINGS                      \
CATCH_INTERNAL_SUPPRESS_ZERO_VARIADIC_WARNINGS                \
CATCH_INTERNAL_SUPPRESS_UNUSED_TEMPLATE_WARNINGS       \
CATCH_INTERNAL_SUPPRESS_UNUSED_VARIABLE_WARNINGS \
CATCH_INTERNAL_SUPPRESS_COMMA_WARNINGS \
template<typename TestType> static void TestFuncName();       \
namespace {\
namespace INTERNAL_CATCH_MAKE_NAMESPACE(TestName) {                                     \
INTERNAL_CATCH_TYPE_GEN                                                  \
INTERNAL_CATCH_NTTP_GEN(INTERNAL_CATCH_REMOVE_PARENS(Signature))         \
template<typename... Types>                               \
struct TestName {                                         \
void reg_tests() {                                          \
size_t index = 0;                                    \
using expander = size_t[];                           \
constexpr char const* tmpl_types[] = {CATCH_REC_LIST(INTERNAL_CATCH_STRINGIZE_WITHOUT_PARENS, INTERNAL_CATCH_REMOVE_PARENS(TmplTypes))};\
constexpr char const* types_list[] = {CATCH_REC_LIST(INTERNAL_CATCH_STRINGIZE_WITHOUT_PARENS, INTERNAL_CATCH_REMOVE_PARENS(TypesList))};\
constexpr auto num_types = sizeof(types_list) / sizeof(types_list[0]);\
(void)expander{(Catch::AutoReg( Catch::makeTestInvoker( &TestFuncName<Types> ), CATCH_INTERNAL_LINEINFO, Catch::StringRef(), Catch::NameAndTags{ Name " - " + std::string(tmpl_types[index / num_types]) + '<' + std::string(types_list[index % num_types]) + '>', Tags } ), index++)... };\
}                                                     \
};                                                        \
static int INTERNAL_CATCH_UNIQUE_NAME( globalRegistrar ) = [](){ \
using TestInit = typename create<TestName, decltype(get_wrapper<INTERNAL_CATCH_REMOVE_PARENS(TmplTypes)>()), TypeList<INTERNAL_CATCH_MAKE_TYPE_LISTS_FROM_TYPES(INTERNAL_CATCH_REMOVE_PARENS(TypesList))>>::type; \
TestInit t;                                           \
t.reg_tests();                                        \
return 0;                                             \
}();                                                      \
}                                                             \
}                                                             \
CATCH_INTERNAL_STOP_WARNINGS_SUPPRESSION                       \
template<typename TestType>                                   \
static void TestFuncName()

#ifndef CATCH_CONFIG_TRADITIONAL_MSVC_PREPROCESSOR
#define INTERNAL_CATCH_TEMPLATE_PRODUCT_TEST_CASE(Name, Tags, ...)\
INTERNAL_CATCH_TEMPLATE_PRODUCT_TEST_CASE2(INTERNAL_CATCH_UNIQUE_NAME( CATCH2_INTERNAL_TEMPLATE_TEST_ ), INTERNAL_CATCH_UNIQUE_NAME( CATCH2_INTERNAL_TEMPLATE_TEST_ ), Name, Tags, typename T,__VA_ARGS__)
#else
#define INTERNAL_CATCH_TEMPLATE_PRODUCT_TEST_CASE(Name, Tags, ...)\
INTERNAL_CATCH_EXPAND_VARGS( INTERNAL_CATCH_TEMPLATE_PRODUCT_TEST_CASE2( INTERNAL_CATCH_UNIQUE_NAME( CATCH2_INTERNAL_TEMPLATE_TEST_ ), INTERNAL_CATCH_UNIQUE_NAME( CATCH2_INTERNAL_TEMPLATE_TEST_ ), Name, Tags, typename T, __VA_ARGS__ ) )
#endif

#ifndef CATCH_CONFIG_TRADITIONAL_MSVC_PREPROCESSOR
#define INTERNAL_CATCH_TEMPLATE_PRODUCT_TEST_CASE_SIG(Name, Tags, Signature, ...)\
INTERNAL_CATCH_TEMPLATE_PRODUCT_TEST_CASE2(INTERNAL_CATCH_UNIQUE_NAME( CATCH2_INTERNAL_TEMPLATE_TEST_ ), INTERNAL_CATCH_UNIQUE_NAME( CATCH2_INTERNAL_TEMPLATE_TEST_ ), Name, Tags, Signature, __VA_ARGS__)
#else
#define INTERNAL_CATCH_TEMPLATE_PRODUCT_TEST_CASE_SIG(Name, Tags, Signature, ...)\
INTERNAL_CATCH_EXPAND_VARGS( INTERNAL_CATCH_TEMPLATE_PRODUCT_TEST_CASE2( INTERNAL_CATCH_UNIQUE_NAME( CATCH2_INTERNAL_TEMPLATE_TEST_ ), INTERNAL_CATCH_UNIQUE_NAME( CATCH2_INTERNAL_TEMPLATE_TEST_ ), Name, Tags, Signature, __VA_ARGS__ ) )
#endif

#define INTERNAL_CATCH_TEMPLATE_LIST_TEST_CASE_2(TestName, TestFunc, Name, Tags, TmplList)\
CATCH_INTERNAL_START_WARNINGS_SUPPRESSION \
CATCH_INTERNAL_SUPPRESS_GLOBALS_WARNINGS \
CATCH_INTERNAL_SUPPRESS_UNUSED_TEMPLATE_WARNINGS \
CATCH_INTERNAL_SUPPRESS_UNUSED_VARIABLE_WARNINGS \
CATCH_INTERNAL_SUPPRESS_COMMA_WARNINGS \
template<typename TestType> static void TestFunc();       \
namespace {\
namespace INTERNAL_CATCH_MAKE_NAMESPACE(TestName){\
INTERNAL_CATCH_TYPE_GEN\
template<typename... Types>                               \
struct TestName {                                         \
void reg_tests() {                                          \
size_t index = 0;                                    \
using expander = size_t[];                           \
(void)expander{(Catch::AutoReg( Catch::makeTestInvoker( &TestFunc<Types> ), CATCH_INTERNAL_LINEINFO, Catch::StringRef(), Catch::NameAndTags{ Name " - " + std::string(INTERNAL_CATCH_STRINGIZE(TmplList)) + " - " + std::to_string(index), Tags } ), index++)... };\
}                                                     \
};\
static int INTERNAL_CATCH_UNIQUE_NAME( globalRegistrar ) = [](){ \
using TestInit = typename convert<TestName, TmplList>::type; \
TestInit t;                                           \
t.reg_tests();                                        \
return 0;                                             \
}();                                                      \
}}\
CATCH_INTERNAL_STOP_WARNINGS_SUPPRESSION                       \
template<typename TestType>                                   \
static void TestFunc()

#define INTERNAL_CATCH_TEMPLATE_LIST_TEST_CASE(Name, Tags, TmplList) \
INTERNAL_CATCH_TEMPLATE_LIST_TEST_CASE_2( INTERNAL_CATCH_UNIQUE_NAME( CATCH2_INTERNAL_TEMPLATE_TEST_ ), INTERNAL_CATCH_UNIQUE_NAME( CATCH2_INTERNAL_TEMPLATE_TEST_ ), Name, Tags, TmplList )


#define INTERNAL_CATCH_TEMPLATE_TEST_CASE_METHOD_2( TestNameClass, TestName, ClassName, Name, Tags, Signature, ... ) \
CATCH_INTERNAL_START_WARNINGS_SUPPRESSION \
CATCH_INTERNAL_SUPPRESS_GLOBALS_WARNINGS \
CATCH_INTERNAL_SUPPRESS_ZERO_VARIADIC_WARNINGS \
CATCH_INTERNAL_SUPPRESS_UNUSED_TEMPLATE_WARNINGS \
CATCH_INTERNAL_SUPPRESS_UNUSED_VARIABLE_WARNINGS \
namespace {\
namespace INTERNAL_CATCH_MAKE_NAMESPACE(TestName){ \
INTERNAL_CATCH_TYPE_GEN\
INTERNAL_CATCH_NTTP_GEN(INTERNAL_CATCH_REMOVE_PARENS(Signature))\
INTERNAL_CATCH_DECLARE_SIG_TEST_METHOD(TestName, ClassName, INTERNAL_CATCH_REMOVE_PARENS(Signature));\
INTERNAL_CATCH_NTTP_REG_METHOD_GEN(TestName, INTERNAL_CATCH_REMOVE_PARENS(Signature))\
template<typename...Types> \
struct TestNameClass{\
TestNameClass(){\
size_t index = 0;                                    \
constexpr char const* tmpl_types[] = {CATCH_REC_LIST(INTERNAL_CATCH_STRINGIZE_WITHOUT_PARENS, __VA_ARGS__)};\
using expander = size_t[];\
(void)expander{(reg_test(Types{}, #ClassName, Catch::NameAndTags{ Name " - " + std::string(tmpl_types[index]), Tags } ), index++)... }; \
}\
};\
static int INTERNAL_CATCH_UNIQUE_NAME( globalRegistrar ) = [](){\
TestNameClass<INTERNAL_CATCH_MAKE_TYPE_LISTS_FROM_TYPES(__VA_ARGS__)>();\
return 0;\
}();\
}\
}\
CATCH_INTERNAL_STOP_WARNINGS_SUPPRESSION \
INTERNAL_CATCH_DEFINE_SIG_TEST_METHOD(TestName, INTERNAL_CATCH_REMOVE_PARENS(Signature))

#ifndef CATCH_CONFIG_TRADITIONAL_MSVC_PREPROCESSOR
#define INTERNAL_CATCH_TEMPLATE_TEST_CASE_METHOD( ClassName, Name, Tags,... ) \
INTERNAL_CATCH_TEMPLATE_TEST_CASE_METHOD_2( INTERNAL_CATCH_UNIQUE_NAME( CATCH2_INTERNAL_TEMPLATE_TEST_CLASS_ ), INTERNAL_CATCH_UNIQUE_NAME( CATCH2_INTERNAL_TEMPLATE_TEST_ ) , ClassName, Name, Tags, typename T, __VA_ARGS__ )
#else
#define INTERNAL_CATCH_TEMPLATE_TEST_CASE_METHOD( ClassName, Name, Tags,... ) \
INTERNAL_CATCH_EXPAND_VARGS( INTERNAL_CATCH_TEMPLATE_TEST_CASE_METHOD_2( INTERNAL_CATCH_UNIQUE_NAME( CATCH2_INTERNAL_TEMPLATE_TEST_CLASS_ ), INTERNAL_CATCH_UNIQUE_NAME( CATCH2_INTERNAL_TEMPLATE_TEST_ ) , ClassName, Name, Tags, typename T, __VA_ARGS__ ) )
#endif

#ifndef CATCH_CONFIG_TRADITIONAL_MSVC_PREPROCESSOR
#define INTERNAL_CATCH_TEMPLATE_TEST_CASE_METHOD_SIG( ClassName, Name, Tags, Signature, ... ) \
INTERNAL_CATCH_TEMPLATE_TEST_CASE_METHOD_2( INTERNAL_CATCH_UNIQUE_NAME( CATCH2_INTERNAL_TEMPLATE_TEST_CLASS_ ), INTERNAL_CATCH_UNIQUE_NAME( CATCH2_INTERNAL_TEMPLATE_TEST_ ) , ClassName, Name, Tags, Signature, __VA_ARGS__ )
#else
#define INTERNAL_CATCH_TEMPLATE_TEST_CASE_METHOD_SIG( ClassName, Name, Tags, Signature, ... ) \
INTERNAL_CATCH_EXPAND_VARGS( INTERNAL_CATCH_TEMPLATE_TEST_CASE_METHOD_2( INTERNAL_CATCH_UNIQUE_NAME( CATCH2_INTERNAL_TEMPLATE_TEST_CLASS_ ), INTERNAL_CATCH_UNIQUE_NAME( CATCH2_INTERNAL_TEMPLATE_TEST_ ) , ClassName, Name, Tags, Signature, __VA_ARGS__ ) )
#endif

#define INTERNAL_CATCH_TEMPLATE_PRODUCT_TEST_CASE_METHOD_2(TestNameClass, TestName, ClassName, Name, Tags, Signature, TmplTypes, TypesList)\
CATCH_INTERNAL_START_WARNINGS_SUPPRESSION \
CATCH_INTERNAL_SUPPRESS_GLOBALS_WARNINGS \
CATCH_INTERNAL_SUPPRESS_ZERO_VARIADIC_WARNINGS \
CATCH_INTERNAL_SUPPRESS_UNUSED_TEMPLATE_WARNINGS \
CATCH_INTERNAL_SUPPRESS_UNUSED_VARIABLE_WARNINGS \
template<typename TestType> \
struct TestName : INTERNAL_CATCH_REMOVE_PARENS(ClassName <TestType>) { \
void test();\
};\
namespace {\
namespace INTERNAL_CATCH_MAKE_NAMESPACE(TestNameClass) {\
INTERNAL_CATCH_TYPE_GEN                  \
INTERNAL_CATCH_NTTP_GEN(INTERNAL_CATCH_REMOVE_PARENS(Signature))\
template<typename...Types>\
struct TestNameClass{\
void reg_tests(){\
std::size_t index = 0;\
using expander = std::size_t[];\
constexpr char const* tmpl_types[] = {CATCH_REC_LIST(INTERNAL_CATCH_STRINGIZE_WITHOUT_PARENS, INTERNAL_CATCH_REMOVE_PARENS(TmplTypes))};\
constexpr char const* types_list[] = {CATCH_REC_LIST(INTERNAL_CATCH_STRINGIZE_WITHOUT_PARENS, INTERNAL_CATCH_REMOVE_PARENS(TypesList))};\
constexpr auto num_types = sizeof(types_list) / sizeof(types_list[0]);\
(void)expander{(Catch::AutoReg( Catch::makeTestInvoker( &TestName<Types>::test ), CATCH_INTERNAL_LINEINFO, #ClassName, Catch::NameAndTags{ Name " - " + std::string(tmpl_types[index / num_types]) + '<' + std::string(types_list[index % num_types]) + '>', Tags } ), index++)... }; \
}\
};\
static int INTERNAL_CATCH_UNIQUE_NAME( globalRegistrar ) = [](){\
using TestInit = typename create<TestNameClass, decltype(get_wrapper<INTERNAL_CATCH_REMOVE_PARENS(TmplTypes)>()), TypeList<INTERNAL_CATCH_MAKE_TYPE_LISTS_FROM_TYPES(INTERNAL_CATCH_REMOVE_PARENS(TypesList))>>::type;\
TestInit t;\
t.reg_tests();\
return 0;\
}(); \
}\
}\
CATCH_INTERNAL_STOP_WARNINGS_SUPPRESSION \
template<typename TestType> \
void TestName<TestType>::test()

#ifndef CATCH_CONFIG_TRADITIONAL_MSVC_PREPROCESSOR
#define INTERNAL_CATCH_TEMPLATE_PRODUCT_TEST_CASE_METHOD( ClassName, Name, Tags, ... )\
INTERNAL_CATCH_TEMPLATE_PRODUCT_TEST_CASE_METHOD_2( INTERNAL_CATCH_UNIQUE_NAME( CATCH2_INTERNAL_TEMPLATE_TEST_ ), INTERNAL_CATCH_UNIQUE_NAME( CATCH2_INTERNAL_TEMPLATE_TEST_ ), ClassName, Name, Tags, typename T, __VA_ARGS__ )
#else
#define INTERNAL_CATCH_TEMPLATE_PRODUCT_TEST_CASE_METHOD( ClassName, Name, Tags, ... )\
INTERNAL_CATCH_EXPAND_VARGS( INTERNAL_CATCH_TEMPLATE_PRODUCT_TEST_CASE_METHOD_2( INTERNAL_CATCH_UNIQUE_NAME( CATCH2_INTERNAL_TEMPLATE_TEST_ ), INTERNAL_CATCH_UNIQUE_NAME( CATCH2_INTERNAL_TEMPLATE_TEST_ ), ClassName, Name, Tags, typename T,__VA_ARGS__ ) )
#endif

#ifndef CATCH_CONFIG_TRADITIONAL_MSVC_PREPROCESSOR
#define INTERNAL_CATCH_TEMPLATE_PRODUCT_TEST_CASE_METHOD_SIG( ClassName, Name, Tags, Signature, ... )\
INTERNAL_CATCH_TEMPLATE_PRODUCT_TEST_CASE_METHOD_2( INTERNAL_CATCH_UNIQUE_NAME( CATCH2_INTERNAL_TEMPLATE_TEST_ ), INTERNAL_CATCH_UNIQUE_NAME( CATCH2_INTERNAL_TEMPLATE_TEST_ ), ClassName, Name, Tags, Signature, __VA_ARGS__ )
#else
#define INTERNAL_CATCH_TEMPLATE_PRODUCT_TEST_CASE_METHOD_SIG( ClassName, Name, Tags, Signature, ... )\
INTERNAL_CATCH_EXPAND_VARGS( INTERNAL_CATCH_TEMPLATE_PRODUCT_TEST_CASE_METHOD_2( INTERNAL_CATCH_UNIQUE_NAME( CATCH2_INTERNAL_TEMPLATE_TEST_ ), INTERNAL_CATCH_UNIQUE_NAME( CATCH2_INTERNAL_TEMPLATE_TEST_ ), ClassName, Name, Tags, Signature,__VA_ARGS__ ) )
#endif

#define INTERNAL_CATCH_TEMPLATE_LIST_TEST_CASE_METHOD_2( TestNameClass, TestName, ClassName, Name, Tags, TmplList) \
CATCH_INTERNAL_START_WARNINGS_SUPPRESSION \
CATCH_INTERNAL_SUPPRESS_GLOBALS_WARNINGS \
CATCH_INTERNAL_SUPPRESS_UNUSED_TEMPLATE_WARNINGS \
CATCH_INTERNAL_SUPPRESS_UNUSED_VARIABLE_WARNINGS \
CATCH_INTERNAL_SUPPRESS_COMMA_WARNINGS \
template<typename TestType> \
struct TestName : INTERNAL_CATCH_REMOVE_PARENS(ClassName <TestType>) { \
void test();\
};\
namespace {\
namespace INTERNAL_CATCH_MAKE_NAMESPACE(TestName){ \
INTERNAL_CATCH_TYPE_GEN\
template<typename...Types>\
struct TestNameClass{\
void reg_tests(){\
size_t index = 0;\
using expander = size_t[];\
(void)expander{(Catch::AutoReg( Catch::makeTestInvoker( &TestName<Types>::test ), CATCH_INTERNAL_LINEINFO, #ClassName, Catch::NameAndTags{ Name " - " + std::string(INTERNAL_CATCH_STRINGIZE(TmplList)) + " - " + std::to_string(index), Tags } ), index++)... }; \
}\
};\
static int INTERNAL_CATCH_UNIQUE_NAME( globalRegistrar ) = [](){\
using TestInit = typename convert<TestNameClass, TmplList>::type;\
TestInit t;\
t.reg_tests();\
return 0;\
}(); \
}}\
CATCH_INTERNAL_STOP_WARNINGS_SUPPRESSION \
template<typename TestType> \
void TestName<TestType>::test()

#define INTERNAL_CATCH_TEMPLATE_LIST_TEST_CASE_METHOD(ClassName, Name, Tags, TmplList) \
INTERNAL_CATCH_TEMPLATE_LIST_TEST_CASE_METHOD_2( INTERNAL_CATCH_UNIQUE_NAME( CATCH2_INTERNAL_TEMPLATE_TEST_ ), INTERNAL_CATCH_UNIQUE_NAME( CATCH2_INTERNAL_TEMPLATE_TEST_ ), ClassName, Name, Tags, TmplList )


#endif 


#if defined(CATCH_CONFIG_PREFIX_ALL) && !defined(CATCH_CONFIG_DISABLE)

#ifndef CATCH_CONFIG_TRADITIONAL_MSVC_PREPROCESSOR
#define CATCH_TEMPLATE_TEST_CASE( ... ) INTERNAL_CATCH_TEMPLATE_TEST_CASE( __VA_ARGS__ )
#define CATCH_TEMPLATE_TEST_CASE_SIG( ... ) INTERNAL_CATCH_TEMPLATE_TEST_CASE_SIG( __VA_ARGS__ )
#define CATCH_TEMPLATE_TEST_CASE_METHOD( className, ... ) INTERNAL_CATCH_TEMPLATE_TEST_CASE_METHOD( className, __VA_ARGS__ )
#define CATCH_TEMPLATE_TEST_CASE_METHOD_SIG( className, ... ) INTERNAL_CATCH_TEMPLATE_TEST_CASE_METHOD_SIG( className, __VA_ARGS__ )
#define CATCH_TEMPLATE_PRODUCT_TEST_CASE( ... ) INTERNAL_CATCH_TEMPLATE_PRODUCT_TEST_CASE( __VA_ARGS__ )
#define CATCH_TEMPLATE_PRODUCT_TEST_CASE_SIG( ... ) INTERNAL_CATCH_TEMPLATE_PRODUCT_TEST_CASE_SIG( __VA_ARGS__ )
#define CATCH_TEMPLATE_PRODUCT_TEST_CASE_METHOD( className, ... ) INTERNAL_CATCH_TEMPLATE_PRODUCT_TEST_CASE_METHOD( className, __VA_ARGS__ )
#define CATCH_TEMPLATE_PRODUCT_TEST_CASE_METHOD_SIG( className, ... ) INTERNAL_CATCH_TEMPLATE_PRODUCT_TEST_CASE_METHOD_SIG( className, __VA_ARGS__ )
#define CATCH_TEMPLATE_LIST_TEST_CASE( ... ) INTERNAL_CATCH_TEMPLATE_LIST_TEST_CASE(__VA_ARGS__)
#define CATCH_TEMPLATE_LIST_TEST_CASE_METHOD( className, ... ) INTERNAL_CATCH_TEMPLATE_LIST_TEST_CASE_METHOD( className, __VA_ARGS__ )
#else
#define CATCH_TEMPLATE_TEST_CASE( ... ) INTERNAL_CATCH_EXPAND_VARGS( INTERNAL_CATCH_TEMPLATE_TEST_CASE( __VA_ARGS__ ) )
#define CATCH_TEMPLATE_TEST_CASE_SIG( ... ) INTERNAL_CATCH_EXPAND_VARGS( INTERNAL_CATCH_TEMPLATE_TEST_CASE_SIG( __VA_ARGS__ ) )
#define CATCH_TEMPLATE_TEST_CASE_METHOD( className, ... ) INTERNAL_CATCH_EXPAND_VARGS( INTERNAL_CATCH_TEMPLATE_TEST_CASE_METHOD( className, __VA_ARGS__ ) )
#define CATCH_TEMPLATE_TEST_CASE_METHOD_SIG( className, ... ) INTERNAL_CATCH_EXPAND_VARGS( INTERNAL_CATCH_TEMPLATE_TEST_CASE_METHOD_SIG( className, __VA_ARGS__ ) )
#define CATCH_TEMPLATE_PRODUCT_TEST_CASE( ... ) INTERNAL_CATCH_EXPAND_VARGS( INTERNAL_CATCH_TEMPLATE_PRODUCT_TEST_CASE( __VA_ARGS__ ) )
#define CATCH_TEMPLATE_PRODUCT_TEST_CASE_SIG( ... ) INTERNAL_CATCH_EXPAND_VARGS( INTERNAL_CATCH_TEMPLATE_PRODUCT_TEST_CASE_SIG( __VA_ARGS__ ) )
#define CATCH_TEMPLATE_PRODUCT_TEST_CASE_METHOD( className, ... ) INTERNAL_CATCH_EXPAND_VARGS( INTERNAL_CATCH_TEMPLATE_PRODUCT_TEST_CASE_METHOD( className, __VA_ARGS__ ) )
#define CATCH_TEMPLATE_PRODUCT_TEST_CASE_METHOD_SIG( className, ... ) INTERNAL_CATCH_EXPAND_VARGS( INTERNAL_CATCH_TEMPLATE_PRODUCT_TEST_CASE_METHOD_SIG( className, __VA_ARGS__ ) )
#define CATCH_TEMPLATE_LIST_TEST_CASE( ... ) INTERNAL_CATCH_EXPAND_VARGS( INTERNAL_CATCH_TEMPLATE_LIST_TEST_CASE( __VA_ARGS__ ) )
#define CATCH_TEMPLATE_LIST_TEST_CASE_METHOD( className, ... ) INTERNAL_CATCH_EXPAND_VARGS( INTERNAL_CATCH_TEMPLATE_LIST_TEST_CASE_METHOD( className, __VA_ARGS__ ) )
#endif

#elif defined(CATCH_CONFIG_PREFIX_ALL) && defined(CATCH_CONFIG_DISABLE)

#ifndef CATCH_CONFIG_TRADITIONAL_MSVC_PREPROCESSOR
#define CATCH_TEMPLATE_TEST_CASE( ... ) INTERNAL_CATCH_TEMPLATE_TEST_CASE_NO_REGISTRATION(__VA_ARGS__)
#define CATCH_TEMPLATE_TEST_CASE_SIG( ... ) INTERNAL_CATCH_TEMPLATE_TEST_CASE_SIG_NO_REGISTRATION(__VA_ARGS__)
#define CATCH_TEMPLATE_TEST_CASE_METHOD( className, ... ) INTERNAL_CATCH_TEMPLATE_TEST_CASE_METHOD_NO_REGISTRATION(className, __VA_ARGS__)
#define CATCH_TEMPLATE_TEST_CASE_METHOD_SIG( className, ... ) INTERNAL_CATCH_TEMPLATE_TEST_CASE_METHOD_SIG_NO_REGISTRATION(className, __VA_ARGS__ )
#else
#define CATCH_TEMPLATE_TEST_CASE( ... ) INTERNAL_CATCH_EXPAND_VARGS( INTERNAL_CATCH_TEMPLATE_TEST_CASE_NO_REGISTRATION(__VA_ARGS__) )
#define CATCH_TEMPLATE_TEST_CASE_SIG( ... ) INTERNAL_CATCH_EXPAND_VARGS( INTERNAL_CATCH_TEMPLATE_TEST_CASE_SIG_NO_REGISTRATION(__VA_ARGS__) )
#define CATCH_TEMPLATE_TEST_CASE_METHOD( className, ... ) INTERNAL_CATCH_EXPAND_VARGS( INTERNAL_CATCH_TEMPLATE_TEST_CASE_METHOD_NO_REGISTRATION(className, __VA_ARGS__ ) )
#define CATCH_TEMPLATE_TEST_CASE_METHOD_SIG( className, ... ) INTERNAL_CATCH_EXPAND_VARGS( INTERNAL_CATCH_TEMPLATE_TEST_CASE_METHOD_SIG_NO_REGISTRATION(className, __VA_ARGS__ ) )
#endif

#define CATCH_TEMPLATE_PRODUCT_TEST_CASE( ... ) CATCH_TEMPLATE_TEST_CASE( __VA_ARGS__ )
#define CATCH_TEMPLATE_PRODUCT_TEST_CASE_SIG( ... ) CATCH_TEMPLATE_TEST_CASE( __VA_ARGS__ )
#define CATCH_TEMPLATE_PRODUCT_TEST_CASE_METHOD( className, ... ) CATCH_TEMPLATE_TEST_CASE_METHOD( className, __VA_ARGS__ )
#define CATCH_TEMPLATE_PRODUCT_TEST_CASE_METHOD_SIG( className, ... ) CATCH_TEMPLATE_TEST_CASE_METHOD( className, __VA_ARGS__ )
#define CATCH_TEMPLATE_LIST_TEST_CASE( ... ) CATCH_TEMPLATE_TEST_CASE(__VA_ARGS__)
#define CATCH_TEMPLATE_LIST_TEST_CASE_METHOD( className, ... ) CATCH_TEMPLATE_TEST_CASE_METHOD( className, __VA_ARGS__ )

#elif !defined(CATCH_CONFIG_PREFIX_ALL) && !defined(CATCH_CONFIG_DISABLE)

#ifndef CATCH_CONFIG_TRADITIONAL_MSVC_PREPROCESSOR
#define TEMPLATE_TEST_CASE( ... ) INTERNAL_CATCH_TEMPLATE_TEST_CASE( __VA_ARGS__ )
#define TEMPLATE_TEST_CASE_SIG( ... ) INTERNAL_CATCH_TEMPLATE_TEST_CASE_SIG( __VA_ARGS__ )
#define TEMPLATE_TEST_CASE_METHOD( className, ... ) INTERNAL_CATCH_TEMPLATE_TEST_CASE_METHOD( className, __VA_ARGS__ )
#define TEMPLATE_TEST_CASE_METHOD_SIG( className, ... ) INTERNAL_CATCH_TEMPLATE_TEST_CASE_METHOD_SIG( className, __VA_ARGS__ )
#define TEMPLATE_PRODUCT_TEST_CASE( ... ) INTERNAL_CATCH_TEMPLATE_PRODUCT_TEST_CASE( __VA_ARGS__ )
#define TEMPLATE_PRODUCT_TEST_CASE_SIG( ... ) INTERNAL_CATCH_TEMPLATE_PRODUCT_TEST_CASE_SIG( __VA_ARGS__ )
#define TEMPLATE_PRODUCT_TEST_CASE_METHOD( className, ... ) INTERNAL_CATCH_TEMPLATE_PRODUCT_TEST_CASE_METHOD( className, __VA_ARGS__ )
#define TEMPLATE_PRODUCT_TEST_CASE_METHOD_SIG( className, ... ) INTERNAL_CATCH_TEMPLATE_PRODUCT_TEST_CASE_METHOD_SIG( className, __VA_ARGS__ )
#define TEMPLATE_LIST_TEST_CASE( ... ) INTERNAL_CATCH_TEMPLATE_LIST_TEST_CASE(__VA_ARGS__)
#define TEMPLATE_LIST_TEST_CASE_METHOD( className, ... ) INTERNAL_CATCH_TEMPLATE_LIST_TEST_CASE_METHOD( className, __VA_ARGS__ )
#else
#define TEMPLATE_TEST_CASE( ... ) INTERNAL_CATCH_EXPAND_VARGS( INTERNAL_CATCH_TEMPLATE_TEST_CASE( __VA_ARGS__ ) )
#define TEMPLATE_TEST_CASE_SIG( ... ) INTERNAL_CATCH_EXPAND_VARGS( INTERNAL_CATCH_TEMPLATE_TEST_CASE_SIG( __VA_ARGS__ ) )
#define TEMPLATE_TEST_CASE_METHOD( className, ... ) INTERNAL_CATCH_EXPAND_VARGS( INTERNAL_CATCH_TEMPLATE_TEST_CASE_METHOD( className, __VA_ARGS__ ) )
#define TEMPLATE_TEST_CASE_METHOD_SIG( className, ... ) INTERNAL_CATCH_EXPAND_VARGS( INTERNAL_CATCH_TEMPLATE_TEST_CASE_METHOD_SIG( className, __VA_ARGS__ ) )
#define TEMPLATE_PRODUCT_TEST_CASE( ... ) INTERNAL_CATCH_EXPAND_VARGS( INTERNAL_CATCH_TEMPLATE_PRODUCT_TEST_CASE( __VA_ARGS__ ) )
#define TEMPLATE_PRODUCT_TEST_CASE_SIG( ... ) INTERNAL_CATCH_EXPAND_VARGS( INTERNAL_CATCH_TEMPLATE_PRODUCT_TEST_CASE_SIG( __VA_ARGS__ ) )
#define TEMPLATE_PRODUCT_TEST_CASE_METHOD( className, ... ) INTERNAL_CATCH_EXPAND_VARGS( INTERNAL_CATCH_TEMPLATE_PRODUCT_TEST_CASE_METHOD( className, __VA_ARGS__ ) )
#define TEMPLATE_PRODUCT_TEST_CASE_METHOD_SIG( className, ... ) INTERNAL_CATCH_EXPAND_VARGS( INTERNAL_CATCH_TEMPLATE_PRODUCT_TEST_CASE_METHOD_SIG( className, __VA_ARGS__ ) )
#define TEMPLATE_LIST_TEST_CASE( ... ) INTERNAL_CATCH_EXPAND_VARGS( INTERNAL_CATCH_TEMPLATE_LIST_TEST_CASE( __VA_ARGS__ ) )
#define TEMPLATE_LIST_TEST_CASE_METHOD( className, ... ) INTERNAL_CATCH_EXPAND_VARGS( INTERNAL_CATCH_TEMPLATE_LIST_TEST_CASE_METHOD( className, __VA_ARGS__ ) )
#endif

#elif !defined(CATCH_CONFIG_PREFIX_ALL) && defined(CATCH_CONFIG_DISABLE)

#ifndef CATCH_CONFIG_TRADITIONAL_MSVC_PREPROCESSOR
#define TEMPLATE_TEST_CASE( ... ) INTERNAL_CATCH_TEMPLATE_TEST_CASE_NO_REGISTRATION(__VA_ARGS__)
#define TEMPLATE_TEST_CASE_SIG( ... ) INTERNAL_CATCH_TEMPLATE_TEST_CASE_SIG_NO_REGISTRATION(__VA_ARGS__)
#define TEMPLATE_TEST_CASE_METHOD( className, ... ) INTERNAL_CATCH_TEMPLATE_TEST_CASE_METHOD_NO_REGISTRATION(className, __VA_ARGS__)
#define TEMPLATE_TEST_CASE_METHOD_SIG( className, ... ) INTERNAL_CATCH_TEMPLATE_TEST_CASE_METHOD_SIG_NO_REGISTRATION(className, __VA_ARGS__ )
#else
#define TEMPLATE_TEST_CASE( ... ) INTERNAL_CATCH_EXPAND_VARGS( INTERNAL_CATCH_TEMPLATE_TEST_CASE_NO_REGISTRATION(__VA_ARGS__) )
#define TEMPLATE_TEST_CASE_SIG( ... ) INTERNAL_CATCH_EXPAND_VARGS( INTERNAL_CATCH_TEMPLATE_TEST_CASE_SIG_NO_REGISTRATION(__VA_ARGS__) )
#define TEMPLATE_TEST_CASE_METHOD( className, ... ) INTERNAL_CATCH_EXPAND_VARGS( INTERNAL_CATCH_TEMPLATE_TEST_CASE_METHOD_NO_REGISTRATION(className, __VA_ARGS__ ) )
#define TEMPLATE_TEST_CASE_METHOD_SIG( className, ... ) INTERNAL_CATCH_EXPAND_VARGS( INTERNAL_CATCH_TEMPLATE_TEST_CASE_METHOD_SIG_NO_REGISTRATION(className, __VA_ARGS__ ) )
#endif

#define TEMPLATE_PRODUCT_TEST_CASE( ... ) TEMPLATE_TEST_CASE( __VA_ARGS__ )
#define TEMPLATE_PRODUCT_TEST_CASE_SIG( ... ) TEMPLATE_TEST_CASE( __VA_ARGS__ )
#define TEMPLATE_PRODUCT_TEST_CASE_METHOD( className, ... ) TEMPLATE_TEST_CASE_METHOD( className, __VA_ARGS__ )
#define TEMPLATE_PRODUCT_TEST_CASE_METHOD_SIG( className, ... ) TEMPLATE_TEST_CASE_METHOD( className, __VA_ARGS__ )
#define TEMPLATE_LIST_TEST_CASE( ... ) TEMPLATE_TEST_CASE(__VA_ARGS__)
#define TEMPLATE_LIST_TEST_CASE_METHOD( className, ... ) TEMPLATE_TEST_CASE_METHOD( className, __VA_ARGS__ )

#endif 


#endif 


#ifndef CATCH_TEST_CASE_INFO_HPP_INCLUDED
#define CATCH_TEST_CASE_INFO_HPP_INCLUDED



#include <string>
#include <vector>

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wpadded"
#endif

namespace Catch {


struct Tag {
constexpr Tag(StringRef original_):
original(original_)
{}
StringRef original;

friend bool operator< ( Tag const& lhs, Tag const& rhs );
friend bool operator==( Tag const& lhs, Tag const& rhs );
};

class ITestInvoker;

enum class TestCaseProperties : uint8_t {
None = 0,
IsHidden = 1 << 1,
ShouldFail = 1 << 2,
MayFail = 1 << 3,
Throws = 1 << 4,
NonPortable = 1 << 5,
Benchmark = 1 << 6
};


struct TestCaseInfo : Detail::NonCopyable {

TestCaseInfo(StringRef _className,
NameAndTags const& _tags,
SourceLineInfo const& _lineInfo);

bool isHidden() const;
bool throws() const;
bool okToFail() const;
bool expectedToFail() const;

void addFilenameTag();

friend bool operator<( TestCaseInfo const& lhs,
TestCaseInfo const& rhs );


std::string tagsAsString() const;

std::string name;
StringRef className;
private:
std::string backingTags;
void internalAppendTag(StringRef tagString);
public:
std::vector<Tag> tags;
SourceLineInfo lineInfo;
TestCaseProperties properties = TestCaseProperties::None;
};


class TestCaseHandle {
TestCaseInfo* m_info;
ITestInvoker* m_invoker;
public:
TestCaseHandle(TestCaseInfo* info, ITestInvoker* invoker) :
m_info(info), m_invoker(invoker) {}

void invoke() const {
m_invoker->invoke();
}

TestCaseInfo const& getTestCaseInfo() const;
};

Detail::unique_ptr<TestCaseInfo>
makeTestCaseInfo( StringRef className,
NameAndTags const& nameAndTags,
SourceLineInfo const& lineInfo );
}

#ifdef __clang__
#pragma clang diagnostic pop
#endif

#endif 


#ifndef CATCH_TRANSLATE_EXCEPTION_HPP_INCLUDED
#define CATCH_TRANSLATE_EXCEPTION_HPP_INCLUDED



#ifndef CATCH_INTERFACES_EXCEPTION_HPP_INCLUDED
#define CATCH_INTERFACES_EXCEPTION_HPP_INCLUDED


#include <string>
#include <vector>

namespace Catch {
using exceptionTranslateFunction = std::string(*)();

class IExceptionTranslator;
using ExceptionTranslators = std::vector<Detail::unique_ptr<IExceptionTranslator const>>;

class IExceptionTranslator {
public:
virtual ~IExceptionTranslator(); 
virtual std::string translate( ExceptionTranslators::const_iterator it, ExceptionTranslators::const_iterator itEnd ) const = 0;
};

class IExceptionTranslatorRegistry {
public:
virtual ~IExceptionTranslatorRegistry(); 
virtual std::string translateActiveException() const = 0;
};

} 

#endif 

#include <exception>

namespace Catch {

class ExceptionTranslatorRegistrar {
template<typename T>
class ExceptionTranslator : public IExceptionTranslator {
public:

ExceptionTranslator( std::string(*translateFunction)( T const& ) )
: m_translateFunction( translateFunction )
{}

std::string translate( ExceptionTranslators::const_iterator it, ExceptionTranslators::const_iterator itEnd ) const override {
#if !defined(CATCH_CONFIG_DISABLE_EXCEPTIONS)
try {
if( it == itEnd )
std::rethrow_exception(std::current_exception());
else
return (*it)->translate( it+1, itEnd );
}
catch( T const& ex ) {
return m_translateFunction( ex );
}
#else
return "You should never get here!";
#endif
}

protected:
std::string(*m_translateFunction)( T const& );
};

public:
template<typename T>
ExceptionTranslatorRegistrar( std::string(*translateFunction)( T const& ) ) {
getMutableRegistryHub().registerTranslator(
Detail::make_unique<ExceptionTranslator<T>>(translateFunction)
);
}
};

} 

#define INTERNAL_CATCH_TRANSLATE_EXCEPTION2( translatorName, signature ) \
static std::string translatorName( signature ); \
CATCH_INTERNAL_START_WARNINGS_SUPPRESSION \
CATCH_INTERNAL_SUPPRESS_GLOBALS_WARNINGS \
namespace{ Catch::ExceptionTranslatorRegistrar INTERNAL_CATCH_UNIQUE_NAME( catch_internal_ExceptionRegistrar )( &translatorName ); } \
CATCH_INTERNAL_STOP_WARNINGS_SUPPRESSION \
static std::string translatorName( signature )

#define INTERNAL_CATCH_TRANSLATE_EXCEPTION( signature ) INTERNAL_CATCH_TRANSLATE_EXCEPTION2( INTERNAL_CATCH_UNIQUE_NAME( catch_internal_ExceptionTranslator ), signature )

#if defined(CATCH_CONFIG_DISABLE)
#define INTERNAL_CATCH_TRANSLATE_EXCEPTION_NO_REG( translatorName, signature) \
static std::string translatorName( signature )
#endif


#if !defined(CATCH_CONFIG_DISABLE)
#define CATCH_TRANSLATE_EXCEPTION( signature ) INTERNAL_CATCH_TRANSLATE_EXCEPTION( signature )
#else
#define CATCH_TRANSLATE_EXCEPTION( signature ) INTERNAL_CATCH_TRANSLATE_EXCEPTION_NO_REG( INTERNAL_CATCH_UNIQUE_NAME( catch_internal_ExceptionTranslator ), signature )
#endif


#endif 


#ifndef CATCH_VERSION_HPP_INCLUDED
#define CATCH_VERSION_HPP_INCLUDED

#include <iosfwd>

namespace Catch {

struct Version {
Version( Version const& ) = delete;
Version& operator=( Version const& ) = delete;
Version(    unsigned int _majorVersion,
unsigned int _minorVersion,
unsigned int _patchNumber,
char const * const _branchName,
unsigned int _buildNumber );

unsigned int const majorVersion;
unsigned int const minorVersion;
unsigned int const patchNumber;

char const * const branchName;
unsigned int const buildNumber;

friend std::ostream& operator << ( std::ostream& os, Version const& version );
};

Version const& libraryVersion();
}

#endif 


#ifndef CATCH_VERSION_MACROS_HPP_INCLUDED
#define CATCH_VERSION_MACROS_HPP_INCLUDED

#define CATCH_VERSION_MAJOR 3
#define CATCH_VERSION_MINOR 2
#define CATCH_VERSION_PATCH 1

#endif 




#ifndef CATCH_GENERATORS_ALL_HPP_INCLUDED
#define CATCH_GENERATORS_ALL_HPP_INCLUDED



#ifndef CATCH_GENERATOR_EXCEPTION_HPP_INCLUDED
#define CATCH_GENERATOR_EXCEPTION_HPP_INCLUDED

#include <exception>

namespace Catch {

class GeneratorException : public std::exception {
const char* const m_msg = "";

public:
GeneratorException(const char* msg):
m_msg(msg)
{}

const char* what() const noexcept override final;
};

} 

#endif 


#ifndef CATCH_GENERATORS_HPP_INCLUDED
#define CATCH_GENERATORS_HPP_INCLUDED



#ifndef CATCH_INTERFACES_GENERATORTRACKER_HPP_INCLUDED
#define CATCH_INTERFACES_GENERATORTRACKER_HPP_INCLUDED


#include <string>

namespace Catch {

namespace Generators {
class GeneratorUntypedBase {
mutable std::string m_stringReprCache;

std::size_t m_currentElementIndex = 0;


virtual bool next() = 0;

virtual std::string stringifyImpl() const = 0;

public:
GeneratorUntypedBase() = default;
GeneratorUntypedBase(GeneratorUntypedBase const&) = default;
GeneratorUntypedBase& operator=(GeneratorUntypedBase const&) = default;

virtual ~GeneratorUntypedBase(); 


bool countedNext();

std::size_t currentElementIndex() const { return m_currentElementIndex; }


StringRef currentElementAsString() const;
};
using GeneratorBasePtr = Catch::Detail::unique_ptr<GeneratorUntypedBase>;

} 

class IGeneratorTracker {
public:
virtual ~IGeneratorTracker(); 
virtual auto hasGenerator() const -> bool = 0;
virtual auto getGenerator() const -> Generators::GeneratorBasePtr const& = 0;
virtual void setGenerator( Generators::GeneratorBasePtr&& generator ) = 0;
};

} 

#endif 

#include <vector>
#include <tuple>

namespace Catch {

namespace Generators {

namespace Detail {

[[noreturn]]
void throw_generator_exception(char const * msg);

} 

template<typename T>
class IGenerator : public GeneratorUntypedBase {
std::string stringifyImpl() const override {
return ::Catch::Detail::stringify( get() );
}

public:
~IGenerator() override = default;
IGenerator() = default;
IGenerator(IGenerator const&) = default;
IGenerator& operator=(IGenerator const&) = default;


virtual T const& get() const = 0;
using type = T;
};

template <typename T>
using GeneratorPtr = Catch::Detail::unique_ptr<IGenerator<T>>;

template <typename T>
class GeneratorWrapper final {
GeneratorPtr<T> m_generator;
public:
GeneratorWrapper(IGenerator<T>* generator):
m_generator(generator) {}
GeneratorWrapper(GeneratorPtr<T> generator):
m_generator(CATCH_MOVE(generator)) {}

T const& get() const {
return m_generator->get();
}
bool next() {
return m_generator->countedNext();
}
};


template<typename T>
class SingleValueGenerator final : public IGenerator<T> {
T m_value;
public:
SingleValueGenerator(T const& value) :
m_value(value)
{}
SingleValueGenerator(T&& value):
m_value(CATCH_MOVE(value))
{}

T const& get() const override {
return m_value;
}
bool next() override {
return false;
}
};

template<typename T>
class FixedValuesGenerator final : public IGenerator<T> {
static_assert(!std::is_same<T, bool>::value,
"FixedValuesGenerator does not support bools because of std::vector<bool>"
"specialization, use SingleValue Generator instead.");
std::vector<T> m_values;
size_t m_idx = 0;
public:
FixedValuesGenerator( std::initializer_list<T> values ) : m_values( values ) {}

T const& get() const override {
return m_values[m_idx];
}
bool next() override {
++m_idx;
return m_idx < m_values.size();
}
};

template <typename T, typename DecayedT = std::decay_t<T>>
GeneratorWrapper<DecayedT> value( T&& value ) {
return GeneratorWrapper<DecayedT>(
Catch::Detail::make_unique<SingleValueGenerator<DecayedT>>(
CATCH_FORWARD( value ) ) );
}
template <typename T>
GeneratorWrapper<T> values(std::initializer_list<T> values) {
return GeneratorWrapper<T>(Catch::Detail::make_unique<FixedValuesGenerator<T>>(values));
}

template<typename T>
class Generators : public IGenerator<T> {
std::vector<GeneratorWrapper<T>> m_generators;
size_t m_current = 0;

void add_generator( GeneratorWrapper<T>&& generator ) {
m_generators.emplace_back( CATCH_MOVE( generator ) );
}
void add_generator( T const& val ) {
m_generators.emplace_back( value( val ) );
}
void add_generator( T&& val ) {
m_generators.emplace_back( value( CATCH_MOVE( val ) ) );
}
template <typename U>
std::enable_if_t<!std::is_same<std::decay_t<U>, T>::value>
add_generator( U&& val ) {
add_generator( T( CATCH_FORWARD( val ) ) );
}

template <typename U> void add_generators( U&& valueOrGenerator ) {
add_generator( CATCH_FORWARD( valueOrGenerator ) );
}

template <typename U, typename... Gs>
void add_generators( U&& valueOrGenerator, Gs&&... moreGenerators ) {
add_generator( CATCH_FORWARD( valueOrGenerator ) );
add_generators( CATCH_FORWARD( moreGenerators )... );
}

public:
template <typename... Gs>
Generators(Gs &&... moreGenerators) {
m_generators.reserve(sizeof...(Gs));
add_generators(CATCH_FORWARD(moreGenerators)...);
}

T const& get() const override {
return m_generators[m_current].get();
}

bool next() override {
if (m_current >= m_generators.size()) {
return false;
}
const bool current_status = m_generators[m_current].next();
if (!current_status) {
++m_current;
}
return m_current < m_generators.size();
}
};


template <typename... Ts>
GeneratorWrapper<std::tuple<std::decay_t<Ts>...>>
table( std::initializer_list<std::tuple<std::decay_t<Ts>...>> tuples ) {
return values<std::tuple<Ts...>>( tuples );
}

template <typename T>
struct as {};

template<typename T, typename... Gs>
auto makeGenerators( GeneratorWrapper<T>&& generator, Gs &&... moreGenerators ) -> Generators<T> {
return Generators<T>(CATCH_MOVE(generator), CATCH_FORWARD(moreGenerators)...);
}
template<typename T>
auto makeGenerators( GeneratorWrapper<T>&& generator ) -> Generators<T> {
return Generators<T>(CATCH_MOVE(generator));
}
template<typename T, typename... Gs>
auto makeGenerators( T&& val, Gs &&... moreGenerators ) -> Generators<std::decay_t<T>> {
return makeGenerators( value( CATCH_FORWARD( val ) ), CATCH_FORWARD( moreGenerators )... );
}
template<typename T, typename U, typename... Gs>
auto makeGenerators( as<T>, U&& val, Gs &&... moreGenerators ) -> Generators<T> {
return makeGenerators( value( T( CATCH_FORWARD( val ) ) ), CATCH_FORWARD( moreGenerators )... );
}

auto acquireGeneratorTracker( StringRef generatorName, SourceLineInfo const& lineInfo ) -> IGeneratorTracker&;

template<typename L>
auto generate( StringRef generatorName, SourceLineInfo const& lineInfo, L const& generatorExpression ) -> decltype(std::declval<decltype(generatorExpression())>().get()) {
using UnderlyingType = typename decltype(generatorExpression())::type;

IGeneratorTracker& tracker = acquireGeneratorTracker( generatorName, lineInfo );
if (!tracker.hasGenerator()) {
tracker.setGenerator(Catch::Detail::make_unique<Generators<UnderlyingType>>(generatorExpression()));
}

auto const& generator = static_cast<IGenerator<UnderlyingType> const&>( *tracker.getGenerator() );
return generator.get();
}

} 
} 

#define GENERATE( ... ) \
Catch::Generators::generate( INTERNAL_CATCH_STRINGIZE(INTERNAL_CATCH_UNIQUE_NAME(generator)), \
CATCH_INTERNAL_LINEINFO, \
[ ]{ using namespace Catch::Generators; return makeGenerators( __VA_ARGS__ ); } ) 
#define GENERATE_COPY( ... ) \
Catch::Generators::generate( INTERNAL_CATCH_STRINGIZE(INTERNAL_CATCH_UNIQUE_NAME(generator)), \
CATCH_INTERNAL_LINEINFO, \
[=]{ using namespace Catch::Generators; return makeGenerators( __VA_ARGS__ ); } ) 
#define GENERATE_REF( ... ) \
Catch::Generators::generate( INTERNAL_CATCH_STRINGIZE(INTERNAL_CATCH_UNIQUE_NAME(generator)), \
CATCH_INTERNAL_LINEINFO, \
[&]{ using namespace Catch::Generators; return makeGenerators( __VA_ARGS__ ); } ) 

#endif 


#ifndef CATCH_GENERATORS_ADAPTERS_HPP_INCLUDED
#define CATCH_GENERATORS_ADAPTERS_HPP_INCLUDED


#include <cassert>

namespace Catch {
namespace Generators {

template <typename T>
class TakeGenerator final : public IGenerator<T> {
GeneratorWrapper<T> m_generator;
size_t m_returned = 0;
size_t m_target;
public:
TakeGenerator(size_t target, GeneratorWrapper<T>&& generator):
m_generator(CATCH_MOVE(generator)),
m_target(target)
{
assert(target != 0 && "Empty generators are not allowed");
}
T const& get() const override {
return m_generator.get();
}
bool next() override {
++m_returned;
if (m_returned >= m_target) {
return false;
}

const auto success = m_generator.next();
if (!success) {
m_returned = m_target;
}
return success;
}
};

template <typename T>
GeneratorWrapper<T> take(size_t target, GeneratorWrapper<T>&& generator) {
return GeneratorWrapper<T>(Catch::Detail::make_unique<TakeGenerator<T>>(target, CATCH_MOVE(generator)));
}


template <typename T, typename Predicate>
class FilterGenerator final : public IGenerator<T> {
GeneratorWrapper<T> m_generator;
Predicate m_predicate;
public:
template <typename P = Predicate>
FilterGenerator(P&& pred, GeneratorWrapper<T>&& generator):
m_generator(CATCH_MOVE(generator)),
m_predicate(CATCH_FORWARD(pred))
{
if (!m_predicate(m_generator.get())) {
auto has_initial_value = next();
if (!has_initial_value) {
Detail::throw_generator_exception("No valid value found in filtered generator");
}
}
}

T const& get() const override {
return m_generator.get();
}

bool next() override {
bool success = m_generator.next();
if (!success) {
return false;
}
while (!m_predicate(m_generator.get()) && (success = m_generator.next()) == true);
return success;
}
};


template <typename T, typename Predicate>
GeneratorWrapper<T> filter(Predicate&& pred, GeneratorWrapper<T>&& generator) {
return GeneratorWrapper<T>(Catch::Detail::make_unique<FilterGenerator<T, Predicate>>(CATCH_FORWARD(pred), CATCH_MOVE(generator)));
}

template <typename T>
class RepeatGenerator final : public IGenerator<T> {
static_assert(!std::is_same<T, bool>::value,
"RepeatGenerator currently does not support bools"
"because of std::vector<bool> specialization");
GeneratorWrapper<T> m_generator;
mutable std::vector<T> m_returned;
size_t m_target_repeats;
size_t m_current_repeat = 0;
size_t m_repeat_index = 0;
public:
RepeatGenerator(size_t repeats, GeneratorWrapper<T>&& generator):
m_generator(CATCH_MOVE(generator)),
m_target_repeats(repeats)
{
assert(m_target_repeats > 0 && "Repeat generator must repeat at least once");
}

T const& get() const override {
if (m_current_repeat == 0) {
m_returned.push_back(m_generator.get());
return m_returned.back();
}
return m_returned[m_repeat_index];
}

bool next() override {

if (m_current_repeat == 0) {
const auto success = m_generator.next();
if (!success) {
++m_current_repeat;
}
return m_current_repeat < m_target_repeats;
}

++m_repeat_index;
if (m_repeat_index == m_returned.size()) {
m_repeat_index = 0;
++m_current_repeat;
}
return m_current_repeat < m_target_repeats;
}
};

template <typename T>
GeneratorWrapper<T> repeat(size_t repeats, GeneratorWrapper<T>&& generator) {
return GeneratorWrapper<T>(Catch::Detail::make_unique<RepeatGenerator<T>>(repeats, CATCH_MOVE(generator)));
}

template <typename T, typename U, typename Func>
class MapGenerator final : public IGenerator<T> {
GeneratorWrapper<U> m_generator;
Func m_function;
T m_cache;
public:
template <typename F2 = Func>
MapGenerator(F2&& function, GeneratorWrapper<U>&& generator) :
m_generator(CATCH_MOVE(generator)),
m_function(CATCH_FORWARD(function)),
m_cache(m_function(m_generator.get()))
{}

T const& get() const override {
return m_cache;
}
bool next() override {
const auto success = m_generator.next();
if (success) {
m_cache = m_function(m_generator.get());
}
return success;
}
};

template <typename Func, typename U, typename T = FunctionReturnType<Func, U>>
GeneratorWrapper<T> map(Func&& function, GeneratorWrapper<U>&& generator) {
return GeneratorWrapper<T>(
Catch::Detail::make_unique<MapGenerator<T, U, Func>>(CATCH_FORWARD(function), CATCH_MOVE(generator))
);
}

template <typename T, typename U, typename Func>
GeneratorWrapper<T> map(Func&& function, GeneratorWrapper<U>&& generator) {
return GeneratorWrapper<T>(
Catch::Detail::make_unique<MapGenerator<T, U, Func>>(CATCH_FORWARD(function), CATCH_MOVE(generator))
);
}

template <typename T>
class ChunkGenerator final : public IGenerator<std::vector<T>> {
std::vector<T> m_chunk;
size_t m_chunk_size;
GeneratorWrapper<T> m_generator;
bool m_used_up = false;
public:
ChunkGenerator(size_t size, GeneratorWrapper<T> generator) :
m_chunk_size(size), m_generator(CATCH_MOVE(generator))
{
m_chunk.reserve(m_chunk_size);
if (m_chunk_size != 0) {
m_chunk.push_back(m_generator.get());
for (size_t i = 1; i < m_chunk_size; ++i) {
if (!m_generator.next()) {
Detail::throw_generator_exception("Not enough values to initialize the first chunk");
}
m_chunk.push_back(m_generator.get());
}
}
}
std::vector<T> const& get() const override {
return m_chunk;
}
bool next() override {
m_chunk.clear();
for (size_t idx = 0; idx < m_chunk_size; ++idx) {
if (!m_generator.next()) {
return false;
}
m_chunk.push_back(m_generator.get());
}
return true;
}
};

template <typename T>
GeneratorWrapper<std::vector<T>> chunk(size_t size, GeneratorWrapper<T>&& generator) {
return GeneratorWrapper<std::vector<T>>(
Catch::Detail::make_unique<ChunkGenerator<T>>(size, CATCH_MOVE(generator))
);
}

} 
} 


#endif 


#ifndef CATCH_GENERATORS_RANDOM_HPP_INCLUDED
#define CATCH_GENERATORS_RANDOM_HPP_INCLUDED



#ifndef CATCH_RANDOM_NUMBER_GENERATOR_HPP_INCLUDED
#define CATCH_RANDOM_NUMBER_GENERATOR_HPP_INCLUDED

#include <cstdint>

namespace Catch {

class SimplePcg32 {
using state_type = std::uint64_t;
public:
using result_type = std::uint32_t;
static constexpr result_type (min)() {
return 0;
}
static constexpr result_type (max)() {
return static_cast<result_type>(-1);
}

SimplePcg32():SimplePcg32(0xed743cc4U) {}

explicit SimplePcg32(result_type seed_);

void seed(result_type seed_);
void discard(uint64_t skip);

result_type operator()();

private:
friend bool operator==(SimplePcg32 const& lhs, SimplePcg32 const& rhs);
friend bool operator!=(SimplePcg32 const& lhs, SimplePcg32 const& rhs);



std::uint64_t m_state;
static const std::uint64_t s_inc = (0x13ed0cc53f939476ULL << 1ULL) | 1ULL;
};

} 

#endif 

#include <random>

namespace Catch {
namespace Generators {
namespace Detail {
std::uint32_t getSeed();
}

template <typename Float>
class RandomFloatingGenerator final : public IGenerator<Float> {
Catch::SimplePcg32 m_rng;
std::uniform_real_distribution<Float> m_dist;
Float m_current_number;
public:
RandomFloatingGenerator( Float a, Float b, std::uint32_t seed ):
m_rng(seed),
m_dist(a, b) {
static_cast<void>(next());
}

Float const& get() const override {
return m_current_number;
}
bool next() override {
m_current_number = m_dist(m_rng);
return true;
}
};

template <typename Integer>
class RandomIntegerGenerator final : public IGenerator<Integer> {
Catch::SimplePcg32 m_rng;
std::uniform_int_distribution<Integer> m_dist;
Integer m_current_number;
public:
RandomIntegerGenerator( Integer a, Integer b, std::uint32_t seed ):
m_rng(seed),
m_dist(a, b) {
static_cast<void>(next());
}

Integer const& get() const override {
return m_current_number;
}
bool next() override {
m_current_number = m_dist(m_rng);
return true;
}
};

template <typename T>
std::enable_if_t<std::is_integral<T>::value, GeneratorWrapper<T>>
random(T a, T b) {
static_assert(
!std::is_same<T, char>::value &&
!std::is_same<T, int8_t>::value &&
!std::is_same<T, uint8_t>::value &&
!std::is_same<T, signed char>::value &&
!std::is_same<T, unsigned char>::value &&
!std::is_same<T, bool>::value,
"The requested type is not supported by the underlying random distributions from std" );
return GeneratorWrapper<T>(
Catch::Detail::make_unique<RandomIntegerGenerator<T>>(a, b, Detail::getSeed())
);
}

template <typename T>
std::enable_if_t<std::is_floating_point<T>::value,
GeneratorWrapper<T>>
random(T a, T b) {
return GeneratorWrapper<T>(
Catch::Detail::make_unique<RandomFloatingGenerator<T>>(a, b, Detail::getSeed())
);
}


} 
} 


#endif 


#ifndef CATCH_GENERATORS_RANGE_HPP_INCLUDED
#define CATCH_GENERATORS_RANGE_HPP_INCLUDED


#include <iterator>
#include <type_traits>

namespace Catch {
namespace Generators {


template <typename T>
class RangeGenerator final : public IGenerator<T> {
T m_current;
T m_end;
T m_step;
bool m_positive;

public:
RangeGenerator(T const& start, T const& end, T const& step):
m_current(start),
m_end(end),
m_step(step),
m_positive(m_step > T(0))
{
assert(m_current != m_end && "Range start and end cannot be equal");
assert(m_step != T(0) && "Step size cannot be zero");
assert(((m_positive && m_current <= m_end) || (!m_positive && m_current >= m_end)) && "Step moves away from end");
}

RangeGenerator(T const& start, T const& end):
RangeGenerator(start, end, (start < end) ? T(1) : T(-1))
{}

T const& get() const override {
return m_current;
}

bool next() override {
m_current += m_step;
return (m_positive) ? (m_current < m_end) : (m_current > m_end);
}
};

template <typename T>
GeneratorWrapper<T> range(T const& start, T const& end, T const& step) {
static_assert(std::is_arithmetic<T>::value && !std::is_same<T, bool>::value, "Type must be numeric");
return GeneratorWrapper<T>(Catch::Detail::make_unique<RangeGenerator<T>>(start, end, step));
}

template <typename T>
GeneratorWrapper<T> range(T const& start, T const& end) {
static_assert(std::is_integral<T>::value && !std::is_same<T, bool>::value, "Type must be an integer");
return GeneratorWrapper<T>(Catch::Detail::make_unique<RangeGenerator<T>>(start, end));
}


template <typename T>
class IteratorGenerator final : public IGenerator<T> {
static_assert(!std::is_same<T, bool>::value,
"IteratorGenerator currently does not support bools"
"because of std::vector<bool> specialization");

std::vector<T> m_elems;
size_t m_current = 0;
public:
template <typename InputIterator, typename InputSentinel>
IteratorGenerator(InputIterator first, InputSentinel last):m_elems(first, last) {
if (m_elems.empty()) {
Detail::throw_generator_exception("IteratorGenerator received no valid values");
}
}

T const& get() const override {
return m_elems[m_current];
}

bool next() override {
++m_current;
return m_current != m_elems.size();
}
};

template <typename InputIterator,
typename InputSentinel,
typename ResultType = typename std::iterator_traits<InputIterator>::value_type>
GeneratorWrapper<ResultType> from_range(InputIterator from, InputSentinel to) {
return GeneratorWrapper<ResultType>(Catch::Detail::make_unique<IteratorGenerator<ResultType>>(from, to));
}

template <typename Container,
typename ResultType = typename Container::value_type>
GeneratorWrapper<ResultType> from_range(Container const& cnt) {
return GeneratorWrapper<ResultType>(Catch::Detail::make_unique<IteratorGenerator<ResultType>>(cnt.begin(), cnt.end()));
}


} 
} 


#endif 

#endif 





#ifndef CATCH_INTERFACES_ALL_HPP_INCLUDED
#define CATCH_INTERFACES_ALL_HPP_INCLUDED



#ifndef CATCH_INTERFACES_REPORTER_FACTORY_HPP_INCLUDED
#define CATCH_INTERFACES_REPORTER_FACTORY_HPP_INCLUDED


#include <string>

namespace Catch {

struct ReporterConfig;
class IConfig;
class IEventListener;
using IEventListenerPtr = Detail::unique_ptr<IEventListener>;


class IReporterFactory {
public:
virtual ~IReporterFactory(); 

virtual IEventListenerPtr
create( ReporterConfig&& config ) const = 0;
virtual std::string getDescription() const = 0;
};
using IReporterFactoryPtr = Detail::unique_ptr<IReporterFactory>;

class EventListenerFactory {
public:
virtual ~EventListenerFactory(); 
virtual IEventListenerPtr create( IConfig const* config ) const = 0;
virtual StringRef getName() const = 0;
virtual std::string getDescription() const = 0;
};
} 

#endif 


#ifndef CATCH_INTERFACES_REPORTER_REGISTRY_HPP_INCLUDED
#define CATCH_INTERFACES_REPORTER_REGISTRY_HPP_INCLUDED



#ifndef CATCH_CASE_INSENSITIVE_COMPARISONS_HPP_INCLUDED
#define CATCH_CASE_INSENSITIVE_COMPARISONS_HPP_INCLUDED


namespace Catch {
namespace Detail {
struct CaseInsensitiveLess {
bool operator()( StringRef lhs,
StringRef rhs ) const;
};

struct CaseInsensitiveEqualTo {
bool operator()( StringRef lhs,
StringRef rhs ) const;
};

} 
} 

#endif 

#include <string>
#include <vector>
#include <map>

namespace Catch {

class IConfig;

class IEventListener;
using IEventListenerPtr = Detail::unique_ptr<IEventListener>;
class IReporterFactory;
using IReporterFactoryPtr = Detail::unique_ptr<IReporterFactory>;
struct ReporterConfig;
class EventListenerFactory;

class IReporterRegistry {
public:
using FactoryMap = std::map<std::string, IReporterFactoryPtr, Detail::CaseInsensitiveLess>;
using Listeners = std::vector<Detail::unique_ptr<EventListenerFactory>>;

virtual ~IReporterRegistry(); 
virtual IEventListenerPtr create( std::string const& name, ReporterConfig&& config ) const = 0;
virtual FactoryMap const& getFactories() const = 0;
virtual Listeners const& getListeners() const = 0;
};

} 

#endif 


#ifndef CATCH_INTERFACES_TAG_ALIAS_REGISTRY_HPP_INCLUDED
#define CATCH_INTERFACES_TAG_ALIAS_REGISTRY_HPP_INCLUDED

#include <string>

namespace Catch {

struct TagAlias;

class ITagAliasRegistry {
public:
virtual ~ITagAliasRegistry(); 
virtual TagAlias const* find( std::string const& alias ) const = 0;
virtual std::string expandAliases( std::string const& unexpandedTestSpec ) const = 0;

static ITagAliasRegistry const& get();
};

} 

#endif 

#endif 





#ifndef CATCH_CONFIG_ANDROID_LOGWRITE_HPP_INCLUDED
#define CATCH_CONFIG_ANDROID_LOGWRITE_HPP_INCLUDED


#if defined(__ANDROID__)
#    define CATCH_INTERNAL_CONFIG_ANDROID_LOGWRITE
#endif


#if defined( CATCH_INTERNAL_CONFIG_ANDROID_LOGWRITE ) && \
!defined( CATCH_CONFIG_NO_ANDROID_LOGWRITE ) &&      \
!defined( CATCH_CONFIG_ANDROID_LOGWRITE )
#    define CATCH_CONFIG_ANDROID_LOGWRITE
#endif

#endif 





#ifndef CATCH_CONFIG_UNCAUGHT_EXCEPTIONS_HPP_INCLUDED
#define CATCH_CONFIG_UNCAUGHT_EXCEPTIONS_HPP_INCLUDED

#if defined(_MSC_VER)
#  if _MSC_VER >= 1900 
#    define CATCH_INTERNAL_CONFIG_CPP17_UNCAUGHT_EXCEPTIONS
#  endif
#endif


#include <exception>

#if defined(__cpp_lib_uncaught_exceptions) \
&& !defined(CATCH_INTERNAL_CONFIG_CPP17_UNCAUGHT_EXCEPTIONS)

#  define CATCH_INTERNAL_CONFIG_CPP17_UNCAUGHT_EXCEPTIONS
#endif 


#if defined(CATCH_INTERNAL_CONFIG_CPP17_UNCAUGHT_EXCEPTIONS) \
&& !defined(CATCH_CONFIG_NO_CPP17_UNCAUGHT_EXCEPTIONS) \
&& !defined(CATCH_CONFIG_CPP17_UNCAUGHT_EXCEPTIONS)

#  define CATCH_CONFIG_CPP17_UNCAUGHT_EXCEPTIONS
#endif


#endif 


#ifndef CATCH_CONSOLE_WIDTH_HPP_INCLUDED
#define CATCH_CONSOLE_WIDTH_HPP_INCLUDED


#ifndef CATCH_CONFIG_CONSOLE_WIDTH
#define CATCH_CONFIG_CONSOLE_WIDTH 80
#endif

#endif 


#ifndef CATCH_CONTAINER_NONMEMBERS_HPP_INCLUDED
#define CATCH_CONTAINER_NONMEMBERS_HPP_INCLUDED


#include <cstddef>
#include <initializer_list>

#if defined(CATCH_CPP17_OR_GREATER) || defined(_MSC_VER)

#include <string>

#  if !defined(__cpp_lib_nonmember_container_access)
#      define CATCH_CONFIG_POLYFILL_NONMEMBER_CONTAINER_ACCESS
#  endif

#else
#define CATCH_CONFIG_POLYFILL_NONMEMBER_CONTAINER_ACCESS
#endif



namespace Catch {
namespace Detail {

#if defined(CATCH_CONFIG_POLYFILL_NONMEMBER_CONTAINER_ACCESS)
template <typename Container>
constexpr auto empty(Container const& cont) -> decltype(cont.empty()) {
return cont.empty();
}
template <typename T, std::size_t N>
constexpr bool empty(const T (&)[N]) noexcept {
(void)N;
return false;
}
template <typename T>
constexpr bool empty(std::initializer_list<T> list) noexcept {
return list.size() > 0;
}


template <typename Container>
constexpr auto size(Container const& cont) -> decltype(cont.size()) {
return cont.size();
}
template <typename T, std::size_t N>
constexpr std::size_t size(const T(&)[N]) noexcept {
return N;
}
#endif 

} 
} 



#endif 


#ifndef CATCH_DEBUG_CONSOLE_HPP_INCLUDED
#define CATCH_DEBUG_CONSOLE_HPP_INCLUDED

#include <string>

namespace Catch {
void writeToDebugConsole( std::string const& text );
}

#endif 


#ifndef CATCH_DEBUGGER_HPP_INCLUDED
#define CATCH_DEBUGGER_HPP_INCLUDED


namespace Catch {
bool isDebuggerActive();
}

#ifdef CATCH_PLATFORM_MAC

#if defined(__i386__) || defined(__x86_64__)
#define CATCH_TRAP() __asm__("int $3\n" : : ) 
#elif defined(__aarch64__)
#define CATCH_TRAP()  __asm__(".inst 0xd43e0000")
#endif

#elif defined(CATCH_PLATFORM_IPHONE)

#if defined(__i386__) || defined(__x86_64__)
#define CATCH_TRAP()  __asm__("int $3")
#elif defined(__aarch64__)
#define CATCH_TRAP()  __asm__(".inst 0xd4200000")
#elif defined(__arm__) && !defined(__thumb__)
#define CATCH_TRAP()  __asm__(".inst 0xe7f001f0")
#elif defined(__arm__) &&  defined(__thumb__)
#define CATCH_TRAP()  __asm__(".inst 0xde01")
#endif

#elif defined(CATCH_PLATFORM_LINUX)
#if defined(__GNUC__) && (defined(__i386) || defined(__x86_64))
#define CATCH_TRAP() asm volatile ("int $3") 
#else 
#include <signal.h>

#define CATCH_TRAP() raise(SIGTRAP)
#endif
#elif defined(_MSC_VER)
#define CATCH_TRAP() __debugbreak()
#elif defined(__MINGW32__)
extern "C" __declspec(dllimport) void __stdcall DebugBreak();
#define CATCH_TRAP() DebugBreak()
#endif

#ifndef CATCH_BREAK_INTO_DEBUGGER
#ifdef CATCH_TRAP
#define CATCH_BREAK_INTO_DEBUGGER() []{ if( Catch::isDebuggerActive() ) { CATCH_TRAP(); } }()
#else
#define CATCH_BREAK_INTO_DEBUGGER() []{}()
#endif
#endif

#endif 


#ifndef CATCH_ENFORCE_HPP_INCLUDED
#define CATCH_ENFORCE_HPP_INCLUDED


#include <exception>

namespace Catch {
#if !defined(CATCH_CONFIG_DISABLE_EXCEPTIONS)
template <typename Ex>
[[noreturn]]
void throw_exception(Ex const& e) {
throw e;
}
#else 
[[noreturn]]
void throw_exception(std::exception const& e);
#endif

[[noreturn]]
void throw_logic_error(std::string const& msg);
[[noreturn]]
void throw_domain_error(std::string const& msg);
[[noreturn]]
void throw_runtime_error(std::string const& msg);

} 

#define CATCH_MAKE_MSG(...) \
(Catch::ReusableStringStream() << __VA_ARGS__).str()

#define CATCH_INTERNAL_ERROR(...) \
Catch::throw_logic_error(CATCH_MAKE_MSG( CATCH_INTERNAL_LINEINFO << ": Internal Catch2 error: " << __VA_ARGS__))

#define CATCH_ERROR(...) \
Catch::throw_domain_error(CATCH_MAKE_MSG( __VA_ARGS__ ))

#define CATCH_RUNTIME_ERROR(...) \
Catch::throw_runtime_error(CATCH_MAKE_MSG( __VA_ARGS__ ))

#define CATCH_ENFORCE( condition, ... ) \
do{ if( !(condition) ) CATCH_ERROR( __VA_ARGS__ ); } while(false)


#endif 


#ifndef CATCH_ENUM_VALUES_REGISTRY_HPP_INCLUDED
#define CATCH_ENUM_VALUES_REGISTRY_HPP_INCLUDED


#include <vector>

namespace Catch {

namespace Detail {

Catch::Detail::unique_ptr<EnumInfo> makeEnumInfo( StringRef enumName, StringRef allValueNames, std::vector<int> const& values );

class EnumValuesRegistry : public IMutableEnumValuesRegistry {

std::vector<Catch::Detail::unique_ptr<EnumInfo>> m_enumInfos;

EnumInfo const& registerEnum( StringRef enumName, StringRef allEnums, std::vector<int> const& values) override;
};

std::vector<StringRef> parseEnums( StringRef enums );

} 

} 

#endif 


#ifndef CATCH_ERRNO_GUARD_HPP_INCLUDED
#define CATCH_ERRNO_GUARD_HPP_INCLUDED

namespace Catch {

class ErrnoGuard {
public:

ErrnoGuard();
~ErrnoGuard();
private:
int m_oldErrno;
};

}

#endif 


#ifndef CATCH_EXCEPTION_TRANSLATOR_REGISTRY_HPP_INCLUDED
#define CATCH_EXCEPTION_TRANSLATOR_REGISTRY_HPP_INCLUDED


#include <vector>
#include <string>

namespace Catch {

class ExceptionTranslatorRegistry : public IExceptionTranslatorRegistry {
public:
~ExceptionTranslatorRegistry() override;
void registerTranslator( Detail::unique_ptr<IExceptionTranslator>&& translator );
std::string translateActiveException() const override;
std::string tryTranslators() const;

private:
ExceptionTranslators m_translators;
};
}

#endif 


#ifndef CATCH_FATAL_CONDITION_HANDLER_HPP_INCLUDED
#define CATCH_FATAL_CONDITION_HANDLER_HPP_INCLUDED


#include <cassert>

namespace Catch {


class FatalConditionHandler {
bool m_started = false;

void engage_platform();
void disengage_platform() noexcept;
public:
FatalConditionHandler();
~FatalConditionHandler();

void engage() {
assert(!m_started && "Handler cannot be installed twice.");
m_started = true;
engage_platform();
}

void disengage() noexcept {
assert(m_started && "Handler cannot be uninstalled without being installed first");
m_started = false;
disengage_platform();
}
};

class FatalConditionHandlerGuard {
FatalConditionHandler* m_handler;
public:
FatalConditionHandlerGuard(FatalConditionHandler* handler):
m_handler(handler) {
m_handler->engage();
}
~FatalConditionHandlerGuard() {
m_handler->disengage();
}
};

} 

#endif 


#ifndef CATCH_FLOATING_POINT_HELPERS_HPP_INCLUDED
#define CATCH_FLOATING_POINT_HELPERS_HPP_INCLUDED



#ifndef CATCH_POLYFILLS_HPP_INCLUDED
#define CATCH_POLYFILLS_HPP_INCLUDED

namespace Catch {
bool isnan(float f);
bool isnan(double d);
}

#endif 

#include <cassert>
#include <cmath>
#include <cstdint>
#include <utility>
#include <limits>

namespace Catch {
namespace Detail {

uint32_t convertToBits(float f);
uint64_t convertToBits(double d);

} 



#if defined( __GNUC__ ) || defined( __clang__ )
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Wfloat-equal"
#endif


template <typename FP>
uint64_t ulpDistance( FP lhs, FP rhs ) {
assert( std::numeric_limits<FP>::is_iec559 &&
"ulpDistance assumes IEEE-754 format for floating point types" );
assert( !Catch::isnan( lhs ) &&
"Distance between NaN and number is not meaningful" );
assert( !Catch::isnan( rhs ) &&
"Distance between NaN and number is not meaningful" );

if ( lhs == rhs ) { return 0; }

static constexpr FP positive_zero{};

if ( lhs == positive_zero ) { lhs = positive_zero; }
if ( rhs == positive_zero ) { rhs = positive_zero; }

if ( std::signbit( lhs ) != std::signbit( rhs ) ) {
return ulpDistance( std::abs( lhs ), positive_zero ) +
ulpDistance( std::abs( rhs ), positive_zero );
}

uint64_t lc = Detail::convertToBits( lhs );
uint64_t rc = Detail::convertToBits( rhs );

if ( lc < rc ) {
std::swap( lc, rc );
}

return lc - rc;
}

#if defined( __GNUC__ ) || defined( __clang__ )
#    pragma GCC diagnostic pop
#endif


} 

#endif 


#ifndef CATCH_GETENV_HPP_INCLUDED
#define CATCH_GETENV_HPP_INCLUDED

namespace Catch {
namespace Detail {

char const* getEnv(char const* varName);

}
}

#endif 


#ifndef CATCH_ISTREAM_HPP_INCLUDED
#define CATCH_ISTREAM_HPP_INCLUDED


#include <iosfwd>
#include <cstddef>
#include <ostream>
#include <string>

namespace Catch {

class IStream {
public:
virtual ~IStream(); 
virtual std::ostream& stream() = 0;

virtual bool isConsole() const { return false; }
};


auto makeStream( std::string const& filename ) -> Detail::unique_ptr<IStream>;

}

#endif 


#ifndef CATCH_LEAK_DETECTOR_HPP_INCLUDED
#define CATCH_LEAK_DETECTOR_HPP_INCLUDED

namespace Catch {

struct LeakDetector {
LeakDetector();
~LeakDetector();
};

}
#endif 


#ifndef CATCH_LIST_HPP_INCLUDED
#define CATCH_LIST_HPP_INCLUDED


#include <set>
#include <string>


namespace Catch {

class IEventListener;
class Config;


struct ReporterDescription {
std::string name, description;
};
struct ListenerDescription {
StringRef name;
std::string description;
};

struct TagInfo {
void add(StringRef spelling);
std::string all() const;

std::set<StringRef> spellings;
std::size_t count = 0;
};

bool list( IEventListener& reporter, Config const& config );

} 

#endif 


#ifndef CATCH_OUTPUT_REDIRECT_HPP_INCLUDED
#define CATCH_OUTPUT_REDIRECT_HPP_INCLUDED


#include <cstdio>
#include <iosfwd>
#include <string>

namespace Catch {

class RedirectedStream {
std::ostream& m_originalStream;
std::ostream& m_redirectionStream;
std::streambuf* m_prevBuf;

public:
RedirectedStream( std::ostream& originalStream, std::ostream& redirectionStream );
~RedirectedStream();
};

class RedirectedStdOut {
ReusableStringStream m_rss;
RedirectedStream m_cout;
public:
RedirectedStdOut();
auto str() const -> std::string;
};

class RedirectedStdErr {
ReusableStringStream m_rss;
RedirectedStream m_cerr;
RedirectedStream m_clog;
public:
RedirectedStdErr();
auto str() const -> std::string;
};

class RedirectedStreams {
public:
RedirectedStreams(RedirectedStreams const&) = delete;
RedirectedStreams& operator=(RedirectedStreams const&) = delete;
RedirectedStreams(RedirectedStreams&&) = delete;
RedirectedStreams& operator=(RedirectedStreams&&) = delete;

RedirectedStreams(std::string& redirectedCout, std::string& redirectedCerr);
~RedirectedStreams();
private:
std::string& m_redirectedCout;
std::string& m_redirectedCerr;
RedirectedStdOut m_redirectedStdOut;
RedirectedStdErr m_redirectedStdErr;
};

#if defined(CATCH_CONFIG_NEW_CAPTURE)

class TempFile {
public:
TempFile(TempFile const&) = delete;
TempFile& operator=(TempFile const&) = delete;
TempFile(TempFile&&) = delete;
TempFile& operator=(TempFile&&) = delete;

TempFile();
~TempFile();

std::FILE* getFile();
std::string getContents();

private:
std::FILE* m_file = nullptr;
#if defined(_MSC_VER)
char m_buffer[L_tmpnam] = { 0 };
#endif
};


class OutputRedirect {
public:
OutputRedirect(OutputRedirect const&) = delete;
OutputRedirect& operator=(OutputRedirect const&) = delete;
OutputRedirect(OutputRedirect&&) = delete;
OutputRedirect& operator=(OutputRedirect&&) = delete;


OutputRedirect(std::string& stdout_dest, std::string& stderr_dest);
~OutputRedirect();

private:
int m_originalStdout = -1;
int m_originalStderr = -1;
TempFile m_stdoutFile;
TempFile m_stderrFile;
std::string& m_stdoutDest;
std::string& m_stderrDest;
};

#endif

} 

#endif 


#ifndef CATCH_PARSE_NUMBERS_HPP_INCLUDED
#define CATCH_PARSE_NUMBERS_HPP_INCLUDED


#include <string>

namespace Catch {


Optional<unsigned int> parseUInt(std::string const& input, int base = 10);
}

#endif 


#ifndef CATCH_REPORTER_REGISTRY_HPP_INCLUDED
#define CATCH_REPORTER_REGISTRY_HPP_INCLUDED


#include <map>

namespace Catch {

class ReporterRegistry : public IReporterRegistry {
public:

ReporterRegistry();
~ReporterRegistry() override; 

IEventListenerPtr create( std::string const& name, ReporterConfig&& config ) const override;

void registerReporter( std::string const& name, IReporterFactoryPtr factory );
void registerListener( Detail::unique_ptr<EventListenerFactory> factory );

FactoryMap const& getFactories() const override;
Listeners const& getListeners() const override;

private:
FactoryMap m_factories;
Listeners m_listeners;
};
}

#endif 


#ifndef CATCH_RUN_CONTEXT_HPP_INCLUDED
#define CATCH_RUN_CONTEXT_HPP_INCLUDED



#ifndef CATCH_TEST_CASE_TRACKER_HPP_INCLUDED
#define CATCH_TEST_CASE_TRACKER_HPP_INCLUDED


#include <string>
#include <vector>

namespace Catch {
namespace TestCaseTracking {

struct NameAndLocation {
std::string name;
SourceLineInfo location;

NameAndLocation( std::string const& _name, SourceLineInfo const& _location );
friend bool operator==(NameAndLocation const& lhs, NameAndLocation const& rhs) {
return lhs.name == rhs.name
&& lhs.location == rhs.location;
}
};

class ITracker;

using ITrackerPtr = Catch::Detail::unique_ptr<ITracker>;

class ITracker {
NameAndLocation m_nameAndLocation;

using Children = std::vector<ITrackerPtr>;

protected:
enum CycleState {
NotStarted,
Executing,
ExecutingChildren,
NeedsAnotherRun,
CompletedSuccessfully,
Failed
};

ITracker* m_parent = nullptr;
Children m_children;
CycleState m_runState = NotStarted;

public:
ITracker( NameAndLocation const& nameAndLoc, ITracker* parent ):
m_nameAndLocation( nameAndLoc ),
m_parent( parent )
{}


NameAndLocation const& nameAndLocation() const {
return m_nameAndLocation;
}
ITracker* parent() const {
return m_parent;
}

virtual ~ITracker(); 



virtual bool isComplete() const = 0;
bool isSuccessfullyCompleted() const;
bool isOpen() const;
bool hasStarted() const;

virtual void close() = 0; 
virtual void fail() = 0;
void markAsNeedingAnotherRun();

void addChild( ITrackerPtr&& child );

ITracker* findChild( NameAndLocation const& nameAndLocation );
bool hasChildren() const {
return !m_children.empty();
}


void openChild();


virtual bool isSectionTracker() const;

virtual bool isGeneratorTracker() const;
};

class TrackerContext {

enum RunState {
NotStarted,
Executing,
CompletedCycle
};

ITrackerPtr m_rootTracker;
ITracker* m_currentTracker = nullptr;
RunState m_runState = NotStarted;

public:

ITracker& startRun();
void endRun();

void startCycle();
void completeCycle();

bool completedCycle() const;
ITracker& currentTracker();
void setCurrentTracker( ITracker* tracker );
};

class TrackerBase : public ITracker {
protected:

TrackerContext& m_ctx;

public:
TrackerBase( NameAndLocation const& nameAndLocation, TrackerContext& ctx, ITracker* parent );

bool isComplete() const override;

void open();

void close() override;
void fail() override;

private:
void moveToParent();
void moveToThis();
};

class SectionTracker : public TrackerBase {
std::vector<StringRef> m_filters;
std::string m_trimmed_name;
public:
SectionTracker( NameAndLocation const& nameAndLocation, TrackerContext& ctx, ITracker* parent );

bool isSectionTracker() const override;

bool isComplete() const override;

static SectionTracker& acquire( TrackerContext& ctx, NameAndLocation const& nameAndLocation );

void tryOpen();

void addInitialFilters( std::vector<std::string> const& filters );
void addNextFilters( std::vector<StringRef> const& filters );
std::vector<StringRef> const& getFilters() const;
StringRef trimmedName() const;
};

} 

using TestCaseTracking::ITracker;
using TestCaseTracking::TrackerContext;
using TestCaseTracking::SectionTracker;

} 

#endif 

#include <string>

namespace Catch {

class IMutableContext;
class IGeneratorTracker;
class IConfig;


class RunContext : public IResultCapture {

public:
RunContext( RunContext const& ) = delete;
RunContext& operator =( RunContext const& ) = delete;

explicit RunContext( IConfig const* _config, IEventListenerPtr&& reporter );

~RunContext() override;

Totals runTest(TestCaseHandle const& testCase);

public: 

void handleExpr
(   AssertionInfo const& info,
ITransientExpression const& expr,
AssertionReaction& reaction ) override;
void handleMessage
(   AssertionInfo const& info,
ResultWas::OfType resultType,
StringRef message,
AssertionReaction& reaction ) override;
void handleUnexpectedExceptionNotThrown
(   AssertionInfo const& info,
AssertionReaction& reaction ) override;
void handleUnexpectedInflightException
(   AssertionInfo const& info,
std::string const& message,
AssertionReaction& reaction ) override;
void handleIncomplete
(   AssertionInfo const& info ) override;
void handleNonExpr
(   AssertionInfo const &info,
ResultWas::OfType resultType,
AssertionReaction &reaction ) override;

bool sectionStarted( SectionInfo const& sectionInfo, Counts& assertions ) override;

void sectionEnded( SectionEndInfo const& endInfo ) override;
void sectionEndedEarly( SectionEndInfo const& endInfo ) override;

auto acquireGeneratorTracker( StringRef generatorName, SourceLineInfo const& lineInfo ) -> IGeneratorTracker& override;

void benchmarkPreparing( StringRef name ) override;
void benchmarkStarting( BenchmarkInfo const& info ) override;
void benchmarkEnded( BenchmarkStats<> const& stats ) override;
void benchmarkFailed( StringRef error ) override;

void pushScopedMessage( MessageInfo const& message ) override;
void popScopedMessage( MessageInfo const& message ) override;

void emplaceUnscopedMessage( MessageBuilder const& builder ) override;

std::string getCurrentTestName() const override;

const AssertionResult* getLastResult() const override;

void exceptionEarlyReported() override;

void handleFatalErrorCondition( StringRef message ) override;

bool lastAssertionPassed() override;

void assertionPassed() override;

public:
bool aborting() const;

private:

void runCurrentTest( std::string& redirectedCout, std::string& redirectedCerr );
void invokeActiveTestCase();

void resetAssertionInfo();
bool testForMissingAssertions( Counts& assertions );

void assertionEnded( AssertionResult const& result );
void reportExpr
(   AssertionInfo const &info,
ResultWas::OfType resultType,
ITransientExpression const *expr,
bool negated );

void populateReaction( AssertionReaction& reaction );

private:

void handleUnfinishedSections();

TestRunInfo m_runInfo;
IMutableContext& m_context;
TestCaseHandle const* m_activeTestCase = nullptr;
ITracker* m_testCaseTracker = nullptr;
Optional<AssertionResult> m_lastResult;

IConfig const* m_config;
Totals m_totals;
IEventListenerPtr m_reporter;
std::vector<MessageInfo> m_messages;
std::vector<ScopedMessage> m_messageScopes; 
AssertionInfo m_lastAssertionInfo;
std::vector<SectionEndInfo> m_unfinishedSections;
std::vector<ITracker*> m_activeSections;
TrackerContext m_trackerContext;
FatalConditionHandler m_fatalConditionhandler;
bool m_lastAssertionPassed = false;
bool m_shouldReportUnexpected = true;
bool m_includeSuccessfulResults;
};

void seedRng(IConfig const& config);
unsigned int rngSeed();
} 

#endif 


#ifndef CATCH_SHARDING_HPP_INCLUDED
#define CATCH_SHARDING_HPP_INCLUDED


#include <cmath>
#include <algorithm>

namespace Catch {

template<typename Container>
Container createShard(Container const& container, std::size_t const shardCount, std::size_t const shardIndex) {
assert(shardCount > shardIndex);

if (shardCount == 1) {
return container;
}

const std::size_t totalTestCount = container.size();

const std::size_t shardSize = totalTestCount / shardCount;
const std::size_t leftoverTests = totalTestCount % shardCount;

const std::size_t startIndex = shardIndex * shardSize + (std::min)(shardIndex, leftoverTests);
const std::size_t endIndex = (shardIndex + 1) * shardSize + (std::min)(shardIndex + 1, leftoverTests);

auto startIterator = std::next(container.begin(), static_cast<std::ptrdiff_t>(startIndex));
auto endIterator = std::next(container.begin(), static_cast<std::ptrdiff_t>(endIndex));

return Container(startIterator, endIterator);
}

}

#endif 


#ifndef CATCH_SINGLETONS_HPP_INCLUDED
#define CATCH_SINGLETONS_HPP_INCLUDED

namespace Catch {

struct ISingleton {
virtual ~ISingleton(); 
};


void addSingleton( ISingleton* singleton );
void cleanupSingletons();


template<typename SingletonImplT, typename InterfaceT = SingletonImplT, typename MutableInterfaceT = InterfaceT>
class Singleton : SingletonImplT, public ISingleton {

static auto getInternal() -> Singleton* {
static Singleton* s_instance = nullptr;
if( !s_instance ) {
s_instance = new Singleton;
addSingleton( s_instance );
}
return s_instance;
}

public:
static auto get() -> InterfaceT const& {
return *getInternal();
}
static auto getMutable() -> MutableInterfaceT& {
return *getInternal();
}
};

} 

#endif 


#ifndef CATCH_STARTUP_EXCEPTION_REGISTRY_HPP_INCLUDED
#define CATCH_STARTUP_EXCEPTION_REGISTRY_HPP_INCLUDED


#include <vector>
#include <exception>

namespace Catch {

class StartupExceptionRegistry {
#if !defined(CATCH_CONFIG_DISABLE_EXCEPTIONS)
public:
void add(std::exception_ptr const& exception) noexcept;
std::vector<std::exception_ptr> const& getExceptions() const noexcept;
private:
std::vector<std::exception_ptr> m_exceptions;
#endif
};

} 

#endif 



#ifndef CATCH_STDSTREAMS_HPP_INCLUDED
#define CATCH_STDSTREAMS_HPP_INCLUDED

#include <iosfwd>

namespace Catch {

std::ostream& cout();
std::ostream& cerr();
std::ostream& clog();

} 

#endif


#ifndef CATCH_STRING_MANIP_HPP_INCLUDED
#define CATCH_STRING_MANIP_HPP_INCLUDED


#include <string>
#include <iosfwd>
#include <vector>

namespace Catch {

bool startsWith( std::string const& s, std::string const& prefix );
bool startsWith( StringRef s, char prefix );
bool endsWith( std::string const& s, std::string const& suffix );
bool endsWith( std::string const& s, char suffix );
bool contains( std::string const& s, std::string const& infix );
void toLowerInPlace( std::string& s );
std::string toLower( std::string const& s );
char toLower( char c );
std::string trim( std::string const& str );
StringRef trim(StringRef ref);

std::vector<StringRef> splitStringRef( StringRef str, char delimiter );
bool replaceInPlace( std::string& str, std::string const& replaceThis, std::string const& withThis );


class pluralise {
std::uint64_t m_count;
StringRef m_label;

public:
constexpr pluralise(std::uint64_t count, StringRef label):
m_count(count),
m_label(label)
{}

friend std::ostream& operator << ( std::ostream& os, pluralise const& pluraliser );
};
}

#endif 


#ifndef CATCH_TAG_ALIAS_REGISTRY_HPP_INCLUDED
#define CATCH_TAG_ALIAS_REGISTRY_HPP_INCLUDED


#include <map>
#include <string>

namespace Catch {
struct SourceLineInfo;

class TagAliasRegistry : public ITagAliasRegistry {
public:
~TagAliasRegistry() override;
TagAlias const* find( std::string const& alias ) const override;
std::string expandAliases( std::string const& unexpandedTestSpec ) const override;
void add( std::string const& alias, std::string const& tag, SourceLineInfo const& lineInfo );

private:
std::map<std::string, TagAlias> m_registry;
};

} 

#endif 


#ifndef CATCH_TEST_CASE_INFO_HASHER_HPP_INCLUDED
#define CATCH_TEST_CASE_INFO_HASHER_HPP_INCLUDED

#include <cstdint>

namespace Catch {

struct TestCaseInfo;

class TestCaseInfoHasher {
public:
using hash_t = std::uint64_t;
TestCaseInfoHasher( hash_t seed );
uint32_t operator()( TestCaseInfo const& t ) const;

private:
hash_t m_seed;
};

} 

#endif 


#ifndef CATCH_TEST_CASE_REGISTRY_IMPL_HPP_INCLUDED
#define CATCH_TEST_CASE_REGISTRY_IMPL_HPP_INCLUDED


#include <vector>

namespace Catch {

class TestCaseHandle;
class IConfig;
class TestSpec;

std::vector<TestCaseHandle> sortTests( IConfig const& config, std::vector<TestCaseHandle> const& unsortedTestCases );

bool isThrowSafe( TestCaseHandle const& testCase, IConfig const& config );
bool matchTest( TestCaseHandle const& testCase, TestSpec const& testSpec, IConfig const& config );

void enforceNoDuplicateTestCases( std::vector<TestCaseHandle> const& functions );

std::vector<TestCaseHandle> filterTests( std::vector<TestCaseHandle> const& testCases, TestSpec const& testSpec, IConfig const& config );
std::vector<TestCaseHandle> const& getAllTestCasesSorted( IConfig const& config );

class TestRegistry : public ITestCaseRegistry {
public:
~TestRegistry() override = default;

void registerTest( Detail::unique_ptr<TestCaseInfo> testInfo, Detail::unique_ptr<ITestInvoker> testInvoker );

std::vector<TestCaseInfo*> const& getAllInfos() const override;
std::vector<TestCaseHandle> const& getAllTests() const override;
std::vector<TestCaseHandle> const& getAllTestsSorted( IConfig const& config ) const override;

private:
std::vector<Detail::unique_ptr<TestCaseInfo>> m_owned_test_infos;
std::vector<TestCaseInfo*> m_viewed_test_infos;

std::vector<Detail::unique_ptr<ITestInvoker>> m_invokers;
std::vector<TestCaseHandle> m_handles;
mutable TestRunOrder m_currentSortOrder = TestRunOrder::Declared;
mutable std::vector<TestCaseHandle> m_sortedFunctions;
};


class TestInvokerAsFunction final : public ITestInvoker {
using TestType = void(*)();
TestType m_testAsFunction;
public:
TestInvokerAsFunction(TestType testAsFunction) noexcept:
m_testAsFunction(testAsFunction) {}

void invoke() const override;
};



} 


#endif 


#ifndef CATCH_TEST_SPEC_PARSER_HPP_INCLUDED
#define CATCH_TEST_SPEC_PARSER_HPP_INCLUDED

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wpadded"
#endif


#include <vector>
#include <string>

namespace Catch {

class ITagAliasRegistry;

class TestSpecParser {
enum Mode{ None, Name, QuotedName, Tag, EscapedName };
Mode m_mode = None;
Mode lastMode = None;
bool m_exclusion = false;
std::size_t m_pos = 0;
std::size_t m_realPatternPos = 0;
std::string m_arg;
std::string m_substring;
std::string m_patternName;
std::vector<std::size_t> m_escapeChars;
TestSpec::Filter m_currentFilter;
TestSpec m_testSpec;
ITagAliasRegistry const* m_tagAliases = nullptr;

public:
TestSpecParser( ITagAliasRegistry const& tagAliases );

TestSpecParser& parse( std::string const& arg );
TestSpec testSpec();

private:
bool visitChar( char c );
void startNewMode( Mode mode );
bool processNoneChar( char c );
void processNameChar( char c );
bool processOtherChar( char c );
void endMode();
void escape();
bool isControlChar( char c ) const;
void saveLastMode();
void revertBackToLastMode();
void addFilter();
bool separate();

std::string preprocessPattern();
void addNamePattern();
void addTagPattern();

inline void addCharToPattern(char c) {
m_substring += c;
m_patternName += c;
m_realPatternPos++;
}

};

} 

#ifdef __clang__
#pragma clang diagnostic pop
#endif

#endif 


#ifndef CATCH_TEXTFLOW_HPP_INCLUDED
#define CATCH_TEXTFLOW_HPP_INCLUDED

#include <cassert>
#include <string>
#include <vector>

namespace Catch {
namespace TextFlow {

class Columns;


class Column {
std::string m_string;
size_t m_width = CATCH_CONFIG_CONSOLE_WIDTH - 1;
size_t m_indent = 0;
size_t m_initialIndent = std::string::npos;

public:

class const_iterator {
friend Column;
struct EndTag {};

Column const& m_column;
size_t m_lineStart = 0;
size_t m_lineLength = 0;
size_t m_parsedTo = 0;
bool m_addHyphen = false;

const_iterator( Column const& column, EndTag ):
m_column( column ), m_lineStart( m_column.m_string.size() ) {}

void calcLength();

size_t indentSize() const;

std::string addIndentAndSuffix( size_t position,
size_t length ) const;

public:
using difference_type = std::ptrdiff_t;
using value_type = std::string;
using pointer = value_type*;
using reference = value_type&;
using iterator_category = std::forward_iterator_tag;

explicit const_iterator( Column const& column );

std::string operator*() const;

const_iterator& operator++();
const_iterator operator++( int );

bool operator==( const_iterator const& other ) const {
return m_lineStart == other.m_lineStart && &m_column == &other.m_column;
}
bool operator!=( const_iterator const& other ) const {
return !operator==( other );
}
};
using iterator = const_iterator;

explicit Column( std::string const& text ): m_string( text ) {}

Column& width( size_t newWidth ) {
assert( newWidth > 0 );
m_width = newWidth;
return *this;
}
Column& indent( size_t newIndent ) {
m_indent = newIndent;
return *this;
}
Column& initialIndent( size_t newIndent ) {
m_initialIndent = newIndent;
return *this;
}

size_t width() const { return m_width; }
const_iterator begin() const { return const_iterator( *this ); }
const_iterator end() const { return { *this, const_iterator::EndTag{} }; }

friend std::ostream& operator<<( std::ostream& os,
Column const& col );

Columns operator+( Column const& other );
};

Column Spacer( size_t spaceWidth );

class Columns {
std::vector<Column> m_columns;

public:
class iterator {
friend Columns;
struct EndTag {};

std::vector<Column> const& m_columns;
std::vector<Column::const_iterator> m_iterators;
size_t m_activeIterators;

iterator( Columns const& columns, EndTag );

public:
using difference_type = std::ptrdiff_t;
using value_type = std::string;
using pointer = value_type*;
using reference = value_type&;
using iterator_category = std::forward_iterator_tag;

explicit iterator( Columns const& columns );

auto operator==( iterator const& other ) const -> bool {
return m_iterators == other.m_iterators;
}
auto operator!=( iterator const& other ) const -> bool {
return m_iterators != other.m_iterators;
}
std::string operator*() const;
iterator& operator++();
iterator operator++( int );
};
using const_iterator = iterator;

iterator begin() const { return iterator( *this ); }
iterator end() const { return { *this, iterator::EndTag() }; }

Columns& operator+=( Column const& col );
Columns operator+( Column const& col );

friend std::ostream& operator<<( std::ostream& os,
Columns const& cols );
};

} 
} 
#endif 


#ifndef CATCH_TO_STRING_HPP_INCLUDED
#define CATCH_TO_STRING_HPP_INCLUDED

#include <string>


namespace Catch {
template <typename T>
std::string to_string(T const& t) {
#if defined(CATCH_CONFIG_CPP11_TO_STRING)
return std::to_string(t);
#else
ReusableStringStream rss;
rss << t;
return rss.str();
#endif
}
} 

#endif 


#ifndef CATCH_UNCAUGHT_EXCEPTIONS_HPP_INCLUDED
#define CATCH_UNCAUGHT_EXCEPTIONS_HPP_INCLUDED

namespace Catch {
bool uncaught_exceptions();
} 

#endif 


#ifndef CATCH_XMLWRITER_HPP_INCLUDED
#define CATCH_XMLWRITER_HPP_INCLUDED


#include <iosfwd>
#include <vector>

namespace Catch {
enum class XmlFormatting {
None = 0x00,
Indent = 0x01,
Newline = 0x02,
};

XmlFormatting operator | (XmlFormatting lhs, XmlFormatting rhs);
XmlFormatting operator & (XmlFormatting lhs, XmlFormatting rhs);


class XmlEncode {
public:
enum ForWhat { ForTextNodes, ForAttributes };

XmlEncode( StringRef str, ForWhat forWhat = ForTextNodes );

void encodeTo( std::ostream& os ) const;

friend std::ostream& operator << ( std::ostream& os, XmlEncode const& xmlEncode );

private:
StringRef m_str;
ForWhat m_forWhat;
};

class XmlWriter {
public:

class ScopedElement {
public:
ScopedElement( XmlWriter* writer, XmlFormatting fmt );

ScopedElement( ScopedElement&& other ) noexcept;
ScopedElement& operator=( ScopedElement&& other ) noexcept;

~ScopedElement();

ScopedElement&
writeText( StringRef text,
XmlFormatting fmt = XmlFormatting::Newline |
XmlFormatting::Indent );

ScopedElement& writeAttribute( StringRef name,
StringRef attribute );
template <typename T,
typename = typename std::enable_if_t<
!std::is_convertible<T, StringRef>::value>>
ScopedElement& writeAttribute( StringRef name,
T const& attribute ) {
m_writer->writeAttribute( name, attribute );
return *this;
}

private:
XmlWriter* m_writer = nullptr;
XmlFormatting m_fmt;
};

XmlWriter( std::ostream& os );
~XmlWriter();

XmlWriter( XmlWriter const& ) = delete;
XmlWriter& operator=( XmlWriter const& ) = delete;

XmlWriter& startElement( std::string const& name, XmlFormatting fmt = XmlFormatting::Newline | XmlFormatting::Indent);

ScopedElement scopedElement( std::string const& name, XmlFormatting fmt = XmlFormatting::Newline | XmlFormatting::Indent);

XmlWriter& endElement(XmlFormatting fmt = XmlFormatting::Newline | XmlFormatting::Indent);

XmlWriter& writeAttribute( StringRef name, StringRef attribute );

XmlWriter& writeAttribute( StringRef name, bool attribute );

XmlWriter& writeAttribute( StringRef name, char const* attribute );

template <typename T,
typename = typename std::enable_if_t<
!std::is_convertible<T, StringRef>::value>>
XmlWriter& writeAttribute( StringRef name, T const& attribute ) {
ReusableStringStream rss;
rss << attribute;
return writeAttribute( name, rss.str() );
}

XmlWriter& writeText( StringRef text,
XmlFormatting fmt = XmlFormatting::Newline |
XmlFormatting::Indent );

XmlWriter& writeComment( StringRef text,
XmlFormatting fmt = XmlFormatting::Newline |
XmlFormatting::Indent );

void writeStylesheetRef( StringRef url );

void ensureTagClosed();

private:

void applyFormatting(XmlFormatting fmt);

void writeDeclaration();

void newlineIfNecessary();

bool m_tagIsOpen = false;
bool m_needsNewline = false;
std::vector<std::string> m_tags;
std::string m_indent;
std::ostream& m_os;
};

}

#endif 




#ifndef CATCH_MATCHERS_ALL_HPP_INCLUDED
#define CATCH_MATCHERS_ALL_HPP_INCLUDED



#ifndef CATCH_MATCHERS_HPP_INCLUDED
#define CATCH_MATCHERS_HPP_INCLUDED



#ifndef CATCH_MATCHERS_IMPL_HPP_INCLUDED
#define CATCH_MATCHERS_IMPL_HPP_INCLUDED


namespace Catch {

template<typename ArgT, typename MatcherT>
class MatchExpr : public ITransientExpression {
ArgT && m_arg;
MatcherT const& m_matcher;
public:
MatchExpr( ArgT && arg, MatcherT const& matcher )
:   ITransientExpression{ true, matcher.match( arg ) }, 
m_arg( CATCH_FORWARD(arg) ),
m_matcher( matcher )
{}

void streamReconstructedExpression( std::ostream& os ) const override {
os << Catch::Detail::stringify( m_arg )
<< ' '
<< m_matcher.toString();
}
};

namespace Matchers {
template <typename ArgT>
class MatcherBase;
}

using StringMatcher = Matchers::MatcherBase<std::string>;

void handleExceptionMatchExpr( AssertionHandler& handler, StringMatcher const& matcher );

template<typename ArgT, typename MatcherT>
auto makeMatchExpr( ArgT && arg, MatcherT const& matcher ) -> MatchExpr<ArgT, MatcherT> {
return MatchExpr<ArgT, MatcherT>( CATCH_FORWARD(arg), matcher );
}

} 


#define INTERNAL_CHECK_THAT( macroName, matcher, resultDisposition, arg ) \
do { \
Catch::AssertionHandler catchAssertionHandler( macroName##_catch_sr, CATCH_INTERNAL_LINEINFO, CATCH_INTERNAL_STRINGIFY(arg) ", " CATCH_INTERNAL_STRINGIFY(matcher), resultDisposition ); \
INTERNAL_CATCH_TRY { \
catchAssertionHandler.handleExpr( Catch::makeMatchExpr( arg, matcher ) ); \
} INTERNAL_CATCH_CATCH( catchAssertionHandler ) \
INTERNAL_CATCH_REACT( catchAssertionHandler ) \
} while( false )


#define INTERNAL_CATCH_THROWS_MATCHES( macroName, exceptionType, resultDisposition, matcher, ... ) \
do { \
Catch::AssertionHandler catchAssertionHandler( macroName##_catch_sr, CATCH_INTERNAL_LINEINFO, CATCH_INTERNAL_STRINGIFY(__VA_ARGS__) ", " CATCH_INTERNAL_STRINGIFY(exceptionType) ", " CATCH_INTERNAL_STRINGIFY(matcher), resultDisposition ); \
if( catchAssertionHandler.allowThrows() ) \
try { \
static_cast<void>(__VA_ARGS__ ); \
catchAssertionHandler.handleUnexpectedExceptionNotThrown(); \
} \
catch( exceptionType const& ex ) { \
catchAssertionHandler.handleExpr( Catch::makeMatchExpr( ex, matcher ) ); \
} \
catch( ... ) { \
catchAssertionHandler.handleUnexpectedInflightException(); \
} \
else \
catchAssertionHandler.handleThrowingCallSkipped(); \
INTERNAL_CATCH_REACT( catchAssertionHandler ) \
} while( false )


#endif 

#include <string>
#include <vector>

namespace Catch {
namespace Matchers {

class MatcherUntypedBase {
public:
MatcherUntypedBase() = default;

MatcherUntypedBase(MatcherUntypedBase const&) = default;
MatcherUntypedBase(MatcherUntypedBase&&) = default;

MatcherUntypedBase& operator = (MatcherUntypedBase const&) = delete;
MatcherUntypedBase& operator = (MatcherUntypedBase&&) = delete;

std::string toString() const;

protected:
virtual ~MatcherUntypedBase(); 
virtual std::string describe() const = 0;
mutable std::string m_cachedToString;
};


template<typename T>
class MatcherBase : public MatcherUntypedBase {
public:
virtual bool match( T const& arg ) const = 0;
};

namespace Detail {

template<typename ArgT>
class MatchAllOf final : public MatcherBase<ArgT> {
std::vector<MatcherBase<ArgT> const*> m_matchers;

public:
MatchAllOf() = default;
MatchAllOf(MatchAllOf const&) = delete;
MatchAllOf& operator=(MatchAllOf const&) = delete;
MatchAllOf(MatchAllOf&&) = default;
MatchAllOf& operator=(MatchAllOf&&) = default;


bool match( ArgT const& arg ) const override {
for( auto matcher : m_matchers ) {
if (!matcher->match(arg))
return false;
}
return true;
}
std::string describe() const override {
std::string description;
description.reserve( 4 + m_matchers.size()*32 );
description += "( ";
bool first = true;
for( auto matcher : m_matchers ) {
if( first )
first = false;
else
description += " and ";
description += matcher->toString();
}
description += " )";
return description;
}

friend MatchAllOf operator&& (MatchAllOf&& lhs, MatcherBase<ArgT> const& rhs) {
lhs.m_matchers.push_back(&rhs);
return CATCH_MOVE(lhs);
}
friend MatchAllOf operator&& (MatcherBase<ArgT> const& lhs, MatchAllOf&& rhs) {
rhs.m_matchers.insert(rhs.m_matchers.begin(), &lhs);
return CATCH_MOVE(rhs);
}
};

template<typename ArgT>
MatchAllOf<ArgT> operator&& (MatchAllOf<ArgT> const& lhs, MatcherBase<ArgT> const& rhs) = delete;
template<typename ArgT>
MatchAllOf<ArgT> operator&& (MatcherBase<ArgT> const& lhs, MatchAllOf<ArgT> const& rhs) = delete;

template<typename ArgT>
class MatchAnyOf final : public MatcherBase<ArgT> {
std::vector<MatcherBase<ArgT> const*> m_matchers;
public:
MatchAnyOf() = default;
MatchAnyOf(MatchAnyOf const&) = delete;
MatchAnyOf& operator=(MatchAnyOf const&) = delete;
MatchAnyOf(MatchAnyOf&&) = default;
MatchAnyOf& operator=(MatchAnyOf&&) = default;

bool match( ArgT const& arg ) const override {
for( auto matcher : m_matchers ) {
if (matcher->match(arg))
return true;
}
return false;
}
std::string describe() const override {
std::string description;
description.reserve( 4 + m_matchers.size()*32 );
description += "( ";
bool first = true;
for( auto matcher : m_matchers ) {
if( first )
first = false;
else
description += " or ";
description += matcher->toString();
}
description += " )";
return description;
}

friend MatchAnyOf operator|| (MatchAnyOf&& lhs, MatcherBase<ArgT> const& rhs) {
lhs.m_matchers.push_back(&rhs);
return CATCH_MOVE(lhs);
}
friend MatchAnyOf operator|| (MatcherBase<ArgT> const& lhs, MatchAnyOf&& rhs) {
rhs.m_matchers.insert(rhs.m_matchers.begin(), &lhs);
return CATCH_MOVE(rhs);
}
};

template<typename ArgT>
MatchAnyOf<ArgT> operator|| (MatchAnyOf<ArgT> const& lhs, MatcherBase<ArgT> const& rhs) = delete;
template<typename ArgT>
MatchAnyOf<ArgT> operator|| (MatcherBase<ArgT> const& lhs, MatchAnyOf<ArgT> const& rhs) = delete;

template<typename ArgT>
class MatchNotOf final : public MatcherBase<ArgT> {
MatcherBase<ArgT> const& m_underlyingMatcher;

public:
explicit MatchNotOf( MatcherBase<ArgT> const& underlyingMatcher ):
m_underlyingMatcher( underlyingMatcher )
{}

bool match( ArgT const& arg ) const override {
return !m_underlyingMatcher.match( arg );
}

std::string describe() const override {
return "not " + m_underlyingMatcher.toString();
}
};

} 

template <typename T>
Detail::MatchAllOf<T> operator&& (MatcherBase<T> const& lhs, MatcherBase<T> const& rhs) {
return Detail::MatchAllOf<T>{} && lhs && rhs;
}
template <typename T>
Detail::MatchAnyOf<T> operator|| (MatcherBase<T> const& lhs, MatcherBase<T> const& rhs) {
return Detail::MatchAnyOf<T>{} || lhs || rhs;
}

template <typename T>
Detail::MatchNotOf<T> operator! (MatcherBase<T> const& matcher) {
return Detail::MatchNotOf<T>{ matcher };
}


} 
} 


#if defined(CATCH_CONFIG_PREFIX_ALL) && !defined(CATCH_CONFIG_DISABLE)
#define CATCH_REQUIRE_THROWS_WITH( expr, matcher ) INTERNAL_CATCH_THROWS_STR_MATCHES( "CATCH_REQUIRE_THROWS_WITH", Catch::ResultDisposition::Normal, matcher, expr )
#define CATCH_REQUIRE_THROWS_MATCHES( expr, exceptionType, matcher ) INTERNAL_CATCH_THROWS_MATCHES( "CATCH_REQUIRE_THROWS_MATCHES", exceptionType, Catch::ResultDisposition::Normal, matcher, expr )

#define CATCH_CHECK_THROWS_WITH( expr, matcher ) INTERNAL_CATCH_THROWS_STR_MATCHES( "CATCH_CHECK_THROWS_WITH", Catch::ResultDisposition::ContinueOnFailure, matcher, expr )
#define CATCH_CHECK_THROWS_MATCHES( expr, exceptionType, matcher ) INTERNAL_CATCH_THROWS_MATCHES( "CATCH_CHECK_THROWS_MATCHES", exceptionType, Catch::ResultDisposition::ContinueOnFailure, matcher, expr )

#define CATCH_CHECK_THAT( arg, matcher ) INTERNAL_CHECK_THAT( "CATCH_CHECK_THAT", matcher, Catch::ResultDisposition::ContinueOnFailure, arg )
#define CATCH_REQUIRE_THAT( arg, matcher ) INTERNAL_CHECK_THAT( "CATCH_REQUIRE_THAT", matcher, Catch::ResultDisposition::Normal, arg )

#elif defined(CATCH_CONFIG_PREFIX_ALL) && defined(CATCH_CONFIG_DISABLE)

#define CATCH_REQUIRE_THROWS_WITH( expr, matcher )                   (void)(0)
#define CATCH_REQUIRE_THROWS_MATCHES( expr, exceptionType, matcher ) (void)(0)

#define CATCH_CHECK_THROWS_WITH( expr, matcher )                     (void)(0)
#define CATCH_CHECK_THROWS_MATCHES( expr, exceptionType, matcher )   (void)(0)

#define CATCH_CHECK_THAT( arg, matcher )                             (void)(0)
#define CATCH_REQUIRE_THAT( arg, matcher )                           (void)(0)

#elif !defined(CATCH_CONFIG_PREFIX_ALL) && !defined(CATCH_CONFIG_DISABLE)

#define REQUIRE_THROWS_WITH( expr, matcher ) INTERNAL_CATCH_THROWS_STR_MATCHES( "REQUIRE_THROWS_WITH", Catch::ResultDisposition::Normal, matcher, expr )
#define REQUIRE_THROWS_MATCHES( expr, exceptionType, matcher ) INTERNAL_CATCH_THROWS_MATCHES( "REQUIRE_THROWS_MATCHES", exceptionType, Catch::ResultDisposition::Normal, matcher, expr )

#define CHECK_THROWS_WITH( expr, matcher ) INTERNAL_CATCH_THROWS_STR_MATCHES( "CHECK_THROWS_WITH", Catch::ResultDisposition::ContinueOnFailure, matcher, expr )
#define CHECK_THROWS_MATCHES( expr, exceptionType, matcher ) INTERNAL_CATCH_THROWS_MATCHES( "CHECK_THROWS_MATCHES", exceptionType, Catch::ResultDisposition::ContinueOnFailure, matcher, expr )

#define CHECK_THAT( arg, matcher ) INTERNAL_CHECK_THAT( "CHECK_THAT", matcher, Catch::ResultDisposition::ContinueOnFailure, arg )
#define REQUIRE_THAT( arg, matcher ) INTERNAL_CHECK_THAT( "REQUIRE_THAT", matcher, Catch::ResultDisposition::Normal, arg )

#elif !defined(CATCH_CONFIG_PREFIX_ALL) && defined(CATCH_CONFIG_DISABLE)

#define REQUIRE_THROWS_WITH( expr, matcher )                   (void)(0)
#define REQUIRE_THROWS_MATCHES( expr, exceptionType, matcher ) (void)(0)

#define CHECK_THROWS_WITH( expr, matcher )                     (void)(0)
#define CHECK_THROWS_MATCHES( expr, exceptionType, matcher )   (void)(0)

#define CHECK_THAT( arg, matcher )                             (void)(0)
#define REQUIRE_THAT( arg, matcher )                           (void)(0)

#endif 

#endif 


#ifndef CATCH_MATCHERS_CONTAINER_PROPERTIES_HPP_INCLUDED
#define CATCH_MATCHERS_CONTAINER_PROPERTIES_HPP_INCLUDED



#ifndef CATCH_MATCHERS_TEMPLATED_HPP_INCLUDED
#define CATCH_MATCHERS_TEMPLATED_HPP_INCLUDED


#include <array>
#include <algorithm>
#include <string>
#include <type_traits>

namespace Catch {
namespace Matchers {
class MatcherGenericBase : public MatcherUntypedBase {
public:
MatcherGenericBase() = default;
~MatcherGenericBase() override; 

MatcherGenericBase(MatcherGenericBase const&) = default;
MatcherGenericBase(MatcherGenericBase&&) = default;

MatcherGenericBase& operator=(MatcherGenericBase const&) = delete;
MatcherGenericBase& operator=(MatcherGenericBase&&) = delete;
};


namespace Detail {
template<std::size_t N, std::size_t M>
std::array<void const*, N + M> array_cat(std::array<void const*, N> && lhs, std::array<void const*, M> && rhs) {
std::array<void const*, N + M> arr{};
std::copy_n(lhs.begin(), N, arr.begin());
std::copy_n(rhs.begin(), M, arr.begin() + N);
return arr;
}

template<std::size_t N>
std::array<void const*, N+1> array_cat(std::array<void const*, N> && lhs, void const* rhs) {
std::array<void const*, N+1> arr{};
std::copy_n(lhs.begin(), N, arr.begin());
arr[N] = rhs;
return arr;
}

template<std::size_t N>
std::array<void const*, N+1> array_cat(void const* lhs, std::array<void const*, N> && rhs) {
std::array<void const*, N + 1> arr{ {lhs} };
std::copy_n(rhs.begin(), N, arr.begin() + 1);
return arr;
}

template<typename T>
using is_generic_matcher = std::is_base_of<
Catch::Matchers::MatcherGenericBase,
std::remove_cv_t<std::remove_reference_t<T>>
>;

template<typename... Ts>
using are_generic_matchers = Catch::Detail::conjunction<is_generic_matcher<Ts>...>;

template<typename T>
using is_matcher = std::is_base_of<
Catch::Matchers::MatcherUntypedBase,
std::remove_cv_t<std::remove_reference_t<T>>
>;


template<std::size_t N, typename Arg>
bool match_all_of(Arg&&, std::array<void const*, N> const&, std::index_sequence<>) {
return true;
}

template<typename T, typename... MatcherTs, std::size_t N, typename Arg, std::size_t Idx, std::size_t... Indices>
bool match_all_of(Arg&& arg, std::array<void const*, N> const& matchers, std::index_sequence<Idx, Indices...>) {
return static_cast<T const*>(matchers[Idx])->match(arg) && match_all_of<MatcherTs...>(arg, matchers, std::index_sequence<Indices...>{});
}


template<std::size_t N, typename Arg>
bool match_any_of(Arg&&, std::array<void const*, N> const&, std::index_sequence<>) {
return false;
}

template<typename T, typename... MatcherTs, std::size_t N, typename Arg, std::size_t Idx, std::size_t... Indices>
bool match_any_of(Arg&& arg, std::array<void const*, N> const& matchers, std::index_sequence<Idx, Indices...>) {
return static_cast<T const*>(matchers[Idx])->match(arg) || match_any_of<MatcherTs...>(arg, matchers, std::index_sequence<Indices...>{});
}

std::string describe_multi_matcher(StringRef combine, std::string const* descriptions_begin, std::string const* descriptions_end);

template<typename... MatcherTs, std::size_t... Idx>
std::string describe_multi_matcher(StringRef combine, std::array<void const*, sizeof...(MatcherTs)> const& matchers, std::index_sequence<Idx...>) {
std::array<std::string, sizeof...(MatcherTs)> descriptions {{
static_cast<MatcherTs const*>(matchers[Idx])->toString()...
}};

return describe_multi_matcher(combine, descriptions.data(), descriptions.data() + descriptions.size());
}


template<typename... MatcherTs>
class MatchAllOfGeneric final : public MatcherGenericBase {
public:
MatchAllOfGeneric(MatchAllOfGeneric const&) = delete;
MatchAllOfGeneric& operator=(MatchAllOfGeneric const&) = delete;
MatchAllOfGeneric(MatchAllOfGeneric&&) = default;
MatchAllOfGeneric& operator=(MatchAllOfGeneric&&) = default;

MatchAllOfGeneric(MatcherTs const&... matchers) : m_matchers{ {std::addressof(matchers)...} } {}
explicit MatchAllOfGeneric(std::array<void const*, sizeof...(MatcherTs)> matchers) : m_matchers{matchers} {}

template<typename Arg>
bool match(Arg&& arg) const {
return match_all_of<MatcherTs...>(arg, m_matchers, std::index_sequence_for<MatcherTs...>{});
}

std::string describe() const override {
return describe_multi_matcher<MatcherTs...>(" and "_sr, m_matchers, std::index_sequence_for<MatcherTs...>{});
}

std::array<void const*, sizeof...( MatcherTs )> m_matchers;


template<typename... MatchersRHS>
friend
MatchAllOfGeneric<MatcherTs..., MatchersRHS...> operator && (
MatchAllOfGeneric<MatcherTs...>&& lhs,
MatchAllOfGeneric<MatchersRHS...>&& rhs) {
return MatchAllOfGeneric<MatcherTs..., MatchersRHS...>{array_cat(CATCH_MOVE(lhs.m_matchers), CATCH_MOVE(rhs.m_matchers))};
}

template<typename MatcherRHS>
friend std::enable_if_t<is_matcher<MatcherRHS>::value,
MatchAllOfGeneric<MatcherTs..., MatcherRHS>> operator && (
MatchAllOfGeneric<MatcherTs...>&& lhs,
MatcherRHS const& rhs) {
return MatchAllOfGeneric<MatcherTs..., MatcherRHS>{array_cat(CATCH_MOVE(lhs.m_matchers), static_cast<void const*>(&rhs))};
}

template<typename MatcherLHS>
friend std::enable_if_t<is_matcher<MatcherLHS>::value,
MatchAllOfGeneric<MatcherLHS, MatcherTs...>> operator && (
MatcherLHS const& lhs,
MatchAllOfGeneric<MatcherTs...>&& rhs) {
return MatchAllOfGeneric<MatcherLHS, MatcherTs...>{array_cat(static_cast<void const*>(std::addressof(lhs)), CATCH_MOVE(rhs.m_matchers))};
}
};


template<typename... MatcherTs>
class MatchAnyOfGeneric final : public MatcherGenericBase {
public:
MatchAnyOfGeneric(MatchAnyOfGeneric const&) = delete;
MatchAnyOfGeneric& operator=(MatchAnyOfGeneric const&) = delete;
MatchAnyOfGeneric(MatchAnyOfGeneric&&) = default;
MatchAnyOfGeneric& operator=(MatchAnyOfGeneric&&) = default;

MatchAnyOfGeneric(MatcherTs const&... matchers) : m_matchers{ {std::addressof(matchers)...} } {}
explicit MatchAnyOfGeneric(std::array<void const*, sizeof...(MatcherTs)> matchers) : m_matchers{matchers} {}

template<typename Arg>
bool match(Arg&& arg) const {
return match_any_of<MatcherTs...>(arg, m_matchers, std::index_sequence_for<MatcherTs...>{});
}

std::string describe() const override {
return describe_multi_matcher<MatcherTs...>(" or "_sr, m_matchers, std::index_sequence_for<MatcherTs...>{});
}


std::array<void const*, sizeof...( MatcherTs )> m_matchers;

template<typename... MatchersRHS>
friend MatchAnyOfGeneric<MatcherTs..., MatchersRHS...> operator || (
MatchAnyOfGeneric<MatcherTs...>&& lhs,
MatchAnyOfGeneric<MatchersRHS...>&& rhs) {
return MatchAnyOfGeneric<MatcherTs..., MatchersRHS...>{array_cat(CATCH_MOVE(lhs.m_matchers), CATCH_MOVE(rhs.m_matchers))};
}

template<typename MatcherRHS>
friend std::enable_if_t<is_matcher<MatcherRHS>::value,
MatchAnyOfGeneric<MatcherTs..., MatcherRHS>> operator || (
MatchAnyOfGeneric<MatcherTs...>&& lhs,
MatcherRHS const& rhs) {
return MatchAnyOfGeneric<MatcherTs..., MatcherRHS>{array_cat(CATCH_MOVE(lhs.m_matchers), static_cast<void const*>(std::addressof(rhs)))};
}

template<typename MatcherLHS>
friend std::enable_if_t<is_matcher<MatcherLHS>::value,
MatchAnyOfGeneric<MatcherLHS, MatcherTs...>> operator || (
MatcherLHS const& lhs,
MatchAnyOfGeneric<MatcherTs...>&& rhs) {
return MatchAnyOfGeneric<MatcherLHS, MatcherTs...>{array_cat(static_cast<void const*>(std::addressof(lhs)), CATCH_MOVE(rhs.m_matchers))};
}
};


template<typename MatcherT>
class MatchNotOfGeneric final : public MatcherGenericBase {
MatcherT const& m_matcher;

public:
MatchNotOfGeneric(MatchNotOfGeneric const&) = delete;
MatchNotOfGeneric& operator=(MatchNotOfGeneric const&) = delete;
MatchNotOfGeneric(MatchNotOfGeneric&&) = default;
MatchNotOfGeneric& operator=(MatchNotOfGeneric&&) = default;

explicit MatchNotOfGeneric(MatcherT const& matcher) : m_matcher{matcher} {}

template<typename Arg>
bool match(Arg&& arg) const {
return !m_matcher.match(arg);
}

std::string describe() const override {
return "not " + m_matcher.toString();
}

friend MatcherT const& operator ! (MatchNotOfGeneric<MatcherT> const& matcher) {
return matcher.m_matcher;
}
};
} 


template<typename MatcherLHS, typename MatcherRHS>
std::enable_if_t<Detail::are_generic_matchers<MatcherLHS, MatcherRHS>::value, Detail::MatchAllOfGeneric<MatcherLHS, MatcherRHS>>
operator && (MatcherLHS const& lhs, MatcherRHS const& rhs) {
return { lhs, rhs };
}

template<typename MatcherLHS, typename MatcherRHS>
std::enable_if_t<Detail::are_generic_matchers<MatcherLHS, MatcherRHS>::value, Detail::MatchAnyOfGeneric<MatcherLHS, MatcherRHS>>
operator || (MatcherLHS const& lhs, MatcherRHS const& rhs) {
return { lhs, rhs };
}

template<typename MatcherT>
std::enable_if_t<Detail::is_generic_matcher<MatcherT>::value, Detail::MatchNotOfGeneric<MatcherT>>
operator ! (MatcherT const& matcher) {
return Detail::MatchNotOfGeneric<MatcherT>{matcher};
}


template<typename MatcherLHS, typename ArgRHS>
std::enable_if_t<Detail::is_generic_matcher<MatcherLHS>::value, Detail::MatchAllOfGeneric<MatcherLHS, MatcherBase<ArgRHS>>>
operator && (MatcherLHS const& lhs, MatcherBase<ArgRHS> const& rhs) {
return { lhs, rhs };
}

template<typename ArgLHS, typename MatcherRHS>
std::enable_if_t<Detail::is_generic_matcher<MatcherRHS>::value, Detail::MatchAllOfGeneric<MatcherBase<ArgLHS>, MatcherRHS>>
operator && (MatcherBase<ArgLHS> const& lhs, MatcherRHS const& rhs) {
return { lhs, rhs };
}

template<typename MatcherLHS, typename ArgRHS>
std::enable_if_t<Detail::is_generic_matcher<MatcherLHS>::value, Detail::MatchAnyOfGeneric<MatcherLHS, MatcherBase<ArgRHS>>>
operator || (MatcherLHS const& lhs, MatcherBase<ArgRHS> const& rhs) {
return { lhs, rhs };
}

template<typename ArgLHS, typename MatcherRHS>
std::enable_if_t<Detail::is_generic_matcher<MatcherRHS>::value, Detail::MatchAnyOfGeneric<MatcherBase<ArgLHS>, MatcherRHS>>
operator || (MatcherBase<ArgLHS> const& lhs, MatcherRHS const& rhs) {
return { lhs, rhs };
}

} 
} 

#endif 

namespace Catch {
namespace Matchers {

class IsEmptyMatcher final : public MatcherGenericBase {
public:
template <typename RangeLike>
bool match(RangeLike&& rng) const {
#if defined(CATCH_CONFIG_POLYFILL_NONMEMBER_CONTAINER_ACCESS)
using Catch::Detail::empty;
#else
using std::empty;
#endif
return empty(rng);
}

std::string describe() const override;
};

class HasSizeMatcher final : public MatcherGenericBase {
std::size_t m_target_size;
public:
explicit HasSizeMatcher(std::size_t target_size):
m_target_size(target_size)
{}

template <typename RangeLike>
bool match(RangeLike&& rng) const {
#if defined(CATCH_CONFIG_POLYFILL_NONMEMBER_CONTAINER_ACCESS)
using Catch::Detail::size;
#else
using std::size;
#endif
return size(rng) == m_target_size;
}

std::string describe() const override;
};

template <typename Matcher>
class SizeMatchesMatcher final : public MatcherGenericBase {
Matcher m_matcher;
public:
explicit SizeMatchesMatcher(Matcher m):
m_matcher(CATCH_MOVE(m))
{}

template <typename RangeLike>
bool match(RangeLike&& rng) const {
#if defined(CATCH_CONFIG_POLYFILL_NONMEMBER_CONTAINER_ACCESS)
using Catch::Detail::size;
#else
using std::size;
#endif
return m_matcher.match(size(rng));
}

std::string describe() const override {
return "size matches " + m_matcher.describe();
}
};


IsEmptyMatcher IsEmpty();
HasSizeMatcher SizeIs(std::size_t sz);
template <typename Matcher>
std::enable_if_t<Detail::is_matcher<Matcher>::value,
SizeMatchesMatcher<Matcher>> SizeIs(Matcher&& m) {
return SizeMatchesMatcher<Matcher>{CATCH_FORWARD(m)};
}

} 
} 

#endif 


#ifndef CATCH_MATCHERS_CONTAINS_HPP_INCLUDED
#define CATCH_MATCHERS_CONTAINS_HPP_INCLUDED


#include <algorithm>
#include <functional>

namespace Catch {
namespace Matchers {
template <typename T, typename Equality>
class ContainsElementMatcher final : public MatcherGenericBase {
T m_desired;
Equality m_eq;
public:
template <typename T2, typename Equality2>
ContainsElementMatcher(T2&& target, Equality2&& predicate):
m_desired(CATCH_FORWARD(target)),
m_eq(CATCH_FORWARD(predicate))
{}

std::string describe() const override {
return "contains element " + Catch::Detail::stringify(m_desired);
}

template <typename RangeLike>
bool match(RangeLike&& rng) const {
using std::begin; using std::end;

return end(rng) != std::find_if(begin(rng), end(rng),
[&](auto const& elem) {
return m_eq(elem, m_desired);
});
}
};

template <typename Matcher>
class ContainsMatcherMatcher final : public MatcherGenericBase {
Matcher m_matcher;
public:
ContainsMatcherMatcher(Matcher matcher):
m_matcher(CATCH_MOVE(matcher))
{}

template <typename RangeLike>
bool match(RangeLike&& rng) const {
for (auto&& elem : rng) {
if (m_matcher.match(elem)) {
return true;
}
}
return false;
}

std::string describe() const override {
return "contains element matching " + m_matcher.describe();
}
};


template <typename T>
std::enable_if_t<!Detail::is_matcher<T>::value,
ContainsElementMatcher<T, std::equal_to<>>> Contains(T&& elem) {
return { CATCH_FORWARD(elem), std::equal_to<>{} };
}

template <typename Matcher>
std::enable_if_t<Detail::is_matcher<Matcher>::value,
ContainsMatcherMatcher<Matcher>> Contains(Matcher&& matcher) {
return { CATCH_FORWARD(matcher) };
}


template <typename T, typename Equality>
ContainsElementMatcher<T, Equality> Contains(T&& elem, Equality&& eq) {
return { CATCH_FORWARD(elem), CATCH_FORWARD(eq) };
}

}
}

#endif 


#ifndef CATCH_MATCHERS_EXCEPTION_HPP_INCLUDED
#define CATCH_MATCHERS_EXCEPTION_HPP_INCLUDED


namespace Catch {
namespace Matchers {

class ExceptionMessageMatcher final : public MatcherBase<std::exception> {
std::string m_message;
public:

ExceptionMessageMatcher(std::string const& message):
m_message(message)
{}

bool match(std::exception const& ex) const override;

std::string describe() const override;
};

ExceptionMessageMatcher Message(std::string const& message);

} 
} 

#endif 


#ifndef CATCH_MATCHERS_FLOATING_POINT_HPP_INCLUDED
#define CATCH_MATCHERS_FLOATING_POINT_HPP_INCLUDED


namespace Catch {
namespace Matchers {

namespace Detail {
enum class FloatingPointKind : uint8_t;
}

class  WithinAbsMatcher final : public MatcherBase<double> {
public:
WithinAbsMatcher(double target, double margin);
bool match(double const& matchee) const override;
std::string describe() const override;
private:
double m_target;
double m_margin;
};

class WithinUlpsMatcher final : public MatcherBase<double> {
public:
WithinUlpsMatcher( double target,
uint64_t ulps,
Detail::FloatingPointKind baseType );
bool match(double const& matchee) const override;
std::string describe() const override;
private:
double m_target;
uint64_t m_ulps;
Detail::FloatingPointKind m_type;
};

class WithinRelMatcher final : public MatcherBase<double> {
public:
WithinRelMatcher( double target, double epsilon );
bool match(double const& matchee) const override;
std::string describe() const override;
private:
double m_target;
double m_epsilon;
};

WithinUlpsMatcher WithinULP(double target, uint64_t maxUlpDiff);
WithinUlpsMatcher WithinULP(float target, uint64_t maxUlpDiff);
WithinAbsMatcher WithinAbs(double target, double margin);

WithinRelMatcher WithinRel(double target, double eps);
WithinRelMatcher WithinRel(double target);
WithinRelMatcher WithinRel(float target, float eps);
WithinRelMatcher WithinRel(float target);

} 
} 

#endif 


#ifndef CATCH_MATCHERS_PREDICATE_HPP_INCLUDED
#define CATCH_MATCHERS_PREDICATE_HPP_INCLUDED


#include <string>

namespace Catch {
namespace Matchers {

namespace Detail {
std::string finalizeDescription(const std::string& desc);
} 

template <typename T, typename Predicate>
class PredicateMatcher final : public MatcherBase<T> {
Predicate m_predicate;
std::string m_description;
public:

PredicateMatcher(Predicate&& elem, std::string const& descr)
:m_predicate(CATCH_FORWARD(elem)),
m_description(Detail::finalizeDescription(descr))
{}

bool match( T const& item ) const override {
return m_predicate(item);
}

std::string describe() const override {
return m_description;
}
};


template<typename T, typename Pred>
PredicateMatcher<T, Pred> Predicate(Pred&& predicate, std::string const& description = "") {
static_assert(is_callable<Pred(T)>::value, "Predicate not callable with argument T");
static_assert(std::is_same<bool, FunctionReturnType<Pred, T>>::value, "Predicate does not return bool");
return PredicateMatcher<T, Pred>(CATCH_FORWARD(predicate), description);
}

} 
} 

#endif 


#ifndef CATCH_MATCHERS_QUANTIFIERS_HPP_INCLUDED
#define CATCH_MATCHERS_QUANTIFIERS_HPP_INCLUDED


namespace Catch {
namespace Matchers {
template <typename Matcher>
class AllMatchMatcher final : public MatcherGenericBase {
Matcher m_matcher;
public:
AllMatchMatcher(Matcher matcher):
m_matcher(CATCH_MOVE(matcher))
{}

std::string describe() const override {
return "all match " + m_matcher.describe();
}

template <typename RangeLike>
bool match(RangeLike&& rng) const {
for (auto&& elem : rng) {
if (!m_matcher.match(elem)) {
return false;
}
}
return true;
}
};

template <typename Matcher>
class NoneMatchMatcher final : public MatcherGenericBase {
Matcher m_matcher;
public:
NoneMatchMatcher(Matcher matcher):
m_matcher(CATCH_MOVE(matcher))
{}

std::string describe() const override {
return "none match " + m_matcher.describe();
}

template <typename RangeLike>
bool match(RangeLike&& rng) const {
for (auto&& elem : rng) {
if (m_matcher.match(elem)) {
return false;
}
}
return true;
}
};

template <typename Matcher>
class AnyMatchMatcher final : public MatcherGenericBase {
Matcher m_matcher;
public:
AnyMatchMatcher(Matcher matcher):
m_matcher(CATCH_MOVE(matcher))
{}

std::string describe() const override {
return "any match " + m_matcher.describe();
}

template <typename RangeLike>
bool match(RangeLike&& rng) const {
for (auto&& elem : rng) {
if (m_matcher.match(elem)) {
return true;
}
}
return false;
}
};

class AllTrueMatcher final : public MatcherGenericBase {
public:
std::string describe() const override;

template <typename RangeLike>
bool match(RangeLike&& rng) const {
for (auto&& elem : rng) {
if (!elem) {
return false;
}
}
return true;
}
};

class NoneTrueMatcher final : public MatcherGenericBase {
public:
std::string describe() const override;

template <typename RangeLike>
bool match(RangeLike&& rng) const {
for (auto&& elem : rng) {
if (elem) {
return false;
}
}
return true;
}
};

class AnyTrueMatcher final : public MatcherGenericBase {
public:
std::string describe() const override;

template <typename RangeLike>
bool match(RangeLike&& rng) const {
for (auto&& elem : rng) {
if (elem) {
return true;
}
}
return false;
}
};

template <typename Matcher>
AllMatchMatcher<Matcher> AllMatch(Matcher&& matcher) {
return { CATCH_FORWARD(matcher) };
}

template <typename Matcher>
NoneMatchMatcher<Matcher> NoneMatch(Matcher&& matcher) {
return { CATCH_FORWARD(matcher) };
}

template <typename Matcher>
AnyMatchMatcher<Matcher> AnyMatch(Matcher&& matcher) {
return { CATCH_FORWARD(matcher) };
}

AllTrueMatcher AllTrue();

NoneTrueMatcher NoneTrue();

AnyTrueMatcher AnyTrue();
}
}

#endif 


#ifndef CATCH_MATCHERS_STRING_HPP_INCLUDED
#define CATCH_MATCHERS_STRING_HPP_INCLUDED


#include <string>

namespace Catch {
namespace Matchers {

struct CasedString {
CasedString( std::string const& str, CaseSensitive caseSensitivity );
std::string adjustString( std::string const& str ) const;
StringRef caseSensitivitySuffix() const;

CaseSensitive m_caseSensitivity;
std::string m_str;
};

class StringMatcherBase : public MatcherBase<std::string> {
protected:
CasedString m_comparator;
StringRef m_operation;

public:
StringMatcherBase( StringRef operation,
CasedString const& comparator );
std::string describe() const override;
};

class StringEqualsMatcher final : public StringMatcherBase {
public:
StringEqualsMatcher( CasedString const& comparator );
bool match( std::string const& source ) const override;
};
class StringContainsMatcher final : public StringMatcherBase {
public:
StringContainsMatcher( CasedString const& comparator );
bool match( std::string const& source ) const override;
};
class StartsWithMatcher final : public StringMatcherBase {
public:
StartsWithMatcher( CasedString const& comparator );
bool match( std::string const& source ) const override;
};
class EndsWithMatcher final : public StringMatcherBase {
public:
EndsWithMatcher( CasedString const& comparator );
bool match( std::string const& source ) const override;
};

class RegexMatcher final : public MatcherBase<std::string> {
std::string m_regex;
CaseSensitive m_caseSensitivity;

public:
RegexMatcher( std::string regex, CaseSensitive caseSensitivity );
bool match( std::string const& matchee ) const override;
std::string describe() const override;
};

StringEqualsMatcher Equals( std::string const& str, CaseSensitive caseSensitivity = CaseSensitive::Yes );
StringContainsMatcher ContainsSubstring( std::string const& str, CaseSensitive caseSensitivity = CaseSensitive::Yes );
EndsWithMatcher EndsWith( std::string const& str, CaseSensitive caseSensitivity = CaseSensitive::Yes );
StartsWithMatcher StartsWith( std::string const& str, CaseSensitive caseSensitivity = CaseSensitive::Yes );
RegexMatcher Matches( std::string const& regex, CaseSensitive caseSensitivity = CaseSensitive::Yes );

} 
} 

#endif 


#ifndef CATCH_MATCHERS_VECTOR_HPP_INCLUDED
#define CATCH_MATCHERS_VECTOR_HPP_INCLUDED


#include <algorithm>

namespace Catch {
namespace Matchers {

template<typename T, typename Alloc>
class VectorContainsElementMatcher final : public MatcherBase<std::vector<T, Alloc>> {
T const& m_comparator;

public:
VectorContainsElementMatcher(T const& comparator):
m_comparator(comparator)
{}

bool match(std::vector<T, Alloc> const& v) const override {
for (auto const& el : v) {
if (el == m_comparator) {
return true;
}
}
return false;
}

std::string describe() const override {
return "Contains: " + ::Catch::Detail::stringify( m_comparator );
}
};

template<typename T, typename AllocComp, typename AllocMatch>
class ContainsMatcher final : public MatcherBase<std::vector<T, AllocMatch>> {
std::vector<T, AllocComp> const& m_comparator;

public:
ContainsMatcher(std::vector<T, AllocComp> const& comparator):
m_comparator( comparator )
{}

bool match(std::vector<T, AllocMatch> const& v) const override {
if (m_comparator.size() > v.size())
return false;
for (auto const& comparator : m_comparator) {
auto present = false;
for (const auto& el : v) {
if (el == comparator) {
present = true;
break;
}
}
if (!present) {
return false;
}
}
return true;
}
std::string describe() const override {
return "Contains: " + ::Catch::Detail::stringify( m_comparator );
}
};

template<typename T, typename AllocComp, typename AllocMatch>
class EqualsMatcher final : public MatcherBase<std::vector<T, AllocMatch>> {
std::vector<T, AllocComp> const& m_comparator;

public:
EqualsMatcher(std::vector<T, AllocComp> const& comparator):
m_comparator( comparator )
{}

bool match(std::vector<T, AllocMatch> const& v) const override {
if (m_comparator.size() != v.size())
return false;
for (std::size_t i = 0; i < v.size(); ++i)
if (m_comparator[i] != v[i])
return false;
return true;
}
std::string describe() const override {
return "Equals: " + ::Catch::Detail::stringify( m_comparator );
}
};

template<typename T, typename AllocComp, typename AllocMatch>
class ApproxMatcher final : public MatcherBase<std::vector<T, AllocMatch>> {
std::vector<T, AllocComp> const& m_comparator;
mutable Catch::Approx approx = Catch::Approx::custom();

public:
ApproxMatcher(std::vector<T, AllocComp> const& comparator):
m_comparator( comparator )
{}

bool match(std::vector<T, AllocMatch> const& v) const override {
if (m_comparator.size() != v.size())
return false;
for (std::size_t i = 0; i < v.size(); ++i)
if (m_comparator[i] != approx(v[i]))
return false;
return true;
}
std::string describe() const override {
return "is approx: " + ::Catch::Detail::stringify( m_comparator );
}
template <typename = std::enable_if_t<std::is_constructible<double, T>::value>>
ApproxMatcher& epsilon( T const& newEpsilon ) {
approx.epsilon(static_cast<double>(newEpsilon));
return *this;
}
template <typename = std::enable_if_t<std::is_constructible<double, T>::value>>
ApproxMatcher& margin( T const& newMargin ) {
approx.margin(static_cast<double>(newMargin));
return *this;
}
template <typename = std::enable_if_t<std::is_constructible<double, T>::value>>
ApproxMatcher& scale( T const& newScale ) {
approx.scale(static_cast<double>(newScale));
return *this;
}
};

template<typename T, typename AllocComp, typename AllocMatch>
class UnorderedEqualsMatcher final : public MatcherBase<std::vector<T, AllocMatch>> {
std::vector<T, AllocComp> const& m_target;

public:
UnorderedEqualsMatcher(std::vector<T, AllocComp> const& target):
m_target(target)
{}
bool match(std::vector<T, AllocMatch> const& vec) const override {
if (m_target.size() != vec.size()) {
return false;
}
return std::is_permutation(m_target.begin(), m_target.end(), vec.begin());
}

std::string describe() const override {
return "UnorderedEquals: " + ::Catch::Detail::stringify(m_target);
}
};



template<typename T, typename AllocComp = std::allocator<T>, typename AllocMatch = AllocComp>
ContainsMatcher<T, AllocComp, AllocMatch> Contains( std::vector<T, AllocComp> const& comparator ) {
return ContainsMatcher<T, AllocComp, AllocMatch>(comparator);
}

template<typename T, typename Alloc = std::allocator<T>>
VectorContainsElementMatcher<T, Alloc> VectorContains( T const& comparator ) {
return VectorContainsElementMatcher<T, Alloc>(comparator);
}

template<typename T, typename AllocComp = std::allocator<T>, typename AllocMatch = AllocComp>
EqualsMatcher<T, AllocComp, AllocMatch> Equals( std::vector<T, AllocComp> const& comparator ) {
return EqualsMatcher<T, AllocComp, AllocMatch>(comparator);
}

template<typename T, typename AllocComp = std::allocator<T>, typename AllocMatch = AllocComp>
ApproxMatcher<T, AllocComp, AllocMatch> Approx( std::vector<T, AllocComp> const& comparator ) {
return ApproxMatcher<T, AllocComp, AllocMatch>(comparator);
}

template<typename T, typename AllocComp = std::allocator<T>, typename AllocMatch = AllocComp>
UnorderedEqualsMatcher<T, AllocComp, AllocMatch> UnorderedEquals(std::vector<T, AllocComp> const& target) {
return UnorderedEqualsMatcher<T, AllocComp, AllocMatch>(target);
}

} 
} 

#endif 

#endif 




#ifndef CATCH_REPORTERS_ALL_HPP_INCLUDED
#define CATCH_REPORTERS_ALL_HPP_INCLUDED



#ifndef CATCH_REPORTER_AUTOMAKE_HPP_INCLUDED
#define CATCH_REPORTER_AUTOMAKE_HPP_INCLUDED



#ifndef CATCH_REPORTER_STREAMING_BASE_HPP_INCLUDED
#define CATCH_REPORTER_STREAMING_BASE_HPP_INCLUDED



#ifndef CATCH_REPORTER_COMMON_BASE_HPP_INCLUDED
#define CATCH_REPORTER_COMMON_BASE_HPP_INCLUDED


#include <map>
#include <string>

namespace Catch {
class ColourImpl;


class ReporterBase : public IEventListener {
protected:
Detail::unique_ptr<IStream> m_wrapped_stream;
std::ostream& m_stream;
Detail::unique_ptr<ColourImpl> m_colour;
std::map<std::string, std::string> m_customOptions;

public:
ReporterBase( ReporterConfig&& config );
~ReporterBase() override; 


void listReporters(
std::vector<ReporterDescription> const& descriptions ) override;

void listListeners(
std::vector<ListenerDescription> const& descriptions ) override;

void listTests( std::vector<TestCaseHandle> const& tests ) override;

void listTags( std::vector<TagInfo> const& tags ) override;
};
} 

#endif 

#include <vector>

namespace Catch {

class StreamingReporterBase : public ReporterBase {
public:
StreamingReporterBase(ReporterConfig&& _config):
ReporterBase(CATCH_MOVE(_config))
{}
~StreamingReporterBase() override;

void benchmarkPreparing( StringRef ) override {}
void benchmarkStarting( BenchmarkInfo const& ) override {}
void benchmarkEnded( BenchmarkStats<> const& ) override {}
void benchmarkFailed( StringRef ) override {}

void fatalErrorEncountered( StringRef  ) override {}
void noMatchingTestCases( StringRef  ) override {}
void reportInvalidTestSpec( StringRef  ) override {}

void testRunStarting( TestRunInfo const& _testRunInfo ) override;

void testCaseStarting(TestCaseInfo const& _testInfo) override  {
currentTestCaseInfo = &_testInfo;
}
void testCasePartialStarting( TestCaseInfo const&, uint64_t ) override {}
void sectionStarting(SectionInfo const& _sectionInfo) override {
m_sectionStack.push_back(_sectionInfo);
}

void assertionStarting( AssertionInfo const& ) override {}
void assertionEnded( AssertionStats const& ) override {}

void sectionEnded(SectionStats const& ) override {
m_sectionStack.pop_back();
}
void testCasePartialEnded( TestCaseStats const&, uint64_t ) override {}
void testCaseEnded(TestCaseStats const& ) override {
currentTestCaseInfo = nullptr;
}
void testRunEnded( TestRunStats const&  ) override;

void skipTest(TestCaseInfo const&) override {
}

protected:
TestRunInfo currentTestRunInfo{ "test run has not started yet"_sr };
TestCaseInfo const* currentTestCaseInfo = nullptr;

std::vector<SectionInfo> m_sectionStack;
};

} 

#endif 

#include <string>

namespace Catch {

class AutomakeReporter final : public StreamingReporterBase {
public:
AutomakeReporter(ReporterConfig&& _config):
StreamingReporterBase(CATCH_MOVE(_config))
{}
~AutomakeReporter() override;

static std::string getDescription() {
using namespace std::string_literals;
return "Reports test results in the format of Automake .trs files"s;
}

void testCaseEnded(TestCaseStats const& _testCaseStats) override;
void skipTest(TestCaseInfo const& testInfo) override;
};

} 

#endif 


#ifndef CATCH_REPORTER_COMPACT_HPP_INCLUDED
#define CATCH_REPORTER_COMPACT_HPP_INCLUDED




namespace Catch {

class CompactReporter final : public StreamingReporterBase {
public:
using StreamingReporterBase::StreamingReporterBase;

~CompactReporter() override;

static std::string getDescription();

void noMatchingTestCases( StringRef unmatchedSpec ) override;

void testRunStarting( TestRunInfo const& _testInfo ) override;

void assertionEnded(AssertionStats const& _assertionStats) override;

void sectionEnded(SectionStats const& _sectionStats) override;

void testRunEnded(TestRunStats const& _testRunStats) override;

};

} 

#endif 


#ifndef CATCH_REPORTER_CONSOLE_HPP_INCLUDED
#define CATCH_REPORTER_CONSOLE_HPP_INCLUDED


namespace Catch {
class TablePrinter;

class ConsoleReporter final : public StreamingReporterBase {
Detail::unique_ptr<TablePrinter> m_tablePrinter;

public:
ConsoleReporter(ReporterConfig&& config);
~ConsoleReporter() override;
static std::string getDescription();

void noMatchingTestCases( StringRef unmatchedSpec ) override;
void reportInvalidTestSpec( StringRef arg ) override;

void assertionStarting(AssertionInfo const&) override;

void assertionEnded(AssertionStats const& _assertionStats) override;

void sectionStarting(SectionInfo const& _sectionInfo) override;
void sectionEnded(SectionStats const& _sectionStats) override;

void benchmarkPreparing( StringRef name ) override;
void benchmarkStarting(BenchmarkInfo const& info) override;
void benchmarkEnded(BenchmarkStats<> const& stats) override;
void benchmarkFailed( StringRef error ) override;

void testCaseEnded(TestCaseStats const& _testCaseStats) override;
void testRunEnded(TestRunStats const& _testRunStats) override;
void testRunStarting(TestRunInfo const& _testRunInfo) override;

private:
void lazyPrint();

void lazyPrintWithoutClosingBenchmarkTable();
void lazyPrintRunInfo();
void printTestCaseAndSectionHeader();

void printClosedHeader(std::string const& _name);
void printOpenHeader(std::string const& _name);

void printHeaderString(std::string const& _string, std::size_t indent = 0);

void printTotalsDivider(Totals const& totals);

bool m_headerPrinted = false;
bool m_testRunInfoPrinted = false;
};

} 

#endif 


#ifndef CATCH_REPORTER_CUMULATIVE_BASE_HPP_INCLUDED
#define CATCH_REPORTER_CUMULATIVE_BASE_HPP_INCLUDED


#include <string>
#include <vector>

namespace Catch {

namespace Detail {

class AssertionOrBenchmarkResult {
Optional<AssertionStats> m_assertion;
Optional<BenchmarkStats<>> m_benchmark;
public:
AssertionOrBenchmarkResult(AssertionStats const& assertion);
AssertionOrBenchmarkResult(BenchmarkStats<> const& benchmark);

bool isAssertion() const;
bool isBenchmark() const;

AssertionStats const& asAssertion() const;
BenchmarkStats<> const& asBenchmark() const;
};
}


class CumulativeReporterBase : public ReporterBase {
public:
template<typename T, typename ChildNodeT>
struct Node {
explicit Node( T const& _value ) : value( _value ) {}

using ChildNodes = std::vector<Detail::unique_ptr<ChildNodeT>>;
T value;
ChildNodes children;
};
struct SectionNode {
explicit SectionNode(SectionStats const& _stats) : stats(_stats) {}

bool operator == (SectionNode const& other) const {
return stats.sectionInfo.lineInfo == other.stats.sectionInfo.lineInfo;
}

bool hasAnyAssertions() const;

SectionStats stats;
std::vector<Detail::unique_ptr<SectionNode>> childSections;
std::vector<Detail::AssertionOrBenchmarkResult> assertionsAndBenchmarks;
std::string stdOut;
std::string stdErr;
};


using TestCaseNode = Node<TestCaseStats, SectionNode>;
using TestRunNode = Node<TestRunStats, TestCaseNode>;

CumulativeReporterBase(ReporterConfig&& _config):
ReporterBase(CATCH_MOVE(_config))
{}
~CumulativeReporterBase() override;

void benchmarkPreparing( StringRef ) override {}
void benchmarkStarting( BenchmarkInfo const& ) override {}
void benchmarkEnded( BenchmarkStats<> const& benchmarkStats ) override;
void benchmarkFailed( StringRef ) override {}

void noMatchingTestCases( StringRef ) override {}
void reportInvalidTestSpec( StringRef ) override {}
void fatalErrorEncountered( StringRef  ) override {}

void testRunStarting( TestRunInfo const& ) override {}

void testCaseStarting( TestCaseInfo const& ) override {}
void testCasePartialStarting( TestCaseInfo const&, uint64_t ) override {}
void sectionStarting( SectionInfo const& sectionInfo ) override;

void assertionStarting( AssertionInfo const& ) override {}

void assertionEnded( AssertionStats const& assertionStats ) override;
void sectionEnded( SectionStats const& sectionStats ) override;
void testCasePartialEnded( TestCaseStats const&, uint64_t ) override {}
void testCaseEnded( TestCaseStats const& testCaseStats ) override;
void testRunEnded( TestRunStats const& testRunStats ) override;
virtual void testRunEndedCumulative() = 0;

void skipTest(TestCaseInfo const&) override {}

protected:
bool m_shouldStoreSuccesfulAssertions = true;
bool m_shouldStoreFailedAssertions = true;

Detail::unique_ptr<TestRunNode> m_testRun;

private:
std::vector<Detail::unique_ptr<TestCaseNode>> m_testCases;
Detail::unique_ptr<SectionNode> m_rootSection;
SectionNode* m_deepestSection = nullptr;
std::vector<SectionNode*> m_sectionStack;
};

} 

#endif 


#ifndef CATCH_REPORTER_EVENT_LISTENER_HPP_INCLUDED
#define CATCH_REPORTER_EVENT_LISTENER_HPP_INCLUDED


namespace Catch {


class EventListenerBase : public IEventListener {
public:
using IEventListener::IEventListener;

void reportInvalidTestSpec( StringRef unmatchedSpec ) override;
void fatalErrorEncountered( StringRef error ) override;

void benchmarkPreparing( StringRef name ) override;
void benchmarkStarting( BenchmarkInfo const& benchmarkInfo ) override;
void benchmarkEnded( BenchmarkStats<> const& benchmarkStats ) override;
void benchmarkFailed( StringRef error ) override;

void assertionStarting( AssertionInfo const& assertionInfo ) override;
void assertionEnded( AssertionStats const& assertionStats ) override;

void listReporters(
std::vector<ReporterDescription> const& descriptions ) override;
void listListeners(
std::vector<ListenerDescription> const& descriptions ) override;
void listTests( std::vector<TestCaseHandle> const& tests ) override;
void listTags( std::vector<TagInfo> const& tagInfos ) override;

void noMatchingTestCases( StringRef unmatchedSpec ) override;
void testRunStarting( TestRunInfo const& testRunInfo ) override;
void testCaseStarting( TestCaseInfo const& testInfo ) override;
void testCasePartialStarting( TestCaseInfo const& testInfo,
uint64_t partNumber ) override;
void sectionStarting( SectionInfo const& sectionInfo ) override;
void sectionEnded( SectionStats const& sectionStats ) override;
void testCasePartialEnded( TestCaseStats const& testCaseStats,
uint64_t partNumber ) override;
void testCaseEnded( TestCaseStats const& testCaseStats ) override;
void testRunEnded( TestRunStats const& testRunStats ) override;
void skipTest( TestCaseInfo const& testInfo ) override;
};

} 

#endif 


#ifndef CATCH_REPORTER_HELPERS_HPP_INCLUDED
#define CATCH_REPORTER_HELPERS_HPP_INCLUDED

#include <iosfwd>
#include <string>
#include <vector>


namespace Catch {

class IConfig;
class TestCaseHandle;
class ColourImpl;

std::string getFormattedDuration( double duration );

bool shouldShowDuration( IConfig const& config, double duration );

std::string serializeFilters( std::vector<std::string> const& filters );

struct lineOfChars {
char c;
constexpr lineOfChars( char c_ ): c( c_ ) {}

friend std::ostream& operator<<( std::ostream& out, lineOfChars value );
};


void
defaultListReporters( std::ostream& out,
std::vector<ReporterDescription> const& descriptions,
Verbosity verbosity );


void defaultListListeners( std::ostream& out,
std::vector<ListenerDescription> const& descriptions );


void defaultListTags( std::ostream& out, std::vector<TagInfo> const& tags, bool isFiltered );


void defaultListTests( std::ostream& out,
ColourImpl* streamColour,
std::vector<TestCaseHandle> const& tests,
bool isFiltered,
Verbosity verbosity );


void printTestRunTotals( std::ostream& stream,
ColourImpl& streamColour,
Totals const& totals );

} 

#endif 


#ifndef CATCH_REPORTER_JUNIT_HPP_INCLUDED
#define CATCH_REPORTER_JUNIT_HPP_INCLUDED



namespace Catch {

class JunitReporter final : public CumulativeReporterBase {
public:
JunitReporter(ReporterConfig&& _config);

~JunitReporter() override = default;

static std::string getDescription();

void testRunStarting(TestRunInfo const& runInfo) override;

void testCaseStarting(TestCaseInfo const& testCaseInfo) override;
void assertionEnded(AssertionStats const& assertionStats) override;

void testCaseEnded(TestCaseStats const& testCaseStats) override;

void testRunEndedCumulative() override;

private:
void writeRun(TestRunNode const& testRunNode, double suiteTime);

void writeTestCase(TestCaseNode const& testCaseNode);

void writeSection( std::string const& className,
std::string const& rootName,
SectionNode const& sectionNode,
bool testOkToFail );

void writeAssertions(SectionNode const& sectionNode);
void writeAssertion(AssertionStats const& stats);

XmlWriter xml;
Timer suiteTimer;
std::string stdOutForSuite;
std::string stdErrForSuite;
unsigned int unexpectedExceptions = 0;
bool m_okToFail = false;
};

} 

#endif 


#ifndef CATCH_REPORTER_MULTI_HPP_INCLUDED
#define CATCH_REPORTER_MULTI_HPP_INCLUDED


namespace Catch {

class MultiReporter final : public IEventListener {

std::vector<IEventListenerPtr> m_reporterLikes;
bool m_haveNoncapturingReporters = false;

size_t m_insertedListeners = 0;

void updatePreferences(IEventListener const& reporterish);

public:
using IEventListener::IEventListener;

void addListener( IEventListenerPtr&& listener );
void addReporter( IEventListenerPtr&& reporter );

public: 

void noMatchingTestCases( StringRef unmatchedSpec ) override;
void fatalErrorEncountered( StringRef error ) override;
void reportInvalidTestSpec( StringRef arg ) override;

void benchmarkPreparing( StringRef name ) override;
void benchmarkStarting( BenchmarkInfo const& benchmarkInfo ) override;
void benchmarkEnded( BenchmarkStats<> const& benchmarkStats ) override;
void benchmarkFailed( StringRef error ) override;

void testRunStarting( TestRunInfo const& testRunInfo ) override;
void testCaseStarting( TestCaseInfo const& testInfo ) override;
void testCasePartialStarting(TestCaseInfo const& testInfo, uint64_t partNumber) override;
void sectionStarting( SectionInfo const& sectionInfo ) override;
void assertionStarting( AssertionInfo const& assertionInfo ) override;

void assertionEnded( AssertionStats const& assertionStats ) override;
void sectionEnded( SectionStats const& sectionStats ) override;
void testCasePartialEnded(TestCaseStats const& testInfo, uint64_t partNumber) override;
void testCaseEnded( TestCaseStats const& testCaseStats ) override;
void testRunEnded( TestRunStats const& testRunStats ) override;

void skipTest( TestCaseInfo const& testInfo ) override;

void listReporters(std::vector<ReporterDescription> const& descriptions) override;
void listListeners(std::vector<ListenerDescription> const& descriptions) override;
void listTests(std::vector<TestCaseHandle> const& tests) override;
void listTags(std::vector<TagInfo> const& tags) override;


};

} 

#endif 


#ifndef CATCH_REPORTER_REGISTRARS_HPP_INCLUDED
#define CATCH_REPORTER_REGISTRARS_HPP_INCLUDED


#include <type_traits>

namespace Catch {

namespace Detail {

template <typename T, typename = void>
struct has_description : std::false_type {};

template <typename T>
struct has_description<
T,
void_t<decltype( T::getDescription() )>>
: std::true_type {};

void registerReporterImpl( std::string const& name,
IReporterFactoryPtr reporterPtr );

} 

class IEventListener;
using IEventListenerPtr = Detail::unique_ptr<IEventListener>;

template <typename T>
class ReporterFactory : public IReporterFactory {

IEventListenerPtr create( ReporterConfig&& config ) const override {
return Detail::make_unique<T>( CATCH_MOVE(config) );
}

std::string getDescription() const override {
return T::getDescription();
}
};


template<typename T>
class ReporterRegistrar {
public:
explicit ReporterRegistrar( std::string const& name ) {
registerReporterImpl( name,
Detail::make_unique<ReporterFactory<T>>() );
}
};

template<typename T>
class ListenerRegistrar {

class TypedListenerFactory : public EventListenerFactory {
StringRef m_listenerName;

std::string getDescriptionImpl( std::true_type ) const {
return T::getDescription();
}

std::string getDescriptionImpl( std::false_type ) const {
return "(No description provided)";
}

public:
TypedListenerFactory( StringRef listenerName ):
m_listenerName( listenerName ) {}

IEventListenerPtr create( IConfig const* config ) const override {
return Detail::make_unique<T>( config );
}

StringRef getName() const override {
return m_listenerName;
}

std::string getDescription() const override {
return getDescriptionImpl( Detail::has_description<T>{} );
}
};

public:
ListenerRegistrar(StringRef listenerName) {
getMutableRegistryHub().registerListener( Detail::make_unique<TypedListenerFactory>(listenerName) );
}
};
}

#if !defined(CATCH_CONFIG_DISABLE)

#    define CATCH_REGISTER_REPORTER( name, reporterType )                      \
CATCH_INTERNAL_START_WARNINGS_SUPPRESSION                              \
CATCH_INTERNAL_SUPPRESS_GLOBALS_WARNINGS                               \
namespace {                                                            \
Catch::ReporterRegistrar<reporterType> INTERNAL_CATCH_UNIQUE_NAME( \
catch_internal_RegistrarFor )( name );                         \
}                                                                      \
CATCH_INTERNAL_STOP_WARNINGS_SUPPRESSION

#    define CATCH_REGISTER_LISTENER( listenerType )                            \
CATCH_INTERNAL_START_WARNINGS_SUPPRESSION                              \
CATCH_INTERNAL_SUPPRESS_GLOBALS_WARNINGS                               \
namespace {                                                            \
Catch::ListenerRegistrar<listenerType> INTERNAL_CATCH_UNIQUE_NAME( \
catch_internal_RegistrarFor )( #listenerType );                \
}                                                                      \
CATCH_INTERNAL_STOP_WARNINGS_SUPPRESSION

#else 

#define CATCH_REGISTER_REPORTER(name, reporterType)
#define CATCH_REGISTER_LISTENER(listenerType)

#endif 

#endif 


#ifndef CATCH_REPORTER_SONARQUBE_HPP_INCLUDED
#define CATCH_REPORTER_SONARQUBE_HPP_INCLUDED



namespace Catch {

class SonarQubeReporter final : public CumulativeReporterBase {
public:
SonarQubeReporter(ReporterConfig&& config)
: CumulativeReporterBase(CATCH_MOVE(config))
, xml(m_stream) {
m_preferences.shouldRedirectStdOut = true;
m_preferences.shouldReportAllAssertions = true;
m_shouldStoreSuccesfulAssertions = false;
}

~SonarQubeReporter() override = default;

static std::string getDescription() {
using namespace std::string_literals;
return "Reports test results in the Generic Test Data SonarQube XML format"s;
}

void testRunStarting( TestRunInfo const& testRunInfo ) override;

void testRunEndedCumulative() override {
writeRun( *m_testRun );
xml.endElement();
}

void writeRun( TestRunNode const& groupNode );

void writeTestFile(std::string const& filename, std::vector<TestCaseNode const*> const& testCaseNodes);

void writeTestCase(TestCaseNode const& testCaseNode);

void writeSection(std::string const& rootName, SectionNode const& sectionNode, bool okToFail);

void writeAssertions(SectionNode const& sectionNode, bool okToFail);

void writeAssertion(AssertionStats const& stats, bool okToFail);

private:
XmlWriter xml;
};


} 

#endif 


#ifndef CATCH_REPORTER_TAP_HPP_INCLUDED
#define CATCH_REPORTER_TAP_HPP_INCLUDED


namespace Catch {

class TAPReporter final : public StreamingReporterBase {
public:
TAPReporter( ReporterConfig&& config ):
StreamingReporterBase( CATCH_MOVE(config) ) {
m_preferences.shouldReportAllAssertions = true;
}
~TAPReporter() override = default;

static std::string getDescription() {
using namespace std::string_literals;
return "Reports test results in TAP format, suitable for test harnesses"s;
}

void testRunStarting( TestRunInfo const& testInfo ) override;

void noMatchingTestCases( StringRef unmatchedSpec ) override;

void assertionEnded(AssertionStats const& _assertionStats) override;

void testRunEnded(TestRunStats const& _testRunStats) override;

private:
std::size_t counter = 0;
};

} 

#endif 


#ifndef CATCH_REPORTER_TEAMCITY_HPP_INCLUDED
#define CATCH_REPORTER_TEAMCITY_HPP_INCLUDED


#include <cstring>

#ifdef __clang__
#   pragma clang diagnostic push
#   pragma clang diagnostic ignored "-Wpadded"
#endif

namespace Catch {

class TeamCityReporter final : public StreamingReporterBase {
public:
TeamCityReporter( ReporterConfig&& _config )
:   StreamingReporterBase( CATCH_MOVE(_config) )
{
m_preferences.shouldRedirectStdOut = true;
}

~TeamCityReporter() override;

static std::string getDescription() {
using namespace std::string_literals;
return "Reports test results as TeamCity service messages"s;
}

void testRunStarting( TestRunInfo const& groupInfo ) override;
void testRunEnded( TestRunStats const& testGroupStats ) override;


void assertionEnded(AssertionStats const& assertionStats) override;

void sectionStarting(SectionInfo const& sectionInfo) override {
m_headerPrintedForThisSection = false;
StreamingReporterBase::sectionStarting( sectionInfo );
}

void testCaseStarting(TestCaseInfo const& testInfo) override;

void testCaseEnded(TestCaseStats const& testCaseStats) override;

private:
void printSectionHeader(std::ostream& os);

bool m_headerPrintedForThisSection = false;
Timer m_testTimer;
};

} 

#ifdef __clang__
#   pragma clang diagnostic pop
#endif

#endif 


#ifndef CATCH_REPORTER_XML_HPP_INCLUDED
#define CATCH_REPORTER_XML_HPP_INCLUDED




namespace Catch {
class XmlReporter : public StreamingReporterBase {
public:
XmlReporter(ReporterConfig&& _config);

~XmlReporter() override;

static std::string getDescription();

virtual std::string getStylesheetRef() const;

void writeSourceInfo(SourceLineInfo const& sourceInfo);

public: 

void testRunStarting(TestRunInfo const& testInfo) override;

void testCaseStarting(TestCaseInfo const& testInfo) override;

void sectionStarting(SectionInfo const& sectionInfo) override;

void assertionStarting(AssertionInfo const&) override;

void assertionEnded(AssertionStats const& assertionStats) override;

void sectionEnded(SectionStats const& sectionStats) override;

void testCaseEnded(TestCaseStats const& testCaseStats) override;

void testRunEnded(TestRunStats const& testRunStats) override;

void benchmarkPreparing( StringRef name ) override;
void benchmarkStarting(BenchmarkInfo const&) override;
void benchmarkEnded(BenchmarkStats<> const&) override;
void benchmarkFailed( StringRef error ) override;

void listReporters(std::vector<ReporterDescription> const& descriptions) override;
void listListeners(std::vector<ListenerDescription> const& descriptions) override;
void listTests(std::vector<TestCaseHandle> const& tests) override;
void listTags(std::vector<TagInfo> const& tags) override;

private:
Timer m_testCaseTimer;
XmlWriter m_xml;
int m_sectionDepth = 0;
};

} 

#endif 

#endif 

#endif 
#endif 
