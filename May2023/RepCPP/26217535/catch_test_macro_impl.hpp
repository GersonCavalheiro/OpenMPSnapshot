

#ifndef CATCH_TEST_MACRO_IMPL_HPP_INCLUDED
#define CATCH_TEST_MACRO_IMPL_HPP_INCLUDED

#include <catch2/catch_user_config.hpp>
#include <catch2/internal/catch_assertion_handler.hpp>
#include <catch2/interfaces/catch_interfaces_capture.hpp>
#include <catch2/internal/catch_stringref.hpp>
#include <catch2/internal/catch_source_line_info.hpp>

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
