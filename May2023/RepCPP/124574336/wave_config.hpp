

#if !defined(BOOST_WAVE_CONFIG_HPP_F143F90A_A63F_4B27_AC41_9CA4F14F538D_INCLUDED)
#define BOOST_WAVE_CONFIG_HPP_F143F90A_A63F_4B27_AC41_9CA4F14F538D_INCLUDED

#include <boost/config.hpp>
#include <boost/config/pragma_message.hpp>
#include <boost/detail/workaround.hpp>
#include <boost/version.hpp>
#include <boost/spirit/include/classic_version.hpp>
#include <boost/wave/wave_version.hpp>

#if !defined(BOOST_WAVE_MAX_INCLUDE_LEVEL_DEPTH)
#define BOOST_WAVE_MAX_INCLUDE_LEVEL_DEPTH 1024
#endif

#if !defined(BOOST_WAVE_SUPPORT_VARIADICS_PLACEMARKERS)
#define BOOST_WAVE_SUPPORT_VARIADICS_PLACEMARKERS 1
#endif

#if !defined(BOOST_WAVE_SUPPORT_VA_OPT) && BOOST_WAVE_SUPPORT_VARIADICS_PLACEMARKERS
#define BOOST_WAVE_SUPPORT_VA_OPT 1
#endif

#if !defined(BOOST_WAVE_SUPPORT_WARNING_DIRECTIVE)
#define BOOST_WAVE_SUPPORT_WARNING_DIRECTIVE 1
#endif

#if !defined(BOOST_WAVE_SUPPORT_PRAGMA_ONCE)
#define BOOST_WAVE_SUPPORT_PRAGMA_ONCE 1
#endif

#if !defined(BOOST_WAVE_SUPPORT_PRAGMA_MESSAGE)
#define BOOST_WAVE_SUPPORT_PRAGMA_MESSAGE 1
#endif

#if !defined(BOOST_WAVE_SUPPORT_INCLUDE_NEXT)
#define BOOST_WAVE_SUPPORT_INCLUDE_NEXT 1
#endif

#if !defined(BOOST_WAVE_SUPPORT_CPP0X)
#define BOOST_WAVE_SUPPORT_CPP0X 1
#undef BOOST_WAVE_SUPPORT_VARIADICS_PLACEMARKERS
#define BOOST_WAVE_SUPPORT_VARIADICS_PLACEMARKERS 1
#endif

#if !defined(BOOST_WAVE_SUPPORT_CPP1Z)
#  define BOOST_WAVE_SUPPORT_CPP1Z 1
#  undef BOOST_WAVE_SUPPORT_CPP0X
#  define BOOST_WAVE_SUPPORT_CPP0X 1
#  undef BOOST_WAVE_SUPPORT_VARIADICS_PLACEMARKERS
#  define BOOST_WAVE_SUPPORT_VARIADICS_PLACEMARKERS 1
#  if !defined(BOOST_WAVE_SUPPORT_HAS_INCLUDE)
#    define BOOST_WAVE_SUPPORT_HAS_INCLUDE 1
#  endif
#elif BOOST_WAVE_SUPPORT_CPP1Z == 0
#  undef BOOST_WAVE_SUPPORT_HAS_INCLUDE
#  define BOOST_WAVE_SUPPORT_HAS_INCLUDE 0
#endif

#if !defined(BOOST_WAVE_SUPPORT_CPP2A)
#  define BOOST_WAVE_SUPPORT_CPP2A 1
#  undef BOOST_WAVE_SUPPORT_CPP0X
#  define BOOST_WAVE_SUPPORT_CPP0X 1
#  undef BOOST_WAVE_SUPPORT_VARIADICS_PLACEMARKERS
#  define BOOST_WAVE_SUPPORT_VARIADICS_PLACEMARKERS 1
#  undef BOOST_WAVE_SUPPORT_CPP1Z
#  define BOOST_WAVE_SUPPORT_CPP1Z 1
#  if !defined(BOOST_WAVE_SUPPORT_HAS_INCLUDE)
#    define BOOST_WAVE_SUPPORT_HAS_INCLUDE 1
#  endif
#  if !defined(BOOST_WAVE_SUPPORT_VA_OPT)
#    define BOOST_WAVE_SUPPORT_VA_OPT 1
#  endif
#elif BOOST_WAVE_SUPPORT_CPP2A == 0
#  undef BOOST_WAVE_SUPPORT_VA_OPT
#  define BOOST_WAVE_SUPPORT_VA_OPT 0
#endif

#if !defined(BOOST_WAVE_SUPPORT_MS_EXTENSIONS)
#if defined(BOOST_WINDOWS)
#define BOOST_WAVE_SUPPORT_MS_EXTENSIONS 1
#else
#define BOOST_WAVE_SUPPORT_MS_EXTENSIONS 0
#endif
#endif

#if !defined(BOOST_WAVE_PREPROCESS_ERROR_MESSAGE_BODY)
#define BOOST_WAVE_PREPROCESS_ERROR_MESSAGE_BODY 1
#endif

#if !defined(BOOST_WAVE_EMIT_PRAGMA_DIRECTIVES)
#define BOOST_WAVE_EMIT_PRAGMA_DIRECTIVES 1
#endif

#if !defined(BOOST_WAVE_PREPROCESS_PRAGMA_BODY)
#define BOOST_WAVE_PREPROCESS_PRAGMA_BODY 1
#endif

#if !defined(BOOST_WAVE_ENABLE_COMMANDLINE_MACROS)
#define BOOST_WAVE_ENABLE_COMMANDLINE_MACROS 1
#endif

#if !defined(BOOST_WAVE_PRAGMA_KEYWORD)
#define BOOST_WAVE_PRAGMA_KEYWORD "wave"
#endif

#if !defined(BOOST_WAVE_SUPPORT_THREADING)
#if defined(BOOST_HAS_THREADS)
#define BOOST_WAVE_SUPPORT_THREADING 1
#else
#define BOOST_WAVE_SUPPORT_THREADING 0
#endif
#endif

#if BOOST_WAVE_SUPPORT_THREADING != 0
#ifndef BOOST_SPIRIT_THREADSAFE
#define BOOST_SPIRIT_THREADSAFE
#endif
#define PHOENIX_THREADSAFE 1
#else
#define BOOST_NO_MT 1
#endif

#if !defined(BOOST_WAVE_STRINGTYPE)

#if BOOST_WORKAROUND(__MWERKS__, < 0x3200) || \
(defined(__DECCXX) && defined(__alpha)) || \
defined(BOOST_WAVE_STRINGTYPE_USE_STDSTRING)

#define BOOST_WAVE_STRINGTYPE std::string

#if !defined(BOOST_WAVE_STRINGTYPE_USE_STDSTRING)
#define BOOST_WAVE_STRINGTYPE_USE_STDSTRING 1
#endif

#else

#define BOOST_WAVE_STRINGTYPE boost::wave::util::flex_string<                 \
char, std::char_traits<char>, std::allocator<char>,                   \
boost::wave::util::CowString<                                         \
boost::wave::util::AllocatorStringStorage<char>                   \
>                                                                     \
>                                                                         \


#include <boost/wave/util/flex_string.hpp>

#endif 
#endif 




#define BOOST_SPIRIT_DEBUG_FLAGS_CPP_GRAMMAR            0x0001
#define BOOST_SPIRIT_DEBUG_FLAGS_TIME_CONVERSION        0x0002
#define BOOST_SPIRIT_DEBUG_FLAGS_CPP_EXPR_GRAMMAR       0x0004
#define BOOST_SPIRIT_DEBUG_FLAGS_INTLIT_GRAMMAR         0x0008
#define BOOST_SPIRIT_DEBUG_FLAGS_CHLIT_GRAMMAR          0x0010
#define BOOST_SPIRIT_DEBUG_FLAGS_DEFINED_GRAMMAR        0x0020
#define BOOST_SPIRIT_DEBUG_FLAGS_PREDEF_MACROS_GRAMMAR  0x0040
#define BOOST_SPIRIT_DEBUG_FLAGS_HAS_INCLUDE_GRAMMAR    0x0080

#if !defined(BOOST_SPIRIT_DEBUG_FLAGS_CPP)
#define BOOST_SPIRIT_DEBUG_FLAGS_CPP    0    
#endif

#if !defined(BOOST_WAVE_DUMP_PARSE_TREE)
#define BOOST_WAVE_DUMP_PARSE_TREE 0
#endif
#if BOOST_WAVE_DUMP_PARSE_TREE != 0 && !defined(BOOST_WAVE_DUMP_PARSE_TREE_OUT)
#define BOOST_WAVE_DUMP_PARSE_TREE_OUT std::cerr
#endif

#if !defined(BOOST_WAVE_DUMP_CONDITIONAL_EXPRESSIONS)
#define BOOST_WAVE_DUMP_CONDITIONAL_EXPRESSIONS 0
#endif
#if BOOST_WAVE_DUMP_CONDITIONAL_EXPRESSIONS != 0 && \
!defined(BOOST_WAVE_DUMP_CONDITIONAL_EXPRESSIONS_OUT)
#define BOOST_WAVE_DUMP_CONDITIONAL_EXPRESSIONS_OUT std::cerr
#endif

#if !defined(BOOST_WAVE_SEPARATE_LEXER_INSTANTIATION)
#define BOOST_WAVE_SEPARATE_LEXER_INSTANTIATION 1
#endif

#if !defined(BOOST_WAVE_SEPARATE_GRAMMAR_INSTANTIATION)
#define BOOST_WAVE_SEPARATE_GRAMMAR_INSTANTIATION 1
#endif

#if !defined(BOOST_WAVE_USE_STRICT_LEXER)
#define BOOST_WAVE_USE_STRICT_LEXER 0
#endif

#if !defined(BOOST_WAVE_SERIALIZATION)
#define BOOST_WAVE_SERIALIZATION 0
#endif

#if !defined(BOOST_WAVE_SUPPORT_IMPORT_KEYWORD)
#define BOOST_WAVE_SUPPORT_IMPORT_KEYWORD 0
#endif

#if !defined(BOOST_WAVE_SUPPORT_LONGLONG_INTEGER_LITERALS)
#define BOOST_WAVE_SUPPORT_LONGLONG_INTEGER_LITERALS 0
#endif

namespace boost { namespace wave
{
#if defined(BOOST_HAS_LONG_LONG) && \
BOOST_WAVE_SUPPORT_LONGLONG_INTEGER_LITERALS != 0
typedef boost::long_long_type int_literal_type;
typedef boost::ulong_long_type uint_literal_type;
#else
typedef long int_literal_type;
typedef unsigned long uint_literal_type;
#endif
}}


#define BOOST_WAVE_WCHAR_T_AUTOSELECT       1
#define BOOST_WAVE_WCHAR_T_FORCE_SIGNED     2
#define BOOST_WAVE_WCHAR_T_FORCE_UNSIGNED   3

#if !defined(BOOST_WAVE_WCHAR_T_SIGNEDNESS)
#define BOOST_WAVE_WCHAR_T_SIGNEDNESS BOOST_WAVE_WCHAR_T_AUTOSELECT
#endif

#if !defined(PHOENIX_LIMIT)
#define PHOENIX_LIMIT 6
#endif
#if PHOENIX_LIMIT < 6
#error "Boost.Wave: the constant PHOENIX_LIMIT must be at least defined to 6" \
" to compile the library."
#endif

#if (defined(BOOST_WAVE_DYN_LINK) || defined(BOOST_ALL_DYN_LINK)) && \
!defined(BOOST_WAVE_STATIC_LINK)

#if defined(BOOST_WAVE_SOURCE)
#define BOOST_WAVE_DECL BOOST_SYMBOL_EXPORT
#define BOOST_WAVE_BUILD_DLL
#else
#define BOOST_WAVE_DECL BOOST_SYMBOL_IMPORT
#endif

#endif 

#ifndef BOOST_WAVE_DECL
#define BOOST_WAVE_DECL
#endif

#if BOOST_VERSION >= 103100
#if !defined(BOOST_WAVE_SOURCE) && !defined(BOOST_ALL_NO_LIB) && \
!defined(BOOST_WAVE_NO_LIB)

#define BOOST_LIB_NAME boost_wave

#if defined(BOOST_ALL_DYN_LINK) || defined(BOOST_WAVE_DYN_LINK)
#define BOOST_DYN_LINK
#endif

#include <boost/config/auto_link.hpp>

#endif  
#endif  


#if defined(BOOST_NO_CXX11_VARIADIC_TEMPLATES) || defined(BOOST_NO_CXX11_RVALUE_REFERENCES) \
|| defined(BOOST_NO_CXX11_HDR_THREAD) \
|| defined(BOOST_NO_CXX11_HDR_MUTEX) || defined(BOOST_NO_CXX11_HDR_REGEX)

BOOST_PRAGMA_MESSAGE("C++03 support is deprecated in Boost.Wave 1.74 and will be removed in Boost.Wave 1.77.")

#endif


#if !defined(BOOST_WAVE_USE_DEPRECIATED_PREPROCESSING_HOOKS)
#if BOOST_VERSION < 103500  
#define BOOST_WAVE_USE_DEPRECIATED_PREPROCESSING_HOOKS 1
#else
#define BOOST_WAVE_USE_DEPRECIATED_PREPROCESSING_HOOKS 0
#endif
#elif BOOST_WAVE_USE_DEPRECIATED_PREPROCESSING_HOOKS != 0
BOOST_PRAGMA_MESSAGE("The old preprocessing hooks were deprecated in Boost 1.35 and will be removed in 1.76. See https:
#endif

#endif 
