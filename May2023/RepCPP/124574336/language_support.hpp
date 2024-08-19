
#if !defined(BOOST_LANGUAGE_SUPPORT_HPP_93EDD057_2DEF_44BC_BC9F_FDABB9F51AFA_INCLUDED)
#define BOOST_LANGUAGE_SUPPORT_HPP_93EDD057_2DEF_44BC_BC9F_FDABB9F51AFA_INCLUDED

#include <boost/wave/wave_config.hpp>

#ifdef BOOST_HAS_ABI_HEADERS
#include BOOST_ABI_PREFIX
#endif

namespace boost {
namespace wave {

enum language_support {
support_normal = 0x01,
support_cpp = support_normal,

support_option_long_long = 0x02,

#if BOOST_WAVE_SUPPORT_VARIADICS_PLACEMARKERS != 0
support_option_variadics = 0x04,
support_c99 = support_option_variadics | support_option_long_long | 0x08,
#endif
#if BOOST_WAVE_SUPPORT_CPP0X != 0
support_option_no_newline_at_end_of_file = 0x20,

support_cpp0x = support_option_variadics | support_option_long_long |
support_option_no_newline_at_end_of_file | 0x10,
support_cpp11 = support_cpp0x,
#if BOOST_WAVE_SUPPORT_CPP1Z != 0
support_option_has_include = 0x10000,

support_cpp1z = support_option_variadics | support_option_long_long |
support_option_no_newline_at_end_of_file | support_option_has_include |
0x20000,
support_cpp17 = support_cpp1z,
#if BOOST_WAVE_SUPPORT_CPP2A != 0
support_option_va_opt = 0x40000,

support_cpp2a = support_option_variadics | support_option_long_long |
support_option_no_newline_at_end_of_file | support_option_has_include |
support_option_va_opt | 0x80000,
support_cpp20 = support_cpp2a,
#endif
#endif
#endif

support_option_mask = 0xFFC0,
support_option_emit_contnewlines = 0x0040,
support_option_insert_whitespace = 0x0080,
support_option_preserve_comments = 0x0100,
support_option_no_character_validation = 0x0200,
support_option_convert_trigraphs = 0x0400,
support_option_single_line = 0x0800,
support_option_prefer_pp_numbers = 0x1000,
support_option_emit_line_directives = 0x2000,
support_option_include_guard_detection = 0x4000,
support_option_emit_pragma_directives = 0x8000
};

inline bool
need_cpp(language_support language)
{
return (language & ~support_option_mask) == support_cpp;
}

#if BOOST_WAVE_SUPPORT_CPP0X != 0

inline bool
need_cpp0x(language_support language)
{
return (language & ~support_option_mask) == support_cpp0x;
}

#else

inline bool
need_cpp0x(language_support language)
{
return false;
}

#endif

#if BOOST_WAVE_SUPPORT_CPP2A != 0

inline bool
need_cpp2a(language_support language)
{
return (language & ~support_option_mask) == support_cpp2a;
}

#else

inline bool
need_cpp2a(language_support language)
{
return false;
}

#endif

#if BOOST_WAVE_SUPPORT_VARIADICS_PLACEMARKERS != 0
inline bool
need_c99(language_support language)
{
return (language & ~support_option_mask) == support_c99;
}

#else  

inline bool
need_variadics(language_support language)
{
return false;
}

inline language_support
enable_variadics(language_support language, bool enable = true)
{
return language;
}

inline bool
need_c99(language_support language)
{
return false;
}

#endif 

inline language_support
get_support_options(language_support language)
{
return static_cast<language_support>(language & support_option_mask);
}

inline language_support
set_support_options(language_support language, language_support option)
{
return static_cast<language_support>(
(language & ~support_option_mask) | (option & support_option_mask));
}

#define BOOST_WAVE_NEED_OPTION(option)                                        \
inline bool need_ ## option(language_support language)                    \
{                                                                         \
return (language & support_option_ ## option) ? true : false;         \
}                                                                         \


#define BOOST_WAVE_ENABLE_OPTION(option)                                      \
inline language_support                                                   \
enable_ ## option(language_support language, bool enable = true)          \
{                                                                         \
if (enable)                                                           \
return static_cast<language_support>(language | support_option_ ## option); \
return static_cast<language_support>(language & ~support_option_ ## option);    \
}                                                                         \


#define BOOST_WAVE_OPTION(option)                                             \
BOOST_WAVE_NEED_OPTION(option)                                            \
BOOST_WAVE_ENABLE_OPTION(option)                                          \


BOOST_WAVE_OPTION(long_long)                 
BOOST_WAVE_OPTION(no_character_validation)   
BOOST_WAVE_OPTION(preserve_comments)         
BOOST_WAVE_OPTION(prefer_pp_numbers)         
BOOST_WAVE_OPTION(emit_line_directives)      
BOOST_WAVE_OPTION(single_line)               
BOOST_WAVE_OPTION(convert_trigraphs)         
#if BOOST_WAVE_SUPPORT_PRAGMA_ONCE != 0
BOOST_WAVE_OPTION(include_guard_detection)   
#endif
#if BOOST_WAVE_SUPPORT_VARIADICS_PLACEMARKERS != 0
BOOST_WAVE_OPTION(variadics)                 
#endif
#if BOOST_WAVE_SUPPORT_VA_OPT != 0
BOOST_WAVE_OPTION(va_opt)                    
#endif
#if BOOST_WAVE_EMIT_PRAGMA_DIRECTIVES != 0
BOOST_WAVE_OPTION(emit_pragma_directives)    
#endif
BOOST_WAVE_OPTION(insert_whitespace)         
BOOST_WAVE_OPTION(emit_contnewlines)         
#if BOOST_WAVE_SUPPORT_CPP0X != 0
BOOST_WAVE_OPTION(no_newline_at_end_of_file) 
#endif
#if BOOST_WAVE_SUPPORT_HAS_INCLUDE != 0
BOOST_WAVE_OPTION(has_include)               
#endif

#undef BOOST_WAVE_NEED_OPTION
#undef BOOST_WAVE_ENABLE_OPTION
#undef BOOST_WAVE_OPTION

}   
}   

#ifdef BOOST_HAS_ABI_HEADERS
#include BOOST_ABI_SUFFIX
#endif

#endif 
