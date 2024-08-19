




#if defined(__BORLANDC__) && !defined(__clang__)
#  if (__BORLANDC__ == 0x550) || (__BORLANDC__ == 0x551)
#     if defined(_RTLDLL) && defined(_RWSTD_COMPILE_INSTANTIATE)
#        ifdef BOOST_REGEX_BUILD_DLL
#           error _RWSTD_COMPILE_INSTANTIATE must not be defined when building regex++ as a DLL
#        else
#           pragma message("Defining _RWSTD_COMPILE_INSTANTIATE when linking to the DLL version of the RTL may produce memory corruption problems in std::basic_string, as a result of separate versions of basic_string's static data in the RTL and you're exe/dll: be warned!!")
#        endif
#     endif
#     ifndef _RTLDLL
#        define _RWSTD_COMPILE_INSTANTIATE
#     endif
#     define BOOST_REGEX_NO_EXTERNAL_TEMPLATES
#  endif
#  if (__BORLANDC__ <= 0x540) && !defined(BOOST_REGEX_NO_LIB) && !defined(_NO_VCL)
#     define BOOST_REGEX_STATIC_LINK
#  endif
#  if !defined(__CONSOLE__) && !defined(_NO_VCL)
#     define BOOST_REGEX_USE_VCL
#  endif
#  ifndef _Windows
#     ifndef BOOST_REGEX_NO_LIB
#        define BOOST_REGEX_NO_LIB
#     endif
#     ifndef BOOST_REGEX_STATIC_LINK
#        define BOOST_REGEX_STATIC_LINK
#     endif
#  endif

#if __BORLANDC__ < 0x600
#include <cstring>
#undef strcmp
#undef strcpy
#endif

#endif


