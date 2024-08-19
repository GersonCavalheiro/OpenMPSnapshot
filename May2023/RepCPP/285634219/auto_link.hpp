




#ifdef __cplusplus
#  ifndef BOOST_CONFIG_HPP
#     include <boost/config.hpp>
#  endif
#elif defined(_MSC_VER) && !defined(__MWERKS__) && !defined(__EDG_VERSION__)
#  define BOOST_MSVC _MSC_VER
#  define BOOST_STRINGIZE(X) BOOST_DO_STRINGIZE(X)
#  define BOOST_DO_STRINGIZE(X) #X
#endif
#if defined(BOOST_MSVC) \
|| defined(__BORLANDC__) \
|| (defined(__MWERKS__) && defined(_WIN32) && (__MWERKS__ >= 0x3000)) \
|| (defined(__ICL) && defined(_MSC_EXTENSIONS) && (_MSC_VER >= 1200))

#ifndef BOOST_VERSION_HPP
#  include <boost/version.hpp>
#endif

#ifndef BOOST_LIB_NAME
#  error "Macro BOOST_LIB_NAME not set (internal error)"
#endif

#if defined(__MSVC_RUNTIME_CHECKS) && !defined(_DEBUG)
#  pragma message("Using the /RTC option without specifying a debug runtime will lead to linker errors")
#  pragma message("Hint: go to the code generation options and switch to one of the debugging runtimes")
#  error "Incompatible build options"
#endif
#ifndef BOOST_LIB_TOOLSET
#if defined(BOOST_MSVC) && (BOOST_MSVC < 1300)

#  ifdef UNDER_CE
#    define BOOST_LIB_TOOLSET "evc4"
#  else
#    define BOOST_LIB_TOOLSET "vc6"
#  endif

#elif defined(BOOST_MSVC) && (BOOST_MSVC == 1300)

#  define BOOST_LIB_TOOLSET "vc7"

#elif defined(BOOST_MSVC) && (BOOST_MSVC == 1310)

#  define BOOST_LIB_TOOLSET "vc71"

#elif defined(BOOST_MSVC) && (BOOST_MSVC == 1400)

#  define BOOST_LIB_TOOLSET "vc80"

#elif defined(BOOST_MSVC) && (BOOST_MSVC == 1500)

#  define BOOST_LIB_TOOLSET "vc90"

#elif defined(BOOST_MSVC) && (BOOST_MSVC >= 1600)

#  define BOOST_LIB_TOOLSET "vc100"

#elif defined(__BORLANDC__)

#  define BOOST_LIB_TOOLSET "bcb"

#elif defined(__ICL)

#  define BOOST_LIB_TOOLSET "iw"

#elif defined(__MWERKS__) && (__MWERKS__ <= 0x31FF )

#  define BOOST_LIB_TOOLSET "cw8"

#elif defined(__MWERKS__) && (__MWERKS__ <= 0x32FF )

#  define BOOST_LIB_TOOLSET "cw9"

#endif
#endif 

#if defined(_MT) || defined(__MT__)
#  define BOOST_LIB_THREAD_OPT "-mt"
#else
#  define BOOST_LIB_THREAD_OPT
#endif

#if defined(_MSC_VER) || defined(__MWERKS__)

#  ifdef _DLL

#     if (defined(__SGI_STL_PORT) || defined(_STLPORT_VERSION)) && (defined(_STLP_OWN_IOSTREAMS) || defined(__STL_OWN_IOSTREAMS))

#        if defined(_DEBUG) && (defined(__STL_DEBUG) || defined(_STLP_DEBUG))
#            define BOOST_LIB_RT_OPT "-gdp"
#        elif defined(_DEBUG)
#            define BOOST_LIB_RT_OPT "-gdp"
#            pragma message("warning: STLPort debug versions are built with /D_STLP_DEBUG=1")
#            error "Build options aren't compatible with pre-built libraries"
#        else
#            define BOOST_LIB_RT_OPT "-p"
#        endif

#     elif defined(__SGI_STL_PORT) || defined(_STLPORT_VERSION)

#        if defined(_DEBUG) && (defined(__STL_DEBUG) || defined(_STLP_DEBUG))
#            define BOOST_LIB_RT_OPT "-gdpn"
#        elif defined(_DEBUG)
#            define BOOST_LIB_RT_OPT "-gdpn"
#            pragma message("warning: STLPort debug versions are built with /D_STLP_DEBUG=1")
#            error "Build options aren't compatible with pre-built libraries"
#        else
#            define BOOST_LIB_RT_OPT "-pn"
#        endif

#     else

#        if defined(_DEBUG)
#            define BOOST_LIB_RT_OPT "-gd"
#        else
#            define BOOST_LIB_RT_OPT
#        endif

#     endif

#  else

#     if (defined(__SGI_STL_PORT) || defined(_STLPORT_VERSION)) && (defined(_STLP_OWN_IOSTREAMS) || defined(__STL_OWN_IOSTREAMS))

#        if defined(_DEBUG) && (defined(__STL_DEBUG) || defined(_STLP_DEBUG))
#            define BOOST_LIB_RT_OPT "-sgdp"
#        elif defined(_DEBUG)
#             define BOOST_LIB_RT_OPT "-sgdp"
#            pragma message("warning: STLPort debug versions are built with /D_STLP_DEBUG=1")
#            error "Build options aren't compatible with pre-built libraries"
#        else
#            define BOOST_LIB_RT_OPT "-sp"
#        endif

#     elif defined(__SGI_STL_PORT) || defined(_STLPORT_VERSION)

#        if defined(_DEBUG) && (defined(__STL_DEBUG) || defined(_STLP_DEBUG))
#            define BOOST_LIB_RT_OPT "-sgdpn"
#        elif defined(_DEBUG)
#             define BOOST_LIB_RT_OPT "-sgdpn"
#            pragma message("warning: STLPort debug versions are built with /D_STLP_DEBUG=1")
#            error "Build options aren't compatible with pre-built libraries"
#        else
#            define BOOST_LIB_RT_OPT "-spn"
#        endif

#     else

#        if defined(_DEBUG)
#             define BOOST_LIB_RT_OPT "-sgd"
#        else
#            define BOOST_LIB_RT_OPT "-s"
#        endif

#     endif

#  endif

#elif defined(__BORLANDC__)

#if __BORLANDC__ > 0x561
#pragma defineonoption BOOST_BORLAND_DEBUG -v
#endif
#if defined(__STL_DEBUG) || defined(_STLP_DEBUG)
#error "Pre-built versions of the Boost libraries are not provided in STLPort-debug form"
#endif

#  ifdef _RTLDLL

#     ifdef BOOST_BORLAND_DEBUG
#         define BOOST_LIB_RT_OPT "-d"
#     else
#         define BOOST_LIB_RT_OPT
#     endif

#  else

#     ifdef BOOST_BORLAND_DEBUG
#         define BOOST_LIB_RT_OPT "-sd"
#     else
#         define BOOST_LIB_RT_OPT "-s"
#     endif

#  endif

#endif

#if (defined(_DLL) || defined(_RTLDLL)) && defined(BOOST_DYN_LINK)
#  define BOOST_LIB_PREFIX
#elif defined(BOOST_DYN_LINK)
#  error "Mixing a dll boost library with a static runtime is a really bad idea..."
#else
#  define BOOST_LIB_PREFIX "lib"
#endif

#if defined(BOOST_LIB_NAME) \
&& defined(BOOST_LIB_PREFIX) \
&& defined(BOOST_LIB_TOOLSET) \
&& defined(BOOST_LIB_THREAD_OPT) \
&& defined(BOOST_LIB_RT_OPT) \
&& defined(BOOST_LIB_VERSION)

#ifndef BOOST_AUTO_LINK_NOMANGLE
#  pragma comment(lib, BOOST_LIB_PREFIX BOOST_STRINGIZE(BOOST_LIB_NAME) "-" BOOST_LIB_TOOLSET BOOST_LIB_THREAD_OPT BOOST_LIB_RT_OPT "-" BOOST_LIB_VERSION ".lib")
#  ifdef BOOST_LIB_DIAGNOSTIC
#     pragma message ("Linking to lib file: " BOOST_LIB_PREFIX BOOST_STRINGIZE(BOOST_LIB_NAME) "-" BOOST_LIB_TOOLSET BOOST_LIB_THREAD_OPT BOOST_LIB_RT_OPT "-" BOOST_LIB_VERSION ".lib")
#  endif
#else
#  pragma comment(lib, BOOST_STRINGIZE(BOOST_LIB_NAME) ".lib")
#  ifdef BOOST_LIB_DIAGNOSTIC
#     pragma message ("Linking to lib file: " BOOST_STRINGIZE(BOOST_LIB_NAME) ".lib")
#  endif
#endif

#else
#  error "some required macros where not defined (internal logic error)."
#endif


#endif 

#ifdef BOOST_LIB_PREFIX
#  undef BOOST_LIB_PREFIX
#endif
#if defined(BOOST_LIB_NAME)
#  undef BOOST_LIB_NAME
#endif
#if defined(BOOST_LIB_THREAD_OPT)
#  undef BOOST_LIB_THREAD_OPT
#endif
#if defined(BOOST_LIB_RT_OPT)
#  undef BOOST_LIB_RT_OPT
#endif
#if defined(BOOST_LIB_LINK_OPT)
#  undef BOOST_LIB_LINK_OPT
#endif
#if defined(BOOST_LIB_DEBUG_OPT)
#  undef BOOST_LIB_DEBUG_OPT
#endif
#if defined(BOOST_DYN_LINK)
#  undef BOOST_DYN_LINK
#endif
#if defined(BOOST_AUTO_LINK_NOMANGLE)
#  undef BOOST_AUTO_LINK_NOMANGLE
#endif











