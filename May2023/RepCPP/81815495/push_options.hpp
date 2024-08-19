

#if defined(__COMO__)


#elif defined(__DMC__)


#elif defined(__INTEL_COMPILER) || defined(__ICL) \
|| defined(__ICC) || defined(__ECC)


# if (__GNUC__ == 4 && __GNUC_MINOR__ >= 1) || (__GNUC__ > 4)
#  if !defined(ASIO_DISABLE_VISIBILITY)
#   pragma GCC visibility push (default)
#  endif 
# endif 

#elif defined(__clang__)


# if defined(__OBJC__)
#  if !defined(__APPLE_CC__) || (__APPLE_CC__ <= 1)
#   if !defined(ASIO_DISABLE_OBJC_WORKAROUND)
#    if !defined(Protocol) && !defined(id)
#     define Protocol cpp_Protocol
#     define id cpp_id
#     define ASIO_OBJC_WORKAROUND
#    endif
#   endif
#  endif
# endif

# if !defined(_WIN32) && !defined(__WIN32__) && !defined(WIN32)
#  if !defined(ASIO_DISABLE_VISIBILITY)
#   pragma GCC visibility push (default)
#  endif 
# endif 

# pragma GCC diagnostic push
# pragma GCC diagnostic ignored "-Wnon-virtual-dtor"
# if (__clang_major__ >= 6)
#  pragma GCC diagnostic ignored "-Wzero-as-null-pointer-constant"
# endif 

# pragma push_macro ("emit")
# undef emit

# pragma push_macro ("signal")
# undef signal

# pragma push_macro ("slot")
# undef slot

#elif defined(__GNUC__)


# if defined(__MINGW32__) || defined(__CYGWIN__)
#  pragma pack (push, 8)
# endif

# if defined(__OBJC__)
#  if !defined(__APPLE_CC__) || (__APPLE_CC__ <= 1)
#   if !defined(ASIO_DISABLE_OBJC_WORKAROUND)
#    if !defined(Protocol) && !defined(id)
#     define Protocol cpp_Protocol
#     define id cpp_id
#     define ASIO_OBJC_WORKAROUND
#    endif
#   endif
#  endif
# endif

# if (__GNUC__ == 4 && __GNUC_MINOR__ >= 1) || (__GNUC__ > 4)
#  if !defined(ASIO_DISABLE_VISIBILITY)
#   pragma GCC visibility push (default)
#  endif 
# endif 

# pragma GCC diagnostic push
# pragma GCC diagnostic ignored "-Wnon-virtual-dtor"
# if (__GNUC__ == 4 && __GNUC_MINOR__ >= 7) || (__GNUC__ > 4)
#  pragma GCC diagnostic ignored "-Wzero-as-null-pointer-constant"
# endif 
# if (__GNUC__ >= 7)
#  pragma GCC diagnostic ignored "-Wimplicit-fallthrough"
# endif 

# pragma push_macro ("emit")
# undef emit

# pragma push_macro ("signal")
# undef signal

# pragma push_macro ("slot")
# undef slot

#elif defined(__KCC)


#elif defined(__sgi)


#elif defined(__DECCXX)


#elif defined(__ghs)


#elif defined(__BORLANDC__) && !defined(__clang__)


# pragma option push -a8 -b -Ve- -Vx- -w-inl -vi-
# pragma nopushoptwarn
# pragma nopackwarning
# if !defined(__MT__)
#  error Multithreaded RTL must be selected.
# endif 

#elif defined(__MWERKS__)


#elif defined(__SUNPRO_CC)


#elif defined(__HP_aCC)


#elif defined(__MRC__) || defined(__SC__)


#elif defined(__IBMCPP__)


#elif defined(_MSC_VER)


# pragma warning (disable:4103)
# pragma warning (push)
# pragma warning (disable:4127)
# pragma warning (disable:4180)
# pragma warning (disable:4244)
# pragma warning (disable:4355)
# pragma warning (disable:4510)
# pragma warning (disable:4512)
# pragma warning (disable:4610)
# pragma warning (disable:4675)
# if (_MSC_VER < 1600)
#  pragma warning (disable:4100)
# endif 
# if defined(_M_IX86) && defined(_Wp64)
#  pragma warning (disable:4311)
#  pragma warning (disable:4312)
# endif 
# pragma pack (push, 8)
# if (_MSC_VER < 1300)
#  pragma optimize ("g", off)
# endif
# if !defined(_MT)
#  error Multithreaded RTL must be selected.
# endif 

# if defined(__cplusplus_cli) || defined(__cplusplus_winrt)
#  if !defined(ASIO_DISABLE_CLR_WORKAROUND)
#   if !defined(generic)
#    define generic cpp_generic
#    define ASIO_CLR_WORKAROUND
#   endif
#  endif
# endif

# pragma push_macro ("emit")
# undef emit

# pragma push_macro ("signal")
# undef signal

# pragma push_macro ("slot")
# undef slot

#endif
