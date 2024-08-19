

#if defined(__COMO__)


#elif defined(__DMC__)


#elif defined(__INTEL_COMPILER) || defined(__ICL) \
|| defined(__ICC) || defined(__ECC)


# if (__GNUC__ == 4 && __GNUC_MINOR__ >= 1) || (__GNUC__ > 4)
#  if !defined(BOOST_ASIO_DISABLE_VISIBILITY)
#   pragma GCC visibility pop
#  endif 
# endif 

#elif defined(__clang__)


# if defined(__OBJC__)
#  if !defined(__APPLE_CC__) || (__APPLE_CC__ <= 1)
#   if defined(BOOST_ASIO_OBJC_WORKAROUND)
#    undef Protocol
#    undef id
#    undef BOOST_ASIO_OBJC_WORKAROUND
#   endif
#  endif
# endif

# if !defined(_WIN32) && !defined(__WIN32__) && !defined(WIN32)
#  if !defined(BOOST_ASIO_DISABLE_VISIBILITY)
#   pragma GCC visibility pop
#  endif 
# endif 

# pragma GCC diagnostic pop

#elif defined(__GNUC__)


# if defined(__MINGW32__) || defined(__CYGWIN__)
#  pragma pack (pop)
# endif

# if defined(__OBJC__)
#  if !defined(__APPLE_CC__) || (__APPLE_CC__ <= 1)
#   if defined(BOOST_ASIO_OBJC_WORKAROUND)
#    undef Protocol
#    undef id
#    undef BOOST_ASIO_OBJC_WORKAROUND
#   endif
#  endif
# endif

# if (__GNUC__ == 4 && __GNUC_MINOR__ >= 1) || (__GNUC__ > 4)
#  if !defined(BOOST_ASIO_DISABLE_VISIBILITY)
#   pragma GCC visibility pop
#  endif 
# endif 

# pragma GCC diagnostic pop

#elif defined(__KCC)


#elif defined(__sgi)


#elif defined(__DECCXX)


#elif defined(__ghs)


#elif defined(__BORLANDC__) && !defined(__clang__)


# pragma option pop
# pragma nopushoptwarn
# pragma nopackwarning

#elif defined(__MWERKS__)


#elif defined(__SUNPRO_CC)


#elif defined(__HP_aCC)


#elif defined(__MRC__) || defined(__SC__)


#elif defined(__IBMCPP__)


#elif defined(_MSC_VER)


# pragma warning (pop)
# pragma pack (pop)

# if defined(__cplusplus_cli) || defined(__cplusplus_winrt)
#  if defined(BOOST_ASIO_CLR_WORKAROUND)
#   undef generic
#   undef BOOST_ASIO_CLR_WORKAROUND
#  endif
# endif

#endif
