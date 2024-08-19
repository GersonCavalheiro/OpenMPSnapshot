#ifndef MJOLNIR_UTIL_MACRO_HPP
#define MJOLNIR_UTIL_MACRO_HPP

#define MJOLNIR_STRINGIZE_AUX(x) #x
#define MJOLNIR_STRINGIZE(x)     MJOLNIR_STRINGIZE_AUX(x)

#if defined(__GNUC__)
#  define MJOLNIR_FUNC_NAME __PRETTY_FUNCTION__
#elif defined(_MSC_VER)
#  define MJOLNIR_FUNC_NAME __FUNCSIG__
#else
#  define MJOLNIR_FUNC_NAME __func__
#endif

#if !defined(MJOLNIR_WITH_LITTLE_ENDIAN) && !defined(MJOLNIR_WITH_BIG_ENDIAN)
#  ifdef __GNUC__
#    if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
#      define MJOLNIR_WITH_BIG_ENDIAN
#    elif __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
#      define MJOLNIR_WITH_LITTLE_ENDIAN
#    else
#      error "Mjolnir supports only little or big endian."
#    endif
#  elif defined(_MSC_VER)
#      pragma message("Now compiling on Windows, assuming little endian...")
#      define MJOLNIR_WITH_LITTLE_ENDIAN
#  else
#    error "Unknown platform. Please define MJOLNIR_WITH_LITTLE_ENDIAN or MJOLNIR_WITH_BIG_ENDIAN"
#  endif
#endif

#endif 
