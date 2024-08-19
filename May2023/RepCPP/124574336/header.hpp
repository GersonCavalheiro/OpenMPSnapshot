

#if !defined(BOOST_WINAPI_ENABLE_WARNINGS)

#if defined(_MSC_VER) && !(defined(__INTEL_COMPILER) || defined(__clang__))

#pragma warning(push, 3)
#pragma warning(disable: 4201)
#pragma warning(disable: 28251)

#elif defined(__GNUC__) && !(defined(__INTEL_COMPILER) || defined(__ICL) || defined(__ICC) || defined(__ECC)) \
&& (__GNUC__ * 100 + __GNUC_MINOR__) >= 406

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wlong-long"

#endif

#endif 
