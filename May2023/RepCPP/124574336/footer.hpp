

#if !defined(BOOST_WINAPI_ENABLE_WARNINGS)

#if defined(_MSC_VER) && !(defined(__INTEL_COMPILER) || defined(__clang__))

#pragma warning(pop)

#elif defined(__GNUC__) && !(defined(__INTEL_COMPILER) || defined(__ICL) || defined(__ICC) || defined(__ECC)) \
&& (__GNUC__ * 100 + __GNUC_MINOR__) >= 406

#pragma GCC diagnostic pop

#endif

#endif 
