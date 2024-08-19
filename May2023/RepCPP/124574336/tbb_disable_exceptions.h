


#undef  TBB_USE_EXCEPTIONS
#define TBB_USE_EXCEPTIONS 0


#if _MSC_VER
#if __INTEL_COMPILER
#pragma warning (disable: 583)
#else
#pragma warning (disable: 4530 4577)
#endif
#endif
